"""Joint v2 training with per-skill, held-out component reports each epoch."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data.partinstruct_loader import collate_part2action
from diagnostics.evaluation_protocol import execution_settings, DEFAULT_PROTOCOL_PATH
from .checkpoint import checkpoint_state, save_checkpoint
from .config import load_config
from .dataset import build_dataset, observation_from_batch, provided_goals_from_batch
from .losses import compute_losses
from .metrics import component_rows, write_report
from .reporting import render_example, write_index, write_history
from .policy import build_policy
from .completion import CompletionDataset, BalancedCompletionSampler, completion_loss, validate_completion

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def source_paths():
    paths = list(PACKAGE.rglob('*.py'))+list((PACKAGE/'configs').glob('*.yaml'))
    paths += list(PACKAGE.with_name('pose_token_diffusion').rglob('*.py'))
    paths += [ROOT/'part2action'/name for name in (
        'data/partinstruct_loader.py','data/targets.py','data/geometry.py',
        'scripts/train_pose_token_diffusion_v2.py','scripts/run_pose_token_diffusion_v2.py',
        'models/backbone.py','models/heads.py','models/hierarchical.py','models/temporal.py',
        'models/diffusion_transformer.py','models/part_grounded_diffusion/verify_sources.py',
        'models/part_grounded_diffusion/preserved_sources.json','diagnostics/evaluation_protocol.py',
        'slurm/pose_token_diffusion_v2.sbatch')]
    paths += [ROOT/'part2action'/name for name in (
        'slurm/evaluate_pose_token_v2_development.sbatch','scripts/run_pose_token_v2_development.py')]
    paths += [ROOT/'PartInstruct/PartInstruct/PartGym/env/backend/planner'/name
              for name in ('panda_arm.py','bullet_planner.py')]
    paths.append(DEFAULT_PROTOCOL_PATH)
    return sorted(set(paths))


def forward_batch(model, batch):
    return model(observation_from_batch(batch),target_actions=batch['action_chunk'],
        provided_goals=provided_goals_from_batch(batch) if model.config.pose_source=='provided' else None)


def train_step(model, batch, optimizer, weights=None, *, amp=False, grad_clip=1., collect_gradients=False,
               completion_batch=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch['rgb'].device.type,dtype=torch.bfloat16,enabled=amp):
        output = forward_batch(model,batch)
        loss,metrics = compute_losses(output,batch,model,
            {**(weights or {}),'termination_weight':0.} if completion_batch is not None else weights)
        if completion_batch is not None:
            encoded = model.blocks['encoder'](observation_from_batch(completion_batch))
            terminal = completion_loss(encoded.hierarchy, completion_batch)
            loss = loss + float((weights or {}).get('termination_weight',1.))*terminal
            metrics = {**metrics,'termination':terminal.detach(),'total':loss.detach()}
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite v2 loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],grad_clip)
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite v2 gradients')
    gradients = {name:float(p.grad.float().norm()) if p.grad is not None else 0.
        for name,p in model.named_parameters() if p.requires_grad} if collect_gradients else {}
    optimizer.step()
    return {name:float(value) for name,value in metrics.items()},gradients


def move_batch(batch, device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def representative_indices(dataset, count=16):
    """Training/validation smoke examples spanning real object and skill phases."""
    index = {(s.file_path,s.demo_key,s.t):i for i,s in enumerate(dataset._samples)}
    selected, kinds, objects, seen = [],set(),set(),set()
    for spec in dataset._samples:
        key = spec.file_path,spec.demo_key
        if key in seen:continue
        seen.add(key)
        for phase in dataset.targets_for(spec):
            sample = index.get((*key,phase['start']))
            obj = Path(spec.file_path).stem
            if sample is not None and phase['component_valid'].any() and (phase['kind'] not in kinds or obj not in objects):
                selected.append(sample);kinds.add(phase['kind']);objects.add(obj)
        if len(kinds)>=5 and len(objects)==len(dataset.hdf5_paths):break
    if not selected:raise ValueError('No valid representative v2 smoke observations')
    selected += list(range(min(len(dataset),count)))
    return list(dict.fromkeys(selected))[:count]


def validate(model, loader, weights, directory, *, epoch, device, amp=False,
             max_batches=None, examples=5, seed=1042, completion_only_supervision=False):
    previous_mode = model.training
    model.eval()
    rows, totals, samples, images = [],{},0,[]
    example_skills = set()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices),torch.inference_mode():
        torch.manual_seed(seed)
        for batch_index,raw in enumerate(loader):
            if max_batches is not None and batch_index>=max_batches:break
            batch = move_batch(raw,device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                output = forward_batch(model,batch)
                loss,parts = compute_losses(output,batch,model,weights)
            if not torch.isfinite(loss):raise ValueError('Nonfinite held-out validation loss')
            current = component_rows(output,batch)
            if completion_only_supervision:
                for row in current:row['termination_valid']=False
            rows.extend(current)
            samples += len(current)
            for key,value in parts.items():totals[key]=totals.get(key,0.)+float(value)*len(current)
            for i,row in enumerate(current):
                if len(images)>=examples or row['skill'] in example_skills or not row['position_valid']:continue
                single = {k:v[i:i+1] for k,v in batch.items()}
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                    sampled = model(observation_from_batch(single))
                actions = sampled.actions['action_chunk']
                if not torch.isfinite(actions).all():raise ValueError('Nonfinite held-out action sample')
                images.append((single,sampled,row,actions))
                example_skills.add(row['skill'])
    if not rows:raise ValueError('No held-out validation observations')
    scope = ('Whole-demo held-out development observations; every selected frame is evaluated.' if max_batches is None
             else f'Bounded development validation: at most {max_batches} batches, not full coverage.')
    result = write_report(directory,rows,epoch=epoch,losses={k:v/samples for k,v in totals.items()},scope=scope)
    for batch,output,row,actions in images:
        render_example(Path(directory)/'examples'/f"{row['object']}_{row['demo']}_{row['t']:04d}_{row['skill']}.png",batch,output,row,actions)
    write_index(directory)
    model.train(previous_mode)
    return result


def verify_plan(path):
    plan = json.loads(Path(path).read_text())
    for name,wanted in plan['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=wanted:
            raise ValueError('Pinned pilot source changed: '+name)
    for name,wanted in plan['data_files'].items():
        stat = Path(name).stat()
        if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=wanted:
            raise ValueError('Pilot training data changed: '+name)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PACKAGE/'configs/train.yaml')
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--expected-plan',type=Path)
    parser.add_argument('--smoke-receipt',type=Path)
    args = parser.parse_args()
    plan = verify_plan(args.expected_plan) if args.expected_plan else None
    recipe = yaml.safe_load(args.config.read_text())
    if set(recipe)-{'completion'}!={'model_config','data','goal_labels','train','losses'}:
        raise ValueError('Expected the independent v2 recipe')
    options = recipe['train']
    if options['initialization']!='random_policy_frozen_backbones' or 'init_checkpoint' in options:
        raise ValueError('The v2 pilot requires explicit fresh-policy initialization')
    limit = options.get('validation_max_batches')
    if int(options['epochs'])<1 or (limit is not None and (type(limit) is not int or limit<1)):
        raise ValueError('Epoch count and optional validation batch limit must be positive')
    model_path = args.config.resolve().parent/recipe['model_config']
    config = load_config(model_path)
    if config.pose_source!='predicted':raise ValueError('V2 pilot uses predicted goals throughout')
    data = dict(recipe['data'])
    if options.get('require_demo_manifest') and not data.get('demo_manifest'):
        raise ValueError('Prepare the balanced pilot manifest before starting this recipe')
    if data.get('split','train')!='train':raise ValueError('Training must use the training demo split')
    if (data['action_chunk'],data['n_obs_steps'],data['max_skill_slots'])!=(config.action_horizon,config.n_obs_steps,config.max_skill_slots):
        raise ValueError('Model/data contracts differ')
    if args.out_dir.exists():raise FileExistsError('Choose a fresh v2 output directory')
    smoke = None
    if args.smoke_receipt:
        smoke = json.loads(args.smoke_receipt.read_text())
        if smoke['status']!='passed' or smoke['config_sha256']!=hashlib.sha256(args.config.read_bytes()).hexdigest():
            raise ValueError('Training needs a passed smoke for the exact pilot recipe')
        if plan and smoke['submission_plan_sha256']!=hashlib.sha256(args.expected_plan.read_bytes()).hexdigest():
            raise ValueError('Smoke receipt differs from this pinned submission')
    if not torch.cuda.is_available():raise RuntimeError('Use a Slurm GPU allocation for pretrained-backbone training')
    seed = int(options['seed'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    dataset = build_dataset(data,recipe['goal_labels'])
    validation = build_dataset({**data,'split':'validation'},recipe['goal_labels'])
    train_ids = set(map(tuple,dataset.split_info['demonstrations']))
    val_ids = set(map(tuple,validation.split_info['demonstrations']))
    if not train_ids or not val_ids or train_ids & val_ids:raise ValueError('Invalid held-out demonstration split')
    train_info,val_info = dataset.split_info,validation.split_info
    if args.smoke:
        dataset = Subset(dataset,representative_indices(dataset,data['batch_size']))
        validation = Subset(validation,representative_indices(validation,data['batch_size']))
    for ds in (dataset,validation):(ds.dataset if isinstance(ds,Subset) else ds).close()
    workers = 0 if args.smoke else data['num_workers']
    loader = DataLoader(dataset,batch_size=min(data['batch_size'],len(dataset)),shuffle=True,
        collate_fn=collate_part2action,num_workers=workers,pin_memory=data['pin_memory'],
        drop_last=not args.smoke,generator=torch.Generator().manual_seed(seed),
        **({'multiprocessing_context':'spawn','persistent_workers':True} if workers else {}))
    val_loader = DataLoader(validation,batch_size=data['batch_size'],shuffle=False,num_workers=0,
        collate_fn=collate_part2action,generator=torch.Generator().manual_seed(seed+1000))
    completion_train = completion_validation = completion_loader = completion_val_loader = None
    if recipe.get('completion'):
        completion_options = recipe['completion']
        if set(completion_options) != {'post_transition_steps','allow_endpoint_proxy'} or not data.get('demo_manifest'):
            raise ValueError('Completion pilot needs explicit options and the pinned demonstration manifest')
        completion_train = CompletionDataset(data['demo_manifest'],split='train',
            n_obs_steps=config.n_obs_steps,**completion_options)
        completion_validation = CompletionDataset(data['demo_manifest'],split='validation',
            n_obs_steps=config.n_obs_steps,**completion_options)
        if args.smoke:
            # Include terminal positives, post-transition positives and negatives.
            ids=[]
            for kind in sorted({r['skill'] for r in completion_validation._samples}):
                for label in (0,1):
                    ids += [next(i for i,r in enumerate(completion_validation._samples)
                        if r['skill']==kind and r['target']==label)]
            completion_val_data = Subset(completion_validation,ids)
        else:
            completion_val_data = completion_validation
        completion_loader = DataLoader(completion_train,batch_size=4,
            sampler=BalancedCompletionSampler(completion_train,len(loader),seed+2000),
            collate_fn=collate_part2action,num_workers=0,
            generator=torch.Generator().manual_seed(seed+2001))
        completion_val_loader = DataLoader(completion_val_data,batch_size=data['batch_size'],
            collate_fn=collate_part2action,num_workers=0,
            generator=torch.Generator().manual_seed(seed+3000))
    if not len(loader):raise ValueError('No complete pilot training batch')
    args.out_dir.mkdir(parents=True)
    write(args.out_dir/'config_resolved.json',{'model':config.to_dict(),**recipe,'smoke':args.smoke,
        'execution_settings':execution_settings('test1')})
    write(args.out_dir/'data_split.json',{'train':train_info,'validation':val_info,'overlap':0})
    if completion_train is not None:
        write(args.out_dir/'completion_labels.json',{'train':completion_train.audit,
            'validation':completion_validation.audit,'action_sampling':'Unchanged main batches; four balanced completion-only queries per update. No action targets on completion-only observations.'})
    paths = source_paths()+[args.config.resolve(),model_path.resolve()]
    if data.get('demo_manifest'):paths.append(Path(data['demo_manifest']).resolve())
    snapshot = args.out_dir/'source_snapshot';snapshot.mkdir()
    for i,path in enumerate(sorted(set(paths))):shutil.copy2(path,snapshot/f'{i:03d}_{path.name}')
    write(args.out_dir/'source_hashes.json',{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps({'stage':'loading_frozen_backbones','smoke':args.smoke,'train_samples':len(dataset),
        'validation_samples':len(validation),'gpu':torch.cuda.get_device_name()}),flush=True)
    model = build_policy(config)
    digest = hashlib.sha256()
    for name,value in sorted(checkpoint_state(model).items()):
        digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    initial = digest.hexdigest()
    if smoke and initial!=smoke['initial_trainable_state_sha256']:
        raise ValueError('Smoke and training fresh initializations differ')
    if options.get('reference_initialization'):
        reference=json.loads(Path(options['reference_initialization']).read_text())
        if initial!=reference['initial_trainable_state_sha256']:
            raise ValueError('Corrected pilot must retain the original random initialization')
    write(args.out_dir/'initialization.json',{'mode':'random_policy_frozen_backbones','checkpoint_loaded':False,
        'seed':seed,'initial_trainable_state_sha256':initial,'visual_backbone':config.visual_backbone,
        'language_backbone':config.language_backbone,'frozen_backbones_trainable':False})
    model.cuda()
    device = torch.device('cuda')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=options['lr'],weight_decay=options['weight_decay'])
    history = []

    def validation_row(epoch):
        report_path = f'validation/epoch_{epoch:03d}'
        result = validate(model,val_loader,
            {**recipe['losses'],'termination_weight':0.} if completion_val_loader is not None else recipe['losses'],
            args.out_dir/report_path,epoch=epoch,
            device=device,amp=options['amp'],max_batches=1 if args.smoke else limit,
            examples=options.get('validation_examples',5),seed=seed+1000,
            completion_only_supervision=completion_val_loader is not None)
        completion_metrics = validate_completion(model,completion_val_loader,
            args.out_dir/'completion'/f'epoch_{epoch:03d}') if completion_val_loader is not None else None
        return {'validation_loss':result['losses']['total'],
            'goal_position_cm':result['overall']['goal_position_cm']['mean'],
            'phase_accuracy':result['overall']['phase_correct']['mean'],'report':report_path,
            'validation_samples':result['overall']['samples'],
            **({'completion':completion_metrics} if completion_metrics is not None else {})}

    if not args.smoke:
        history.append({'epoch':0,'optimizer_steps':0,**validation_row(0)})
        write_history(args.out_dir,history)
        print(json.dumps({'stage':'untrained_validation',**history[-1]}),flush=True)
    cumulative = 0
    for epoch in range(1,1+(1 if args.smoke else int(options['epochs']))):
        totals,order,seen = {},hashlib.sha256(),0
        completion_iterator = iter(completion_loader) if completion_loader is not None else None
        for count,raw in enumerate(loader,1):
            order.update(repr(raw['meta']).encode())
            metrics,gradients = train_step(model,move_batch(raw,device),optimizer,recipe['losses'],
                amp=options['amp'],grad_clip=options['grad_clip'],collect_gradients=args.smoke,
                completion_batch=move_batch(next(completion_iterator),device) if completion_iterator is not None else None)
            n=len(raw['meta']);seen+=n
            for key,value in metrics.items():totals[key]=totals.get(key,0.)+value*n
            if count%options.get('log_every_steps',50)==0 or count==1:
                print(json.dumps({'epoch':epoch,'step':count,'steps_per_epoch':len(loader),**metrics}),flush=True)
            if args.smoke:break
        cumulative += count
        row={'epoch':epoch,'optimizer_steps':count,'batch_order_sha256':order.hexdigest(),
            **{key:value/seen for key,value in totals.items()},**validation_row(epoch)}
        history.append(row)
        save_checkpoint(model,args.out_dir/f'epoch_{epoch:03d}.pt',training=row)
        write_history(args.out_dir,history)
        print(json.dumps({'stage':'epoch_completed',**row}),flush=True)
        if args.smoke:
            for name in ('position','rotation','opening','skill','applicability'):
                if gradients.get(f'blocks.goal_head.{name}.weight',0.)<=0:
                    raise ValueError('Missing smoke gradient into goal head '+name)
            write(args.out_dir/'smoke.json',{'status':'passed','losses':metrics,'gradient_norms':gradients,
                'initial_trainable_state_sha256':initial,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
                'submission_plan_sha256':hashlib.sha256(args.expected_plan.read_bytes()).hexdigest() if args.expected_plan else None,
                'scope':'One pretrained-backbone GPU update, held-out component reporting and action sampling; not a performance result.'})
    write(args.out_dir/'completion.json',{'status':'completed','smoke':args.smoke,
        'epochs':len(history)-(not args.smoke),'optimizer_steps':cumulative,'validation_reports':[r['report'] for r in history]})
    for ds in (dataset,validation):(ds.dataset if isinstance(ds,Subset) else ds).close()
    for ds in (completion_train,completion_validation):
        if ds is not None:ds.close()


if __name__=='__main__':main()
