"""Matched checkpoint initialization and real-data backward preflight."""
import argparse
import gc
from pathlib import Path

import torch

from _common import ROOT, load_yaml, resolve_paths, save_json, set_seed
from train import build_dataset, build_dataloader, build_model, compute_losses, curriculum_state, forward_kwargs, freeze_prefixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    results = {}
    for variant in ["control","contact"]:
        source = ROOT/'configs/architecture_update'/f'hierarchical_world_geometry_10epoch_normalized_global_{variant}.yaml'
        cfg = resolve_paths(load_yaml(source), ROOT); set_seed(42)
        cfg['data'].update(max_demos_per_file=1,batch_size=2,num_workers=0)
        dataset = build_dataset(cfg['data'])
        batch = next(iter(build_dataloader(dataset,cfg['data'])))
        batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        model = build_model(cfg).cuda()
        state = torch.load(cfg['train']['init_checkpoint'],map_location='cpu',weights_only=False)
        missing, unexpected = model.load_state_dict(state['model'],strict=False)
        learned_missing = [k for k in missing if not k.startswith(('visual.model.','text.model.'))]
        assert learned_missing == ['contact_geometry_conditioner.contact_weight'] and not unexpected
        freeze_prefixes(model,cfg['train']['freeze_prefixes'])
        module = model.contact_geometry_conditioner
        features = torch.randn(2,4,cfg['model']['hidden_dim'],device='cuda')
        contact = torch.randn(2,3,device='cuda');tcp = torch.randn(2,7,device='cuda')
        expected = features+module.projection(torch.zeros(2,3,device='cuda'))[:,None,:]
        assert torch.equal(module(features,contact,tcp,enabled=False)[0],expected)
        optimizer = torch.optim.AdamW(model.trainable_parameters(),lr=cfg['train']['lr'])
        model.train(); _,probability,weights = curriculum_state(0,cfg)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(batch['rgb'],batch['task_instruction'],target_action=batch['action_chunk'],
                      **forward_kwargs(batch,True,probability,True))
            total,parts=compute_losses(out,batch,weights,set(cfg['heads']))
        assert torch.isfinite(total)
        total.backward()
        for name,p in model.named_parameters():
            if p.grad is not None: assert torch.isfinite(p.grad).all(),name
        grad=float(module.contact_weight.grad)
        assert grad != 0 if variant=='contact' else grad == 0
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()),cfg['train']['grad_clip'])
        optimizer.step()
        results[variant]={'loss':float(total),'weight_after_step':float(module.contact_weight),
                          'weight_gradient':grad,'control_bias_preserved':True,
                          'missing_new_parameters':learned_missing,'finite_gradients':True,
                          'checkpoint':cfg['train']['init_checkpoint'],'device':torch.cuda.get_device_name(0)}
        save_json(args.out_dir/'verification.json',results)
        print('PASS',variant,results[variant],flush=True)
        del model,optimizer,out,total,batch,dataset,state
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
