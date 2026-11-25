"""Image overlays and a compact report for held-out component validation."""
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.geometry import camera_to_world_numpy


def project_world(points, width=300, height=300):
    origin = camera_to_world_numpy(np.zeros((1,3),dtype=np.float32))[0]
    basis = camera_to_world_numpy(np.eye(3,dtype=np.float32))-origin
    camera = (np.asarray(points,dtype=np.float32)-origin) @ np.linalg.inv(basis)
    depth = camera[...,2]
    focal = width/(2*np.tan(np.pi/6))
    xy = camera[...,:2]/np.maximum(depth[...,None],1e-6)*focal+[width/2,height/2]
    valid = np.isfinite(xy).all(-1) & (depth>1e-6)
    valid &= (xy[...,0]>=0)&(xy[...,0]<width)&(xy[...,1]>=0)&(xy[...,1]<height)
    return xy,valid


def render_example(path, batch, output, row, sampled_actions=None):
    """One held-out sample, already forwarded without target goals/phase inputs."""
    def array(value):return value.detach().float().cpu().numpy()
    rgb = array(batch['rgb'][0])
    if rgb.ndim==4:rgb=rgb[-1]
    image = Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0)*255,0,255)))
    width,height = image.size
    panels = [image.copy() for _ in range(3)]
    draws = [ImageDraw.Draw(panel) for panel in panels]
    phase,predicted_phase = row['reference_phase'],row['predicted_phase']
    camera = array(batch['agentview_pcd'][0])[:,:3]
    focal = width/(2*np.tan(np.pi/6))
    xy = camera[:,:2]/np.maximum(camera[:,2:3],1e-6)*focal+[width/2,height/2]
    visible = np.isfinite(xy).all(-1)&(camera[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    labels = array(batch['point_part_labels'][0])>.5
    weights = array(output.grounding_bank[phase].weights[0])
    for i in np.flatnonzero(labels&visible)[::max(1,int(labels.sum())//150)]:
        x,y=xy[i];draws[0].ellipse((x-1,y-1,x+1,y+1),fill='#00ff60')
    for i in np.argsort(weights)[-128:]:
        if visible[i]:
            x,y=xy[i];draws[1].ellipse((x-2,y-2,x+2,y+2),fill='#ff9838')

    def trajectory(draw, points, color):
        pixels,valid = project_world(points,width,height)
        for i in range(1,len(pixels)):
            if valid[i-1] and valid[i]:draw.line([tuple(pixels[i-1]),tuple(pixels[i])],fill=color,width=2)
        for pixel,okay in zip(pixels,valid):
            if okay:
                x,y=pixel;draw.ellipse((x-2,y-2,x+2,y+2),fill=color)

    def pose(draw, position, color, rotation=None):
        points = np.asarray(position)[None]
        pixel,valid = project_world(points,width,height)
        if valid[0]:
            x,y=pixel[0];draw.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3)
        if rotation is not None:
            for axis in range(3):
                trajectory(draw,np.stack([position,position+.035*rotation[:,axis]]),['#ff3030','#20ff50','#4080ff'][axis])
    from models.pose_token_diffusion.blocks.geometry import quaternion_matrix
    target = array(batch['goal_poses'][0,phase])
    target_rotation = array(quaternion_matrix(batch['goal_poses'][0,phase,:4])) if row['rotation_valid'] else None
    if row['position_valid']:
        pose(draws[0],target[4:7],'#00ff60',target_rotation)
        pose(draws[1],target[4:7],'#00ff60')
    pose(draws[1],array(output.predicted_goals.position_world[0,phase]),'#ff9838',
        array(output.predicted_goals.rotation_world[0,phase]) if row['rotation_valid'] else None)
    pose(draws[2],array(output.predicted_goals.position_world[0,predicted_phase]),'#e480ff')
    trajectory(draws[2],array(batch['action_chunk'][0])[:,:3],'#00ff60')
    if sampled_actions is not None:trajectory(draws[2],array(sampled_actions[0])[:,:3],'#e480ff')
    canvas=Image.new('RGB',(width*3,height+120),'#15202b')
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    except OSError:font=ImageFont.load_default()
    titles=['Reference target/part (green)','Annotated-slot prediction (orange)','Selected goal / sampled actions (purple)']
    for i,panel in enumerate(panels):
        canvas.paste(panel,(width*i,24));draw.text((width*i+5,5),titles[i],fill='white',font=font)
    error='masked' if row['goal_position_cm'] is None else f"{row['goal_position_cm']:.2f} cm"
    lines=[f"{row['object']} / {row['demo']} / t={row['t']} / {row['skill']} | reference phase {phase}, predicted phase {predicted_phase}",
        f"Goal position error: {error}. Green trajectory: recorded commands. Purple: sampled commands, not a simulator rollout.",
        'Part labels and reference phase are diagnostic annotations only; they were not policy inputs.',
        'Touch targets are geometry proxies. Markers outside the image are omitted; numeric values remain in samples.csv.']
    for i,line in enumerate(lines):draw.text((5,height+30+i*20),line,fill='white',font=font)
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);canvas.save(path)


def write_index(directory):
    directory=Path(directory)
    cards='\n'.join(f'<figure><img width="100%" src="{html.escape(str(p.relative_to(directory)))}"><figcaption>{html.escape(p.stem)}</figcaption></figure>'
        for p in sorted((directory/'examples').glob('*.png')))
    text=(directory/'README.md').read_text()
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V2 component validation</title>'
        '<style>body{font:15px sans-serif;max-width:1100px;margin:30px auto;background:#f5f7fa}pre{white-space:pre-wrap}figure{margin:20px 0}</style>'
        '<a href="summary.csv">Summary CSV</a> | <a href="samples.csv">Per-sample CSV</a> | <a href="metrics.json">All metrics</a>'
        '<pre>'+html.escape(text)+'</pre>'+cards)


def write_history(directory, history):
    directory=Path(directory)
    (directory/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
    lines=['# V2 pilot training progress','','Frozen pretrained DINOv2/T5; fresh trainable policy; predicted pose input.',
        'Epoch 0 is untrained-policy validation. These metrics are not physical task success.','',
        '| Epoch | Training loss | Validation loss | Goal position mean cm | Phase accuracy | Details |',
        '| --- | --- | --- | --- | --- | --- |']
    def fmt(value):return 'N/A' if value is None else f'{value:.3f}'
    for row in history:
        lines.append(f"| {row['epoch']} | {fmt(row.get('total'))} | {fmt(row['validation_loss'])} | {fmt(row['goal_position_cm'])} | {fmt(row['phase_accuracy'])} | [Report]({row['report']}/index.html) |")
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
