"""Export four unsuccessful DP3 pilot rollouts with verified log outcomes."""
from pathlib import Path
import json
import cv2
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2]
source = root/'results/partinstruct/dp3_eval/media/1500/scissors/split-test1'
out = root/'results/partinstruct/visualizations/dp3_failures'
out.mkdir(parents=True, exist_ok=True)
logs = json.loads((source/'rollout_logs.json').read_text())
metadata = json.loads((source/'video_meta.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
trials=[(1,0,'touch_screw','Touch the scissors screw'),
        (5,2,'grasp_and_move','Grasp top; move up, back, down'),
        (7,3,'grasp_move_release','Grasp top; move forward; release'),
        (9,4,'grasp_and_move_twice','Grasp left; move up, forward')]
records=[]
for task,index,slug,title in trials:
 prefix=f'test1/{task}/env_{index}/'
 assert logs[prefix+f'success_{index}']==0
 steps=logs[prefix+f'steps_{index}'];completion=logs[prefix+f'completion_rate_{index}'];episode=logs[prefix+f'ep_id_{index}']
 filename=f'task_type-{task}_env-{index+1}.mp4'
 meta=next(m for m in metadata if m['video_file']==filename)
 cap=cv2.VideoCapture(str(source/filename));fps=cap.get(cv2.CAP_PROP_FPS)
 frames=[];clean=[];count=0
 while True:
  ok,bgr=cap.read()
  if not ok:break
  count+=1
  # Keep every other frame for smaller report files; preserve source playback speed.
  if (count-1)%2:continue
  rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));rgb.thumbnail((640,640),Image.Resampling.LANCZOS);clean.append(rgb)
  canvas=Image.new('RGB',(640,rgb.height+100),'#111923');canvas.paste(rgb,((640-rgb.width)//2,70));d=ImageDraw.Draw(canvas)
  d.text((12,8),'DP3-S: task not completed',font=font,fill='#ff8b80')
  d.text((12,35),title+' | test1 | epoch 1500',font=small,fill='white')
  d.text((12,rgb.height+77),f'GT part input | episode {episode} | {steps} steps | completion {completion:.0%}',font=small,fill='white')
  frames.append(canvas)
 cap.release();assert count==steps==250
 stem=f'dp3_scissors_{slug}_failed'
 for suffix,sequence in [('',frames),('_clean',clean)]:
  sequence[0].save(out/(stem+suffix+'.gif'),save_all=True,append_images=sequence[1:],duration=round(2000/fps),loop=0,optimize=False)
 frames[-1].save(out/(stem+'_final.png'))
 records.append(dict(task_type=task,episode=episode,success=False,steps=steps,completion_rate=completion,source=str(source/filename),metadata=meta,gif=stem+'.gif',source_frames=count,exported_frames=len(frames),source_fps=fps))
 print(stem,flush=True)
(out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
(out/'README.md').write_text('''# DP3 unsuccessful pilot trials

Four trials failed to complete their full task by the 250-step limit.
Touch screw has completion 0%; task types 5 and 7 have completion 33%;
task type 9 has completion 25%. These fractions come from rollout logs;
partial completion is not full success or a diagnosis of the failure cause.

Each trial has labeled and clean GIFs and a final-frame PNG. GIFs retain every
other source frame at 10 fps, preserving the original 12.5-second duration.
All use checkpoint 1500, scissors test1, and privileged ground-truth part input.
Together with the success GIF, they represent the five-trial pilot (1/5 success).
Provenance is in manifest.json; exporter is export_dp3_failure_gifs.py.
''')
