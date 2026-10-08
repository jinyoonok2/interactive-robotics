"""Export the verified successful DP3 pilot video for presentation."""
from pathlib import Path
import json
import cv2
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2]
source = root/'results/partinstruct/dp3_eval/media/1500/scissors/split-test1'
out = root/'results/partinstruct/visualizations/dp3_success'
out.mkdir(parents=True, exist_ok=True)
logs = json.loads((source/'rollout_logs.json').read_text())
assert logs['test1/2/env_1/success_1'] == 1
assert logs['test1/2/env_1/ep_id_1'] == 300471
meta = next(m for m in json.loads((source/'video_meta.json').read_text()) if m['video_file']=='task_type-2_env-2.mp4')
assert "'part_grasp': 'left'" in meta['tasks']
cap = cv2.VideoCapture(str(source/meta['video_file']))
fps = cap.get(cv2.CAP_PROP_FPS)
frames=[]; clean=[]
try:
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
except OSError:
 font=small=ImageFont.load_default()
while True:
 ok,bgr=cap.read()
 if not ok:break
 rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
 rgb.thumbnail((640,640),Image.Resampling.LANCZOS)
 clean.append(rgb)
 canvas=Image.new('RGB',(max(640,rgb.width),rgb.height+100),'#111923')
 canvas.paste(rgb,((canvas.width-rgb.width)//2,70))
 d=ImageDraw.Draw(canvas)
 d.text((12,8),'DP3-S: successful scissors grasp',font=font,fill='#43dc94')
 d.text((12,35),'Target: left part | test1 | checkpoint epoch 1500',font=small,fill='white')
 d.text((12,rgb.height+77),'Ground-truth part input | episode 300471 | 83 steps',font=small,fill='white')
 frames.append(canvas)
cap.release()
assert len(frames)==83
for name,sequence in [('dp3_scissors_left_grasp_success.gif',frames),('dp3_scissors_left_grasp_success_clean.gif',clean)]:
 sequence[0].save(out/name,save_all=True,append_images=sequence[1:],duration=round(1000/fps),loop=0,optimize=False)
frames[-1].save(out/'dp3_scissors_left_grasp_success_final.png')
(out/'manifest.json').write_text(json.dumps(dict(source=str(source/meta['video_file']),task_metadata=meta,episode=300471,success=True,frames=len(frames),fps=fps,checkpoint_epoch=1500,privileged_part_input=True,pilot_success_count='1/5'),indent=2)+'\n')
print('Created labeled and clean GIFs:',len(frames),'frames at',fps,'fps')
