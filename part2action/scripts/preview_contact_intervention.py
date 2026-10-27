"""Immediate same-observation command visualization from a short recorded probe."""
import argparse
import base64
import html
import json
import math
from pathlib import Path

import cv2


def plot(tcp, contact, action):
    def xy(p):
        return (35+(p[0]+.3)/1.1*310, 345-(p[1]+.3)/1.1*310)
    tx, ty = xy(tcp); cx, cy = xy(contact); ax, ay = xy(action)
    return f'''<svg viewBox="0 0 380 380"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="orange"/></marker></defs><rect x="35" y="35" width="310" height="310" fill="#121720" stroke="#555"/><text x="40" y="25" fill="white">World XY view (same state)</text><line x1="{tx}" y1="{ty}" x2="{ax}" y2="{ay}" stroke="orange" stroke-width="2" marker-end="url(#arrow)"/><circle cx="{tx}" cy="{ty}" r="6" fill="white"/><circle cx="{cx}" cy="{cy}" r="6" fill="#5dafff"/><circle cx="{ax}" cy="{ay}" r="6" fill="orange"/><text x="35" y="365" fill="#bbb">X: −0.3 to 0.8 m; Y: −0.3 to 0.8 m</text></svg>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = next((args.run_dir/"hierarchical").glob("rollout_results_*.json"))
    trial = json.loads(source.read_text())["results"][0]
    capture = cv2.VideoCapture(trial["rollout_video"])
    sections = []
    for d in trial["diagnostics"]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, d["inference_step"])
        ok, frame = capture.read(); assert ok
        ok, encoded = cv2.imencode(".jpg", frame); assert ok
        image = base64.b64encode(encoded).decode()
        cards = []
        probe = d["contact_intervention"]
        for strength, title in [("1.0", "100% contact"), ("0.25", "25% contact"), ("0.0", "0% contact")]:
            candidate = probe["candidates"][strength]
            action = candidate["action0"]
            distance = math.dist(action[:3], d["contact_xyz"])*100
            cards.append(f'<article><h3>{title}</h3>'+plot(d["tcp_xyz"], d["contact_xyz"], action)+
                f'<p>Command → predicted contact: <b>{distance:.1f} cm</b></p><p>XYZ change from 0%: {candidate["xyz_change_from_zero_cm"]:.1f} cm<br>Rotation change from 0%: {candidate["rotation_change_from_zero_deg"]:.1f}°</p><p>Projected embedding / pooled base RMS: {candidate["projection_to_base_rms_ratio"]:.2f}<br>Contact-dependent signal / base RMS: {candidate["contact_signal_to_base_rms_ratio"]:.2f}</p><p>World XYZ command: '+', '.join(f'{v:.3f}' for v in action[:3])+' m</p></article>')
        sections.append(f'<section id="step{d["inference_step"]}" hidden><h2>Observation {d["inference_step"]}</h2><img width="300" src="data:image/jpeg;base64,{image}"><div class="panels">'+''.join(cards)+'</div></section>')
    capture.release()
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Same-observation contact intervention preview</title><style>body{font:16px system-ui;background:#17191d;color:#eee;max-width:1250px;margin:auto;padding:24px}.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}article{background:#24272d;padding:14px}svg{width:100%}a{color:#9bd0ff}@media(max-width:700px){.panels{grid-template-columns:1fr}}</style><h1>Same observation, different contact input</h1><p>First five observations of the bottle pilot. All three commands use the identical upstream features, predicted contact, robot state and trained checkpoint. These plots show candidate commands, <b>not three executed robot trajectories</b>. The observed images come from the normal-contact pilot.</p><p>White: current TCP. Blue: predicted contact. Orange: commanded position, with an arrow from the TCP. Z and wrist orientation are summarized below each plot; XY arrows show only a top-down projection.</p><label>Observation <input id="slider" type="range" min="0" max="'+str(len(sections)-1)+'" value="0"></label>'+''.join(sections)+'<p>Zeroing contact is an inference intervention, outside normal training inputs. Nearer commanded targets alone do not establish better execution or success. The 39-trial full comparison is running separately.</p><script>function show(){document.querySelectorAll("section").forEach(s=>s.hidden=s.id!=="step"+document.querySelector("#slider").value)}document.querySelector("#slider").oninput=show;show()</script>'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(page)
    print(args.out)


if __name__ == "__main__":
    main()
