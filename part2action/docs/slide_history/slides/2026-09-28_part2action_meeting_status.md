# Part2Action Meeting Status

Date: 2026-09-28

Speaker note: No new Part2Action GIFs yet. Both new runs are still training and will stop on the 48-hour limit before 30 epochs. Official DP3 rollout is queued, not finished.

Order: what we kept → the two runs now training → they will time out → official DP / DP3 → what the queued rollout is.
---

## Slide 1 — What we kept

Part2Action still does not receive the dataset part mask. The part gate predicts the part from the image. That prediction reweights visual tokens. The dataset mask is only the training target.

The contact head is a different output. It predicts one 3D point on the scene cloud. It does not predict the part mask.

The old ±2 cm residual is off. That residual was a nudge added after the action head had already written the command: cloned world XYZ plus at most 2 cm. It did not feed the contact point back into earlier modules. Misses were 10–15 cm, so the nudge sat at the cap. The fair from-scratch pair was 2/39 without it and 2/39 with contact-plus-offset on the whole path. 8/39 stays out of this comparison.

Say: “We still predict the part. We do not get it from the dataset.”

---

## Slide 2 — Two runs, one difference pair

Old experiment configs were removed. Two recipes remain. Both include the measured finger opening. Both train from scratch on bottle, mug, pliers, and scissors. Joint angles stay out. The hand pose already has position and rotation.

Baseline, job `6617401`. One camera frame. The action head writes world XYZ, wrist, and gripper. The contact point is trained and then not used in the command.

Near-contact, job `6617402`. Same stack, plus two changes. A second camera frame goes through the temporal encoder. Inside 3 cm of the contact, XYZ becomes the predicted contact plus an offset of at most 3 cm. Farther away, XYZ stays the cloned world command. Wrist and gripper stay on the action head.

Say: “Baseline clones the path. The new run uses the contact point only at the end.”

---

## Slide 3 — Training status

Checked 2026-09-27, about 22:15 ET. Both jobs are running. The limit is 48 hours. Neither reaches 30 epochs.

Baseline: epoch 12 of 30, about 34% through that epoch, 22.5 hours in. Pace is about 2 hours per epoch, so it stops around epoch 24.

Near-contact: epoch 6 of 30, about 88% through that epoch, 21.5 hours in. Two frames make each epoch about 3.7 hours, so it stops around epoch 13.

A checkpoint is saved at the end of every finished epoch.

Say: “They are learning. The clock stops them early. We compare the last saved epochs, not a full 30.”

---

## Slide 4 — Official DP and DP3

These are the PartInstruct policies, not Part2Action. DP reads the RGB image plus a part mask. DP3 reads the scene cloud plus the part cloud. Both output the 7D action. Neither predicts the mask.

DP3, job `6613129`, finished all 1550 epochs. Last checkpoint is epoch 1500.

DP, job `6615497`, is still training. It resumed from epoch 100 after the earlier out-of-memory stop. It is past epoch 515 of 3050, about 24 hours into a 72-hour job.

Both were trained on scissors only.

Say: “Their model acts. Something else has to name the part.”

---

## Slide 5 — The rollout that is queued

Ground truth here means the simulator fills in the part. The trained DP3 checkpoint still produces the actions. SAM-2 is not used. The OpenAI key is not used.

Job `6619773` is pending. It rolls out DP3 epoch 1500 on scissors `test1`, one trial for each of the five task types. Videos go to `/bigtemp/xna8aw/partinstruct_runs/dp3_s_6613129_eval_gt_6619773`.

The other official mode lets SAM-2 make the mask and, if asked, lets GPT pick the skill. Those SAM-2 weights are not downloaded, so that mode is not running. DP is not in this rollout because it has not finished.

Say: “This tests their action model with a known part. It is not our no-mask setting.”

---

## Slide 6 — Next

Let the two Part2Action jobs run until the time limit. Evaluate the last saved baseline checkpoint against the last saved near-contact checkpoint on the same 39 trials.

Do not treat 8/39 as the comparison. Do not turn the 3 cm offset into a whole-path offset. Do not give Part2Action the dataset part mask or the part cloud.

Official DP3 ground-truth rollout can be scored when `6619773` finishes. DP waits until its own training stops.
