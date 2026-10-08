# Part2Action Physical Grounding and Execution Progress

Date: 2026-09-02

## Slide 1 — Goal and Research Question

Part2Action aims to execute language-conditioned, part-specific manipulation tasks such as “touch the mug handle” or “grasp the bottle lid” by combining visual observations, 3D point clouds, hierarchical skill phases, contact prediction, and robot action generation.

Our central research question is why strong offline phase and action metrics have not translated into reliable physical execution in PartGym, especially when the robot must reach and contact a specific object part.

The baseline completed 8/39 PartGym trials (20.5%); therefore, recent work has focused on identifying whether failures come from phase selection, spatial grounding, contact prediction, arm targets, gripper timing, or low-level physical execution.

## Slide 2 — Architecture Experiments That Did Not Improve Success

Skill-conditioned gating and oracle phase termination did not improve performance, indicating that incorrect phase switching was not the primary bottleneck; the robot often failed physically even when the intended skill phase was known.

Oracle part-mask evaluation did not improve offline action predictions, while gate-only pretraining degraded performance, suggesting that improving the existing visual gate alone was insufficient to repair robot execution.

Delta TCP actions and a separate gripper head were tested as independent ablations, but neither improved overall PartGym success; better action representation or gripper timing cannot compensate when the arm approaches the wrong physical location.

These negative results shifted the research focus away from hierarchy and toward direct measurement of reaching, contact geometry, and physical predicates.

## Slide 3 — Physical Failure Diagnostics

We added privileged PartGym diagnostics for every rollout step, including TCP-to-target-part distance, predicted-contact-to-part distance, arm-action-target-to-part distance, and physical touch, grasp, and placement predicates.

In the matched 39-trial baseline, touch contact prediction was within 3 cm of the requested part in only 6/18 phase visits (33.3%), while grasp contact prediction was within 3 cm in 19/21 visits (90.5%).

Touch physically completed in only 1/18 visits, while grasp completed in 10/21; thirteen failed episodes had contact predictions off the requested part, and eight reached within 5 cm but still failed the required physical predicate.

These results showed two distinct problems: the model often grounds touch on the wrong region, and correct geometric approach still does not guarantee fingertip contact or stable grasping.

## Slide 4 — Failed Exact Contact-Label Correction

The original loader derives one exact contact point from many target-part points; we attempted to improve this by using the first in-phase gripper closure for grasping, minimum TCP-to-part distance for touching, and no contact supervision for movement phases.

The approach appeared reasonable, but the derived touch “contact” moments were still approximately 6.1 cm from the target part on average, meaning closest approach was not a reliable substitute for true physical contact.

After 30 epochs, overall success decreased from 8/39 to 4/39, grasp completion decreased from 10/21 to 3/21, and touch completion remained 1/18; the new checkpoint was therefore rejected.

This experiment demonstrated that a model can achieve low offline error against generated labels while becoming physically worse when those labels do not represent real contact.

## Slide 5 — Point-Level Target-Part Recognition

Instead of guessing one exact touch point, the new head predicts which of the approximately 1,024 scene points belong to the requested part; for “touch the mug handle,” every handle point is positive while mug-body, table, and background points are negative.

A 400-demonstration audit confirmed exact index correspondence between all 31,695 flagged target-part points and the full scene point cloud, so the dataset provides direct point-level supervision without nearest-neighbor label transfer.

The baseline policy, action head, contact head, and hierarchy were frozen while only the new point-part head trained for 10 epochs; it achieved 69.0% training precision, 96.4% recall, and 80.2% F1.

Offline evaluation produced 68.7% precision, 97.2% recall, and 80.5% F1, but this used training demonstrations at a larger stride because no validation split exists; it is promising localization evidence, not proof of improved robot behavior.

## Slide 6 — Oracle Execution and Current Evaluation

A separate privileged controller used PartGym’s true target-part geometry to test physical execution independently of learned perception; touch completion improved from the learned baseline’s 1/18 to 6/18, showing that correct geometry materially helps.

Oracle touch still succeeded in only 33.3% of trials, demonstrating that orientation, fingertip/TCP offset, collision-safe final approach, and contact holding remain important bottlenecks even when the requested part is known.

We are now evaluating the point-part head on 39 held-out PartGym scenes by comparing predicted scene points against simulator target-part geometry while leaving robot actions unchanged.

The original generic-GPU smoke test failed because PyTorch3D lacked kernels for the selected RTX 2080 Ti; compatible H100 smoke job 6556965 and dependent full job 6556966 are queued.

If held-out recognition remains strong, the next experiment will softly add point-part probabilities to contact-attention scores and run the same 39 physical trials; if it generalizes poorly, we will improve precision and semantic point features before connecting it to actions.
