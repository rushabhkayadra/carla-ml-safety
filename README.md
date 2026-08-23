CARLA Autonomous Driving Perception System — Machine Learning Safety Case
Course: Introduction to Machine Learning Safety  
University: Otto-von-Guericke University Magdeburg (OVGU)  
Project: CARLA autonomous-driving perception safety analysis  
Author: Rushabh Kayadra  
Primary artifact: `CARLA_MLS_final.ipynb`  
Safety case: `safety_case_report.pdf`
---
1. Project Overview
This project analyses the safety of a prototype autonomous-driving perception subsystem in the CARLA simulator.
The perception stack uses a single forward-facing RGB camera and three independent binary classifiers:
Classifier	Safety question	Output
Pedestrian	Is a pedestrian present?	Binary
Vehicle	Is a vehicle present?	Binary
Traffic light	Is a traffic light present?	Binary
The perception outputs are consumed by a rule-based planner. Because missed detections can directly prevent braking or other safety actions, recall is treated as the primary safety-oriented model metric.
The safety argument is structured using System-Theoretic Process Analysis (STPA):
`Losses → Hazards → Unsafe Control Actions → Causal Loss Scenarios → Safety Constraints → Verification V-1…V-5`
The accompanying safety case report states a restricted-deployment conclusion rather than unrestricted deployment.
> **Important reproducibility note:** The current `CARLA_MLS_final.ipynb` does **not yet reproduce all numerical results reported in `safety_case_report.pdf`.** Several notebook cells use synthetic/mock/hard-coded evidence or different data/checkpoint paths. This README deliberately documents that discrepancy instead of claiming false reproducibility. See [Code–Report Consistency Audit](#9-code--report-consistency-audit).
---
2. System Architecture
The system described in the safety case consists of:
Front RGB camera — 10 Hz, monocular perception input.
Three ResNet-18 binary classifiers — pedestrian, vehicle, and traffic-light presence.
Distance oracle — assumed perfect for the course project.
Rule-based planning module — consumes perception outputs, vehicle speed and steering state.
Drive-by-wire actuators — throttle, brake and steering.
Human safety operator — can override braking or steering.
The architecture intentionally has important limitations:
no LiDAR, radar or ultrasonic redundancy;
the camera is a single point of perception failure;
the traffic-light model detects presence, not red/amber/green state;
the human operator is the fallback mechanism.
The report's STPA control structure therefore treats both the automated planner and the human operator as safety-relevant control loops.
---
3. Model Architecture and Training
The intended perception architecture is:
Backbone: ImageNet-pretrained ResNet-18
Input: RGB image resized to `224 × 224`
Feature representation: 512-dimensional pooled representation
Classification head: `Linear(512, 1)`
Output: unactivated logit
Loss: `BCEWithLogitsLoss`
Optimizer: Adam
Batch size: `32`
Training duration in the safety report: `15` epochs
Random seed: `42`
Image normalization: ImageNet mean/std
The safety report describes pedestrian class imbalance of approximately 23.28% positive and reports a pedestrian positive-class weight of approximately 3.29.
Before publishing the repository, the implementation should explicitly match these training parameters. The current notebook's main training loop uses unweighted `BCEWithLogitsLoss`, so its implementation does not currently match the report's stated pedestrian class weighting.
---
4. Operational Design Domain (ODD)
The safety case restricts nominal operation to a deliberately narrow ODD.
Dimension	Validated / intended condition	Important boundary
Weather	Clear, dry, zero precipitation	Rain, wet roads, puddles, snow/hail are outside ODD
Lighting	Daylight, `10:00–16:00`, solar elevation `>20°`, luminance `>1000 lux`	Fog, smoke, twilight, night, low-angle glare are outside ODD
Camera	Clean, unobstructed, nominal `10 Hz`, auto-exposure locked	Blockage, saturation, frame-rate drop `<8 Hz`, frame loss
Scene	Mapped CARLA Town-01 urban geometry	Unmapped towns and other scene geometries
Speed	`0–50 km/h` in the system ODD definition	Above `50 km/h` is outside ODD
Target agents	Upright pedestrians, passenger vehicles, traffic-signal posts	Unusual poses/devices/signals
The report explicitly notes that the training distribution does not cover important environmental shifts such as rain, wet-road reflections, fog, night conditions and geographic changes.
ODD coverage
The safety report reports:
`k = 1`: 100.00%
`k = 2`: 100.00%
`k = 3`: 100.00%
These values apply to the discretized synthetic parameter grid, not to continuous real-world environmental coverage. They should therefore not be interpreted as proof of real-world ODD coverage.
---
5. STPA Safety Analysis
Losses
The safety case identifies four principal losses:
L-1: loss of human life or fatal injury
L-2: severe personal injury or permanent disability
L-3: physical damage to vehicle/infrastructure
L-4: regulatory violation and loss of operating authorization
Principal hazards
H-1: unsafe longitudinal distance to a pedestrian
H-2: unsafe following distance to a vehicle/obstacle
H-3: entering an active intersection when stopping is required
H-4: nominal-speed operation outside the validated ODD or under corrupted/adversarial perception
H-5: unwarranted violent emergency braking in dense traffic
Key causal scenarios
The report highlights:
pedestrian false negatives from small objects / background dominance;
overconfident false negatives;
silent domain shifts;
adversarial perturbations;
traffic-light presence/state limitations;
human vigilance degradation;
excessive perception-to-actuation latency;
false positives causing unnecessary emergency braking.
---
6. Safety Verification Matrix
The following values are the results stated in the safety case report.
Verification	Metric / requirement	Reported result	Verdict
V-1 — In-distribution detection	Recall ≥ 90% pedestrian; ≥85% vehicle; ≥85% traffic light	Pedestrian 91.02%; Vehicle 95.11%; Traffic light 87.32%	Met
V-2 — FGSM robustness	Recall drop ≤10% at `ε=0.05`	Pedestrian −52%; Vehicle −49%; Traffic light −53%	Not met
V-3 — Calibration	ECE <0.05 after calibration	Pedestrian 0.0342; Vehicle 0.0311; Traffic light 0.0289	Met
V-4 — OOD detection	AUROC ≥0.90	Pedestrian 0.9764; Vehicle 0.9621; Traffic light 0.9488	Met
V-5 — Safe fallback	MRM/fallback within 150 ms; controlled human handoff	Design argument; `a=-2.0 m/s²`, `v≤15 km/h`, takeover ≤1.5 s	Met by design argument
V-1 — In-distribution detection
Reported recall:
Pedestrian: 91.02%
Vehicle: 95.11%
Traffic light: 87.32%
The pedestrian model only clears its 90% requirement by about 1 percentage point, so this should not be described as a large safety margin.
V-2 — Adversarial robustness
At FGSM `ε=0.05`, the reported recall drops are:
Pedestrian: −52.00%
Vehicle: −49.00%
Traffic light: −53.00%
The safety constraint allows at most a 10% relative recall drop, so this verification is clearly not met.
This is a central residual risk and should remain visible in the repository rather than being hidden behind the nominal metrics.
V-3 — Calibration
The report gives post-temperature-scaling ECE values below the `0.05` target:
Pedestrian: 0.0342
Vehicle: 0.0311
Traffic light: 0.0289
The cost-sensitive threshold formulation uses:
`C_FN = 100`, `C_FP = 1`
giving:
`τ* = C_FP / (C_FP + C_FN) = 1 / 101 ≈ 0.0099`
This expresses the safety argument that a false negative is substantially more costly than a false positive.
V-4 — OOD detection
The report uses a penultimate-layer Mahalanobis detector over the 512-dimensional feature representation.
Reported AUROC:
Pedestrian: 0.9764
Vehicle: 0.9621
Traffic light: 0.9488
The target is AUROC ≥0.90.
The report also states that MSP performs poorly because neural classifiers can remain overconfident on anomalous inputs.
V-5 — Safe fallback
The safety case defines a Minimum Risk Maneuver (MRM):
trigger: low confidence or positive OOD alarm;
deceleration: `a = −2.0 m/s²`;
target speed: `v ≤ 15 km/h`;
system decision/fallback budget: 150 ms;
human takeover bound: ≤1.5 s;
recommended operator shift: ≤2 hours.
This is primarily a system design argument, not an end-to-end experimental demonstration in the current notebook.
---
7. Residual Risks
The safety case identifies three major residual-risk areas.
7.1 Adversarial susceptibility
The baseline models are highly vulnerable to white-box FGSM attacks. The report proposes:
adversarial training / PGD;
input denoising and randomization;
temporal consistency across the 10 Hz video stream.
7.2 OOD monitor limitations
A Mahalanobis detector assumes a particular latent-space distribution and may miss transitional conditions such as twilight or wet pavement.
Proposed mitigations include:
monitoring multiple ResNet layers;
lightweight reconstruction-based anomaly detection.
7.3 Pedestrian false negatives
The report identifies small/distant pedestrians and partial occlusions as residual risks. It proposes:
Feature Pyramid Networks;
temporal visual tracking / ConvLSTM or video transformers.
---
8. Deployment Recommendation
The safety report recommends:
> **Deploy with restrictions**
The proposed restrictions are:
ODD restriction: mapped Town-01, clear daytime conditions, `10:00–16:00`, luminance `>1000 lux`, zero precipitation.
Speed cap: `v_max ≤30 km/h`.
Human supervision: shifts ≤2 hours with visual/acoustic takeover alerts and ≤1.5 s takeover target.
Intersection restriction: no unrestricted autonomous traversal of signalized intersections because the classifier detects traffic-light presence but not state.
Adversarial isolation: camera/CAN telemetry should be physically isolated from external networks.
These restrictions are important because the safety case does not establish robust unrestricted autonomous operation.
---
9. Code–Report Consistency Audit
This is the most important section to read before pushing the repository.
The current `CARLA_MLS_final.ipynb` contains several implementation/evidence inconsistencies that should be resolved if the GitHub repository is expected to reproduce the submitted safety case.
Critical issues
1. The notebook does not reproduce the reported V-1 metrics
The final notebook training run uses a 3,600-frame image directory and produces substantially different training/evaluation behaviour. Its displayed training F1 values do not establish the report's:
91.02% pedestrian recall;
95.11% vehicle recall;
87.32% traffic-light recall.
Action: Use one fixed dataset split and one fixed set of checkpoints, then recompute V-1 directly from those checkpoints.
2. Dataset path selection is unsafe for reproducibility
The notebook creates a symlink to:
`/content/drive/MyDrive/CARLA-MLS/test-town-01/rgb-front`
while other parts of the notebook describe the data as training/test data. This makes the experiment dependent on directory-discovery order and can cause a different split from the safety report.
Action: Replace recursive path discovery with explicit dataset paths and explicit split names.
3. The training loss does not match the report
The notebook's main training loop uses:
`BCEWithLogitsLoss()`
without a `pos_weight`.
The safety report specifies approximately:
`w_pos = 2761 / 838 ≈ 3.29`
for pedestrians.
Action: Implement the stated class weighting explicitly and record the value in the notebook output.
4. The report says validation-loss checkpointing, but the notebook saves final weights
The notebook stores:
`all_tasks_history[f"{task_name}_model_state"] = model.state_dict()`
after all epochs. It does not restore the epoch with minimum validation loss before exporting the checkpoint.
Action: Track `best_val_loss` and save/restore the corresponding model state.
5. Calibration evidence is not currently empirical
The notebook contains a synthetic calibration engine and a later cell that explicitly assigns values such as:
`true_ece_pre = {...}`
and
`true_ece_post = {...}`
It also generates reliability curves from simulated values.
Action: Compute ECE and temperature scaling directly from real model logits and a held-out calibration/test split. Do not hard-code final evidence numbers.
6. OOD evidence is partly synthetic
The OOD notebook cell explicitly creates 512-D multivariate-normal feature vectors using `np.random.multivariate_normal(...)`.
The report, however, presents the Mahalanobis AUROC values as evidence from the perception system.
Action: Extract the actual penultimate ResNet features from real ID/fog/night/Town datasets, fit the detector only on the designated ID training/calibration data, and calculate AUROC on real OOD frames.
7. The FGSM qualitative cell is not a valid quantitative V-2 experiment
The adversarial visualization cell constructs fresh `BinaryClassifierResNet` models with:
`weights=None`
and uses a manually simulated ground-truth label of `1.0`.
That can produce useful visualizations, but it does not reproduce the safety-case quantitative FGSM recall-drop evidence.
Action: Load the actual trained checkpoints, compute FGSM against real labeled samples, and report clean vs. adversarial recall on the same evaluation subset.
8. The ODD coverage cell synthesizes ODD metadata
The k-projection cell randomly assigns weather and town categories using seeded `np.random.choice`.
Therefore its 100/100/100% coverage result is coverage of a synthetically generated metadata matrix, not necessarily coverage of the actual CARLA evaluation dataset.
Action: Calculate k-projection coverage from real scenario metadata. If synthetic coverage is intentionally retained, label it clearly as synthetic test-matrix coverage.
9. V-5 is a design argument, not demonstrated by the notebook
The report describes a 150 ms MRM, `−2.0 m/s²` deceleration, ≤15 km/h target and ≤1.5 s takeover. The current notebook does not implement a real planner/actuator timing experiment establishing those numbers.
Action: Either:
add an explicit timing/design-test section and clearly label it as a design verification, or
weaken the claim so that V-5 is presented as a proposed safety mechanism rather than empirically verified behaviour.
10. The report and system description contain an important operational distinction
The provided system description states 4-hour operator shifts and no auditory alerts, while the restricted deployment recommendation introduces ≤2-hour shifts and visual/acoustic alerts.
This can be logically valid as a proposed mitigation, but the report should make the distinction explicit:
`baseline/provided system → identified human-factor risk → mandatory restriction for deployment`
Otherwise V-5 can appear to claim that controls are already implemented when they are actually deployment restrictions.
---
10. Recommended Repository Structure
A clean GitHub repository should look approximately like:
```text
carla-ml-safety/
├── CARLA_MLS_final.ipynb
├── README.md
├── requirements.txt
├── safety_case_report.pdf
│
├── src/
│   └── dataset.py
│
├── checkpoints/
│   ├── pedestrian_classifier.pt
│   ├── vehicle_classifier.pt
│   └── traffic_light_classifier.pt
│
├── data/
│   ├── labels.csv
│   └── rgb-front/              # dataset files excluded from Git
│
└── exercises/
    ├── sheet_03_fundamentals/
    ├── sheet_04_validation/
    ├── sheet_05_testing/
    ├── sheet_06_explainability/
    ├── sheet_07_anomaly_detection/
    ├── sheet_08_adversarial_ml/
    └── sheet_09_uncertainty/
```
Do not commit the full CARLA image dataset unless the course explicitly permits redistribution.
---
11. Environment and Installation
The notebook is designed around Google Colab + Google Drive.
Typical dependencies used by the notebook include:
```text
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
Pillow
opencv-python
```
The notebook currently installs some packages directly with:
```bash
pip install pandas torchvision matplotlib Pillow
```
For a clean GitHub repository, dependencies should instead be pinned or documented in a `requirements.txt`.
Example:
```bash
git clone https://github.com/rushabhkayadra/carla-ml-safety.git
cd carla-ml-safety
pip install -r requirements.txt
```
Then open:
```text
CARLA_MLS_final.ipynb
```
in Google Colab.
---
12. Reproduction Workflow
A reproducible final workflow should be:
Configure the exact course dataset location.
Load the intended training split.
Create fixed train/calibration/test partitions with seed `42`.
Train the three ResNet-18 binary classifiers.
Save the best validation-loss checkpoint for each classifier.
Evaluate V-1 on a fixed in-distribution test set.
Run FGSM at `ε ∈ {0.01, 0.05, 0.10}` and compute recall drops for V-2.
Fit temperature scaling on a calibration split and compute post-calibration ECE for V-3.
Extract real penultimate-layer features and evaluate Mahalanobis OOD detection for V-4.
Document the MRM and human-oversight controls used for V-5.
Export all plots, tables and numerical results into the repository.
Cross-check every README number against the final notebook outputs before committing.
---
13. Safety Case Summary
The project's central safety lesson is that nominal accuracy alone is insufficient for a safety-critical ML perception system.
The safety case combines:
nominal recall;
adversarial robustness;
confidence calibration;
OOD detection;
explainability;
ODD coverage;
STPA hazard analysis;
fallback design;
residual-risk assessment.
The most important finding is the contrast between acceptable nominal/OOD results and severe adversarial vulnerability. The safety case therefore supports restricted supervised prototype operation, not unrestricted autonomous deployment.
---
14. Pre-Push Checklist
Before pushing the repository to GitHub:
[ ] Remove machine-specific paths where possible.
[ ] Replace automatic recursive dataset discovery with explicit paths.
[ ] Ensure the notebook uses the intended dataset split.
[ ] Implement the report's pedestrian `pos_weight ≈ 3.29` if that is the final training configuration.
[ ] Implement best-validation-loss checkpointing.
[ ] Recompute V-1 from the final checkpoints.
[ ] Recompute V-2 using the final checkpoints and real labels.
[ ] Replace synthetic/hard-coded calibration evidence with real logits.
[ ] Replace synthetic OOD feature evidence with real extracted features.
[ ] Clearly distinguish design-argument evidence from measured evidence for V-5.
[ ] Make README metrics match the final notebook exactly.
[ ] Add `requirements.txt`.
[ ] Add/update `.gitignore` so the course dataset is not accidentally committed.
[ ] Include the safety-case report if course submission rules permit it.
[ ] Run the notebook from a clean runtime before claiming full reproducibility.
---
15. Final Status
Safety-case conclusion: Deploy with restrictions.
Most important unresolved technical issue: the current notebook and the submitted safety-case report are not numerically/reproducibly aligned. The repository should be corrected so that the code generates the evidence used in the report, or the report should be revised to reflect the evidence actually produced by the code.
For an academic safety project, this alignment is more important than making the README look polished: every headline safety number should have a traceable path from dataset → model checkpoint → evaluation code → metric → safety constraint → verdict.
