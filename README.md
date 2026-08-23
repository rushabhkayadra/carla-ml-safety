# CARLA ML Safety -- Autonomous Driving Perception System

## Project Overview

This project evaluates the **machine-learning safety of a CARLA-based
autonomous driving perception system**.
The system uses a forward-facing RGB camera and three independent binary
perception models to detect:

-   pedestrians
-   vehicles
-   traffic lights

The perception models are based on **ImageNet-pre-trained ResNet-18**
networks. The project goes beyond nominal classification performance and
evaluates safety-relevant properties including:

-   in-distribution detection performance
-   adversarial robustness using FGSM
-   confidence calibration using temperature scaling
-   out-of-distribution (OOD) detection using penultimate-layer
    Mahalanobis distance
-   explainability using Grad-CAM
-   safety-oriented fallback behaviour and the connection between ML
    verification evidence and the system safety case

The project is intended as a **supervised prototype evaluation**, not as
a claim of unrestricted autonomous-driving readiness.

------------------------------------------------------------------------

## 1. Quick Results

The final safety-case report summarizes the main verification results as
follows:

  ---------------------------------------------------------------------------
  Verification   Safety property          Main result         Verdict
  -------------- ----------------- ------------------ -----------------------
  **V-1**        In-distribution   Pedestrian recall:         **Met**
                 detection                **91.02%**; 
                                             Vehicle: 
                                          **95.11%**; 
                                       Traffic light: 
                                           **87.32%** 

  **V-2**        FGSM robustness         Recall drop:       **Not Met**
                 (`ε = 0.05`)              Pedestrian 
                                         **-52.00%**, 
                                              Vehicle 
                                         **-49.00%**, 
                                        Traffic light 
                                          **-53.00%** 

  **V-3**        Calibrated          Post-calibration         **Met**
                 uncertainty          ECE reported as 
                                      **\< 0.05** for 
                                      all three heads 

  **V-4**        OOD detection     Mahalanobis AUROC:         **Met**
                                           Pedestrian 
                                          **0.9764**, 
                                              Vehicle 
                                          **0.9621**, 
                                        Traffic light 
                                           **0.9488** 

  **V-5**        Safe system             Minimum Risk         **Met**
                 fallback          Maneuver, operator 
                                         takeover and 
                                         intersection 
                                     protocol defined 
  ---------------------------------------------------------------------------

The most important safety finding is that the baseline perception models
are **highly vulnerable to adversarial perturbations**. At `ε = 0.05`,
the pedestrian detector loses 52% relative recall. Therefore, the safety
case does **not** support unrestricted deployment.

The final deployment recommendation is **deploy with restrictions**,
including a restricted ODD, a speed cap, supervised operation, and
additional controls around signalized intersections and adversarial
isolation.

------------------------------------------------------------------------

## 2. Repository Structure

An expected repository layout is:

``` text
carla-ml-safety/
├── README.md
├── CARLA_MLS_final.ipynb
│
├── src/
│   ├── dataset.py
│   └── labels.csv
│
├── data/
│   ├── labels.csv
│   └── rgb-front/
│       └── <CARLA RGB frames>
│
├── checkpoints/
│   ├── pedestrian_classifier.pt
│   ├── vehicle_classifier.pt
│   └── traffic_light_classifier.pt
│
├── exercises/
│   ├── sheet_03_fundamentals/
│   ├── sheet_04_validation/
│   ├── sheet_05_testing/
│   ├── sheet_06_explainability/
│   ├── 07_anomaly_detection/
│   ├── sheet_08_adversarial_ml/
│   └── sheet_09_uncertainty/
│
└── safety_case_report.pdf
```

The large CARLA image dataset should generally **not be committed to
GitHub**. The notebook is designed to access the extracted RGB frames
through a Google Drive mount during execution.

Generated experiment outputs such as plots, CSV logs and model
checkpoints are stored under the corresponding `exercises/` and
`checkpoints/` directories.

------------------------------------------------------------------------

## 3. Model Architecture

### Perception pipeline

The perception subsystem consists of **three separate binary
classifiers**:

``` text
                    Forward RGB Camera
                           │
                           ▼
                    224 × 224 RGB image
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       ResNet-18       ResNet-18      ResNet-18
       Pedestrian       Vehicle       Traffic Light
             │             │             │
             ▼             ▼             ▼
       Binary logit    Binary logit    Binary logit
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Rule-based Planner
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Cruise      Deceleration   Emergency
                                      braking
```

Each model:

-   uses **ResNet-18**
-   starts from **ImageNet-pre-trained weights**
-   receives a `3 × 224 × 224` RGB image
-   replaces the original 1000-class ImageNet head with a
    **single-neuron linear output**
-   produces an unactivated logit for binary classification
-   is trained using `BCEWithLogitsLoss`
-   is optimized with **Adam**
-   uses a learning rate of `0.001`
-   uses batch size `32`
-   is trained for `15` epochs

The three models are deliberately separated rather than using one
multi-task output head. This isolates the three safety-relevant
detection objectives.

### Safety-oriented extensions

The project also evaluates additional safety mechanisms around the
classifiers:

1.  **FGSM adversarial testing**
2.  **Temperature scaling for confidence calibration**
3.  **Penultimate-layer Mahalanobis OOD detection**
4.  **Grad-CAM explainability**
5.  **Cost-sensitive decision thresholding**
6.  **Minimum Risk Maneuver (MRM) fallback logic**

------------------------------------------------------------------------

## 4. Dataset

The project uses **CARLA simulator RGB camera data** from the clear-day
`Town-01` environment.

The safety case describes a dataset of **3,599 clear-day CARLA Town-01
images**, partitioned into:

-   **2,879 training frames**
-   **720 validation/test frames**
-   **80/20 split**
-   stratification/synchronized frame selection using the project
    metadata

Each frame has three binary labels:

  Label                 Meaning
  --------------------- ------------------------------------
  `has_pedestrian`      Pedestrian present in the frame
  `has_vehicle`         Vehicle present in the frame
  `has_traffic_light`   Traffic light present in the frame

The pedestrian class is the minority class in the reported dataset, with
838 positive and 2,761 negative frames.

### Image preprocessing

Images are:

1.  resized to `224 × 224`
2.  converted to tensors
3.  normalized using standard ImageNet statistics:

``` text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Dataset limitations

The training distribution is intentionally narrow:

-   clear weather
-   dry roads
-   daytime conditions
-   CARLA Town-01

It does not adequately represent rain, wet-road reflections, fog,
night-time conditions, low-angle sunlight or different town geometries.
These gaps are therefore treated as **ODD/safety concerns rather than
ordinary test-set variation**.

------------------------------------------------------------------------

## 5. Installation

### Option A -- Google Colab

The notebook is primarily organized for Google Colab execution.

1.  Open `CARLA_MLS_final.ipynb` in Google Colab.
2.  Make sure the repository is available.
3.  Mount Google Drive when prompted.
4.  Place the extracted CARLA dataset under:

``` text
MyDrive/
└── CARLA-MLS/
    └── <extracted CARLA dataset>
```

The notebook searches the mounted directory for the RGB frame data.

### Option B -- Local Python environment

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

Linux/macOS:

``` bash
source .venv/bin/activate
```

Install the required Python packages:

``` bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow opencv-python
```

Clone the repository:

``` bash
git clone https://github.com/rushabhkayadra/carla-ml-safety.git
cd carla-ml-safety
```

For GPU execution, install a PyTorch build compatible with the CUDA
version available on the target machine.

------------------------------------------------------------------------

## 6. Usage

### 6.1 Prepare the data

Ensure that:

``` text
src/labels.csv
```

is available and that the corresponding RGB frames are accessible under:

``` text
data/rgb-front/
```

The notebook also contains logic for synchronizing metadata rows with
physically available image frames.

### 6.2 Run the notebook

Open:

``` text
CARLA_MLS_final.ipynb
```

and execute the cells sequentially.

The main workflow is:

``` text
Dataset setup
     ↓
Train/test split
     ↓
ResNet-18 initialization
     ↓
Train pedestrian classifier
     ↓
Train vehicle classifier
     ↓
Train traffic-light classifier
     ↓
Export checkpoints and metrics
     ↓
Validation / safety testing
     ↓
Adversarial analysis
     ↓
Grad-CAM explainability
     ↓
OOD detection
     ↓
Uncertainty / calibration analysis
```

### 6.3 Model checkpoints

The training section exports model states to:

``` text
checkpoints/
├── pedestrian_classifier.pt
├── vehicle_classifier.pt
└── traffic_light_classifier.pt
```

### 6.4 Experiment outputs

The notebook generates experiment artifacts in the corresponding
`exercises/` folders, including:

-   training/validation metric logs
-   performance plots
-   FGSM visualizations
-   Grad-CAM visualizations
-   OOD score distributions
-   ROC curves
-   reliability diagrams
-   calibration outputs

------------------------------------------------------------------------

## 7. Key Findings

### 7.1 Nominal performance is acceptable within the ODD

The three classifiers meet the in-distribution recall thresholds
reported in the safety case:

-   pedestrian: **91.02%**
-   vehicle: **95.11%**
-   traffic light: **87.32%**

This provides evidence that the models can perform the intended binary
detection tasks under their validated clear-day operating conditions.

### 7.2 Adversarial robustness is the major weakness

FGSM testing at `ε = 0.05` produces severe degradation:

-   pedestrian recall drop: **52%**
-   vehicle recall drop: **49%**
-   traffic-light recall drop: **53%**

The corresponding safety constraint is therefore **not met**.

This is particularly safety-critical because a false negative in
pedestrian or vehicle detection can propagate directly into the
rule-based planner.

### 7.3 Calibration improves confidence reliability

Temperature scaling is used to reduce overconfident predictions.

The safety case reports post-calibration ECE values below the required
`0.05` threshold, supporting the use of confidence-aware fallback logic.

The project also derives a cost-sensitive pedestrian threshold:

``` text
τ* = CFP / (CFP + CFN)
   = 1 / (1 + 100)
   ≈ 0.0099
```

This reflects the asymmetric safety cost of a false negative compared
with a false positive.

### 7.4 Mahalanobis OOD detection is effective

The project compares feature-space OOD detection and reports strong
performance from a **penultimate-layer Mahalanobis detector**:

-   pedestrian: AUROC **0.9764**
-   vehicle: AUROC **0.9621**
-   traffic light: AUROC **0.9488**

The safety case therefore considers the OOD detection requirement met.

### 7.5 Explainability reveals distribution-dependent behaviour

Grad-CAM analysis shows concentrated activation around relevant objects
under nominal daytime conditions. Under degraded OOD conditions,
activations become more diffuse and can shift toward
environmental/background structures.

This supports the argument that explanation behaviour should be
considered together with OOD monitoring rather than treated as a
standalone safety guarantee.

### 7.6 Architectural limitations remain

Important limitations include:

-   a single monocular RGB camera with no sensor redundancy
-   binary traffic-light **presence** detection rather than
    red/amber/green state recognition
-   dependence on human supervision as a fallback
-   vulnerability to adversarial manipulation
-   limited training distribution
-   potential false negatives for small or partially occluded
    pedestrians

------------------------------------------------------------------------

## 8. Safety-Case Connection

The ML experiments are mapped directly to the verification structure of
the safety case.

  -----------------------------------------------------------------------
  Safety-case             ML / system evidence    Safety relevance
  verification                                    
  ----------------------- ----------------------- -----------------------
  **V-1: In-distribution  Per-class recall        Controls failures
  detection**                                     caused by ordinary
                                                  in-ODD false negatives

  **V-2: Robustness to    FGSM recall-drop        Tests vulnerability to
  perturbations**         analysis                adversarial camera
                                                  manipulation

  **V-3: Calibrated       Temperature scaling +   Enables
  uncertainty**           ECE                     confidence-aware
                                                  fallback decisions

  **V-4: OOD detection**  Penultimate-layer       Detects conditions
                          Mahalanobis AUROC       outside the
                                                  training/validated ODD

  **V-5: Safe system      MRM + human takeover +  Limits propagation of
  fallback**              intersection protocol   perception failures
                                                  into unsafe vehicle
                                                  behaviour
  -----------------------------------------------------------------------

The overall safety argument is:

``` text
ML failure modes
      │
      ▼
Causal loss scenarios
      │
      ▼
Safety constraints
      │
      ├── V-1  In-distribution performance
      ├── V-2  Adversarial robustness
      ├── V-3  Calibration
      ├── V-4  OOD detection
      └── V-5  System fallback
      │
      ▼
Safety case
      │
      ▼
Restricted deployment recommendation
```

The most important connection is that **good classification accuracy
alone is insufficient for the safety claim**. The safety case requires
evidence about uncertainty, robustness, OOD behaviour and system-level
mitigation.

### Residual risks

The safety case carries the following major residual risks forward:

1.  **Adversarial susceptibility** -- the FGSM safety constraint is not
    met.
2.  **OOD transition gaps** -- gradual conditions such as twilight or
    wet-road reflections may not be detected reliably.
3.  **Small-object and occlusion failures** -- distant pedestrians
    remain a false-negative risk.
4.  **Traffic-light state limitation** -- presence detection does not
    establish whether a signal is red or green.
5.  **Single-point-of-failure perception** -- the camera is the only
    perception sensor.
6.  **Human supervision limitations** -- prolonged monitoring can
    increase takeover latency.

### Deployment position

The safety case recommends **deploying only with restrictions**, rather
than unrestricted autonomous deployment.

The stated restrictions include:

-   operation within the validated CARLA Town-01/clear-day ODD
-   maximum cruising speed of **30 km/h**
-   operator shifts limited to **2 hours**
-   takeover latency bounded to **1.5 s**
-   mandatory deceleration or human takeover at signalized intersections
-   physical isolation of camera/CAN communication from external
    networks

------------------------------------------------------------------------


https://github.com/rushabhkayadra/carla-ml-safety
