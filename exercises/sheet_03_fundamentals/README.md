# Exercise Sheet 3: Fundamentals & Baseline Training


## 1. Automatic Differentiation & Optimization Theory

### Exercise 3.1: Computational Graphs
The scalar algebraic function under analytical review is defined as:

$$f(x, y, z) = \frac{(xy)\sqrt{z}}{\exp(x)}$$

#### Graph Decomposition and Primitive Node Mapping
To evaluate the local sensitivities via automatic differentiation, the composite expression is decomposed into elemental, differentiable computational operations:
1. **$a(x, y) = x \cdot y$** $\rightarrow$ Local gradients: $\frac{\partial a}{\partial x} = y$, $\frac{\partial a}{\partial y} = x$
2. **$b(z) = \sqrt{z} = z^{0.5}$** $\rightarrow$ Local gradient: $\frac{\partial b}{\partial z} = 0.5z^{-0.5}$
3. **$c(a, b) = a \cdot b$** $\rightarrow$ Local gradients: $\frac{\partial c}{\partial a} = b$, $\frac{\partial c}{\partial b} = a$
4. **$d(x) = \exp(x)$** $\rightarrow$ Local gradient: $\frac{\partial d}{\partial x} = \exp(x)$
5. **$f(c, d) = \frac{c}{d} = c \cdot d^{-1}$** $\rightarrow$ Local gradients: $\frac{\partial f}{\partial c} = d^{-1}$, $\frac{\partial f}{\partial d} = -c \cdot d^{-2}$

#### Derivative Computation Capacity
Utilizing reverse-mode automatic differentiation, this graph accumulates intermediate localized gradients by applying the multi-variate chain rule from the output node back to the input parameters. Consequently, we can systematically derive the partial derivatives of the objective function with respect to all primary input variables:

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial d}\frac{\partial d}{\partial x}$$

$$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial y}$$

$$\frac{\partial f}{\partial z} = \frac{\partial f}{\partial c}\frac{\partial c}{\partial b}\frac{\partial b}{\partial z}$$

These precise analytical gradients quantify the directional sensitivity of the function output to perturbations in $x$, $y$, and $z$.

### Exercise 3.2: Backpropagation
The core objective of the backpropagation algorithm is to evaluate the exact partial derivatives of a scalar loss function $L$ with respect to every internal weight parameter $\theta$ across a multi-layered neural network topology. 

By executing an initial forward pass to cache intermediate layer activations and a subsequent backward pass to propagate error signals recursively using the chain rule, backpropagation calculates exact analytical gradients. This framework achieves an optimal linear time complexity of $O(N)$ relative to the total number of tunable system parameters $N$, avoiding the prohibitive computational overhead associated with numerical finite-difference gradient approximations.

### Exercise 3.3: Gradient Descent
The gradient descent algorithm functions as a first-order iterative optimization method designed to discover the local minima of a differentiable loss manifold $L(\theta)$. The algorithm updates the underlying network parameters in the direction of steepest descent, which corresponds to the negative gradient vector.

#### Mathematical Parameter Update Formula
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

Where:
* $\theta_t$: Column vector representing the active state of the network parameters at step $t$.
* $\eta$: Positive scalar coefficient defining the operational learning rate.
* $\nabla_\theta L(\theta_t)$: Column vector of structural partial derivatives evaluated at the current state.

#### Inverted Sign Phenomenon
If the operational sign within the parameter update formulation is inverted from negative to positive ($+$), the optimization trajectory switches from minimizing the objective loss space to maximizing it:

$$\theta_{t+1} = \theta_t + \eta \nabla_\theta L(\theta_t)$$

This mathematical formulation describes **Gradient Ascent**. In a deep learning optimization loop, running this path forces the network parameter configurations to diverge from optimal error minimum boundaries. This causes the loss values to increase toward positive infinity, destabilizing the network weights until numerical overflow occurs.

---

## 2. Practical: Training the CARLA Baseline Models

### Exercise 3.4: Dataset Exploration Metrics

The empirical data profile was analyzed by executing the automated validation asset `dataset_explore.py`.

#### 1. Volumetric Data Split Evaluation
* **Training Data Split Volume:** [Insert absolute output count from execution log] samples.
* **Testing Data Split Volume:** [Insert absolute output count from execution log] samples.

#### 2. Categorical Label Distribution Balance
The distribution profiles across the three binary target domains reveal distinct class imbalances:

| Target Object Domain | Positive Instances ($1$) | Negative Instances ($0$) | Calculated Imbalance Ratio ($1:0$) |
| :--- | :--- | :--- | :--- |
| **Pedestrian Detector** | [Insert Count]  | [Insert Count]  | [Insert Ratio]  |
| **Vehicle Detector** | [Insert Count]  | [Insert Count]  | [Insert Ratio]  |
| **Traffic Light Detector**| [Insert Count]  | [Insert Count]  | [Insert Ratio]  |

#### 3. Visual Visualizations and Co-occurrence Patterns
* **Co-occurrence Metrics:** The structural correlation analysis indicates that [Insert correlation insights, e.g., co-occurrence between traffic light presence and vehicles at intersection zones].
* **Visual Baseline Constraints:** The input frames show highly uniform mid-day illumination profiles, zero atmospheric occlusion, high contrast boundaries, and dry urban asphalt road networks. These characteristics reflect the idealized daytime settings of the CARLA simulation dataset.

---

### Exercise 3.5: Binary Classifier Training Setup

#### 1. Architectural Blueprint and Optimization Strategy
* **Model Backbone:** ResNet-18 Deep Convolutional Network pre-trained on the ImageNet weight initialization matrix.
* **Classification Topography:** The default 1000-class dense layer is replaced with an identity mapping layer. This passes features directly into an isolated, multi-layer linear classification head that outputs a single unscaled binary logit.
* **Loss Function Formulation:** Binary Cross-Entropy with Logits Loss (`nn.BCEWithLogitsLoss`), incorporating numerical stability by processing raw logits directly to avoid log-sum-exp underflow:
  $$L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \cdot \log\sigma(x_i) + (1 - y_i) \cdot \log(1 - \sigma(x_i))]$$
* **Optimizer Configuration:** Adam Optimizer parameterized with a fixed learning rate of $\eta = 1\times10^{-4}$.

#### 2. Loss Tracking and Convergence Curves
The optimization loop ran for 8 epochs per target module using `train.py`, generating separate empirical loss curves:

`[Embed generated image asset here: pedestrian_loss_convergence.png]`  
`[Embed generated image asset here: vehicle_loss_convergence.png]`  
`[Embed generated image asset here: traffic_light_loss_convergence.png]`  

* **Convergence Assessment:** [Analyze whether the training and validation lines decreased and plateaued together smoothly, or if the validation error began to diverge, indicating early overfitting.] 

#### 3. Safety Verification: Multi-Model Isolation vs. Multi-Label Framework
Deploying three distinct, physically decoupled networks is a safer engineering choice than using a single multi-label model for several reasons:
* **Prevention of Shared Latent Corruption:** Multi-label networks share a common feature extraction backbone. Optimization updates for one task can introduce gradients that inadvertently degrade features used by another task. Complete structural isolation ensures that errors in traffic light detection do not affect the accuracy of the pedestrian detector.
* **Granular Architectural Modification:** Separate models allow you to tune or replace individual components independently. For instance, you can swap the pedestrian backbone for a larger, safer network without disturbing the vehicle or traffic light models.
* **Independent Calibration and Adversarial Defense:** Downstream mitigation protocols (such as temperature scaling for uncertainty quantification or adversarial training) can be targeted specifically at vulnerable models without imposing computational or statistical performance penalties on the rest of the perception system.

---

## 3. Performance Evaluation

### Exercise 3.6: Quantitative Evaluation Matrix
The trained classifiers were evaluated on the test split using `train.py` , yielding the following performance profile:

| Classifier Subsystem | Accuracy | Precision | Recall | $F_1$-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Pedestrian Model** | [Insert %] | [Insert %] | [Insert %] | [Insert %] |
| **Vehicle Model** | [Insert %] | [Insert %] | [Insert %] | [Insert %] |
| **Traffic Light Model** | [Insert %] | [Insert %] | [Insert %] | [Insert %] |

#### 1. Performance Bottleneck Analysis
* **Weakest Performing Subsystem:** The [Insert lowest scoring model] model achieved the lowest overall classification scores.
* **Causal Hypothesis:** This performance drop is likely due to [Insert reasons, e.g., lower resolution features, smaller object bounding boxes, or severe class imbalance in the training data].

#### 2. Safety Criticality Optimization: Precision vs. Recall
* **Pedestrian Subsystem (Recall Priority):** For pedestrian detection, **Recall** is the critical safety metric. A false negative means a pedestrian is present but undetected, suppressing the emergency braking command and leading to a catastrophic collision. A false positive merely triggers an unnecessary deceleration event.
* **Vehicle Subsystem (Recall Priority):** Similar to the pedestrian model, **Recall** is prioritized to prevent high-speed collisions with leading vehicles within speed-dependent thresholds.
* **Traffic Light Subsystem (Balanced/Recall Priority):** While a false positive causes unnecessary stops at empty intersections, a false negative causes the autonomous vehicle to run a red light into cross-traffic. Thus, **Recall** remains the primary target for safety assurance.

---

## 4. ODD Gap Analysis

### Exercise 3.7: Training Space vs. Real-World Domain Matching

#### 1. Empirical Training Set Coverage
The empirical training space is exclusively composed of data captured under pristine environmental bounds:
* **Atmospheric State:** Clear skies, zero precipitation.
* **Illumination Vector:** Direct daytime solar tracking, stable lux profiles.
* **Infrastructural Scene Types:** Clean urban asphalt tracks within simulated daytime boundaries.

#### 2. Discovered ODD Dimensional Gaps
Comparing these parameters to the system safety goals established in Sheet 2 reveals significant gaps across critical operational dimensions:

| ODD Dimension | Sheet 2 Target Definition | Sheet 3 Empirical Coverage | Identified Operational Gap |
| :--- | :--- | :--- | :--- |
| **Weather Condition** | Dry, Sudden Rain, Fog Onset | Dry Only  | No precipitation profiles present |
| **Lighting State** | Daytime, Low Solar Angles  | Mid-day Clear Sun Only  | Extreme solar glare/shadow transitions missing |
| **Temporal State** | Dynamic Operational Hours  | Static Daytime  | Nighttime or dawn/dusk transitions missing |

#### 3. Safety Implications of Discovered ODD Gaps
Because the three baseline networks lack exposure to these environmental variations, the system cannot guarantee safe or reliable performance outside of clear daytime conditions. 

Transitions into out-of-distribution states—such as a sudden rainstorm or low-angle sun glare—will likely cause unpredictable classification errors, increasing false negative rates. Since the rule-based planner relies entirely on these binary outputs, these environmental gaps introduce severe system-level collision risks that must be detected and mitigated in later sheets.
