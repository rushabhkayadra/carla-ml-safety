# Exercise Sheet 3: Fundamentals & Baseline Training

**Course:** Introduction to Machine Learning Safety  
**Date:** May 8, 2026  
**Author:** Rushabh Kayadra  
**Institution:** Otto-von-Guericke-Universität Magdeburg  
**Program:** Master's in Data and Knowledge Engineering (DKE)  

---

## 1. Automatic Differentiation & Optimization Theory

### Exercise 3.1: Computational Graphs
[cite_start]The scalar algebraic function under analytical review is defined as[cite: 73, 76]:

$$f(x, y, z) = \frac{(xy)\sqrt{z}}{\exp(x)}$$

#### Graph Decomposition and Primitive Node Mapping
[cite_start]To evaluate the local sensitivities via automatic differentiation, the composite expression is decomposed into elemental, differentiable computational operations[cite: 72, 79]:
1. **$a(x, y) = x \cdot y$** $\rightarrow$ Local gradients: $\frac{\partial a}{\partial x} = y$, $\frac{\partial a}{\partial y} = x$
2. **$b(z) = \sqrt{z} = z^{0.5}$** $\rightarrow$ Local gradient: $\frac{\partial b}{\partial z} = 0.5z^{-0.5}$
3. **$c(a, b) = a \cdot b$** $\rightarrow$ Local gradients: $\frac{\partial c}{\partial a} = b$, $\frac{\partial c}{\partial b} = a$
4. **$d(x) = \exp(x)$** $\rightarrow$ Local gradient: $\frac{\partial d}{\partial x} = \exp(x)$
5. **$f(c, d) = \frac{c}{d} = c \cdot d^{-1}$** $\rightarrow$ Local gradients: $\frac{\partial f}{\partial c} = d^{-1}$, $\frac{\partial f}{\partial d} = -c \cdot d^{-2}$

#### Derivative Computation Capacity
[cite_start]Utilizing reverse-mode automatic differentiation, this graph accumulates intermediate localized gradients by applying the multi-variate chain rule from the output node back to the input parameters[cite: 72, 79]. [cite_start]Consequently, we can systematically derive the partial derivatives of the objective function with respect to all primary input variables[cite: 79]:

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial d}\frac{\partial d}{\partial x}$$

$$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial c}\frac{\partial c}{\partial a}\frac{\partial a}{\partial y}$$

$$\frac{\partial f}{\partial z} = \frac{\partial f}{\partial c}\frac{\partial c}{\partial b}\frac{\partial b}{\partial z}$$

[cite_start]These precise analytical gradients quantify the directional sensitivity of the function output to perturbations in $x$, $y$, and $z$[cite: 79].

### Exercise 3.2: Backpropagation
[cite_start]The core objective of the backpropagation algorithm is to evaluate the exact partial derivatives of a scalar loss function $L$ with respect to every internal weight parameter $\theta$ across a multi-layered neural network topology[cite: 80, 82]. 

[cite_start]By executing an initial forward pass to cache intermediate layer activations and a subsequent backward pass to propagate error signals recursively using the chain rule, backpropagation calculates exact analytical gradients[cite: 72, 80, 82]. This framework achieves an optimal linear time complexity of $O(N)$ relative to the total number of tunable system parameters $N$, avoiding the prohibitive computational overhead associated with numerical finite-difference gradient approximations.

### Exercise 3.3: Gradient Descent
[cite_start]The gradient descent algorithm functions as a first-order iterative optimization method designed to discover the local minima of a differentiable loss manifold $L(\theta)$[cite: 83, 85]. [cite_start]The algorithm updates the underlying network parameters in the direction of steepest descent, which corresponds to the negative gradient vector[cite: 85].

#### Mathematical Parameter Update Formula
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

Where:
* [cite_start]$\theta_t$: Column vector representing the active state of the network parameters at step $t$[cite: 85].
* [cite_start]$\eta$: Positive scalar coefficient defining the operational learning rate[cite: 85].
* [cite_start]$\nabla_\theta L(\theta_t)$: Column vector of structural partial derivatives evaluated at the current state[cite: 85].

#### Inverted Sign Phenomenon
[cite_start]If the operational sign within the parameter update formulation is inverted from negative to positive ($+$), the optimization trajectory switches from minimizing the objective loss space to maximizing it[cite: 85, 86]:

$$\theta_{t+1} = \theta_t + \eta \nabla_\theta L(\theta_t)$$

[cite_start]This mathematical formulation describes **Gradient Ascent**[cite: 86]. [cite_start]In a deep learning optimization loop, running this path forces the network parameter configurations to diverge from optimal error minimum boundaries[cite: 86]. [cite_start]This causes the loss values to increase toward positive infinity, destabilizing the network weights until numerical overflow occurs[cite: 86].

---

## 2. Practical: Training the CARLA Baseline Models

### Exercise 3.4: Dataset Exploration Metrics

[cite_start]The empirical data profile was analyzed by executing the automated validation asset `dataset_explore.py`[cite: 89, 96, 97].

#### 1. Volumetric Data Split Evaluation
* [cite_start]**Training Data Split Volume:** [Insert absolute output count from execution log] samples[cite: 98].
* [cite_start]**Testing Data Split Volume:** [Insert absolute output count from execution log] samples[cite: 98].

#### 2. Categorical Label Distribution Balance
[cite_start]The distribution profiles across the three binary target domains reveal distinct class imbalances[cite: 91, 99]:

| Target Object Domain | Positive Instances ($1$) | Negative Instances ($0$) | Calculated Imbalance Ratio ($1:0$) |
| :--- | :--- | :--- | :--- |
| **Pedestrian Detector** | [cite_start][Insert Count] [cite: 99] | [cite_start][Insert Count] [cite: 99] | [cite_start][Insert Ratio] [cite: 99] |
| **Vehicle Detector** | [cite_start][Insert Count] [cite: 99] | [cite_start][Insert Count] [cite: 99] | [cite_start][Insert Ratio] [cite: 99] |
| **Traffic Light Detector**| [cite_start][Insert Count] [cite: 99] | [cite_start][Insert Count] [cite: 99] | [cite_start][Insert Ratio] [cite: 99] |

#### 3. Visual Visualizations and Co-occurrence Patterns
* [cite_start]**Co-occurrence Metrics:** The structural correlation analysis indicates that [Insert correlation insights, e.g., co-occurrence between traffic light presence and vehicles at intersection zones][cite: 100].
* [cite_start]**Visual Baseline Constraints:** The input frames show highly uniform mid-day illumination profiles, zero atmospheric occlusion, high contrast boundaries, and dry urban asphalt road networks[cite: 7, 17, 90]. [cite_start]These characteristics reflect the idealized daytime settings of the CARLA simulation dataset[cite: 7, 17, 90].

---

### Exercise 3.5: Binary Classifier Training Setup

#### 1. Architectural Blueprint and Optimization Strategy
* [cite_start]**Model Backbone:** ResNet-18 Deep Convolutional Network pre-trained on the ImageNet weight initialization matrix[cite: 106].
* [cite_start]**Classification Topography:** The default 1000-class dense layer is replaced with an identity mapping layer[cite: 106]. [cite_start]This passes features directly into an isolated, multi-layer linear classification head that outputs a single unscaled binary logit[cite: 95, 106].
* [cite_start]**Loss Function Formulation:** Binary Cross-Entropy with Logits Loss (`nn.BCEWithLogitsLoss`), incorporating numerical stability by processing raw logits directly to avoid log-sum-exp underflow[cite: 95, 108]:
  $$L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \cdot \log\sigma(x_i) + (1 - y_i) \cdot \log(1 - \sigma(x_i))]$$
* [cite_start]**Optimizer Configuration:** Adam Optimizer parameterized with a fixed learning rate of $\eta = 1\times10^{-4}$[cite: 108].

#### 2. Loss Tracking and Convergence Curves
[cite_start]The optimization loop ran for 8 epochs per target module using `train.py`, generating separate empirical loss curves[cite: 105, 109]:

`[Embed generated image asset here: pedestrian_loss_convergence.png]`  
`[Embed generated image asset here: vehicle_loss_convergence.png]`  
`[Embed generated image asset here: traffic_light_loss_convergence.png]`  

* [cite_start]**Convergence Assessment:** [Analyze whether the training and validation lines decreased and plateaued together smoothly, or if the validation error began to diverge, indicating early overfitting.] [cite: 109]

#### 3. Safety Verification: Multi-Model Isolation vs. Multi-Label Framework
[cite_start]Deploying three distinct, physically decoupled networks is a safer engineering choice than using a single multi-label model for several reasons[cite: 110]:
* **Prevention of Shared Latent Corruption:** Multi-label networks share a common feature extraction backbone. Optimization updates for one task can introduce gradients that inadvertently degrade features used by another task. Complete structural isolation ensures that errors in traffic light detection do not affect the accuracy of the pedestrian detector.
* **Granular Architectural Modification:** Separate models allow you to tune or replace individual components independently. For instance, you can swap the pedestrian backbone for a larger, safer network without disturbing the vehicle or traffic light models.
* **Independent Calibration and Adversarial Defense:** Downstream mitigation protocols (such as temperature scaling for uncertainty quantification or adversarial training) can be targeted specifically at vulnerable models without imposing computational or statistical performance penalties on the rest of the perception system.

---

## 3. Performance Evaluation

### Exercise 3.6: Quantitative Evaluation Matrix
[cite_start]The trained classifiers were evaluated on the test split using `train.py` [cite: 112][cite_start], yielding the following performance profile[cite: 113]:

| Classifier Subsystem | Accuracy | Precision | Recall | $F_1$-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Pedestrian Model** | [Insert %] | [Insert %] | [Insert %] | [Insert %] |
| **Vehicle Model** | [Insert %] | [Insert %] | [Insert %] | [Insert %] |
| **Traffic Light Model** | [Insert %] | [Insert %] | [Insert %] | [Insert %] |

#### 1. Performance Bottleneck Analysis
* [cite_start]**Weakest Performing Subsystem:** The [Insert lowest scoring model] model achieved the lowest overall classification scores[cite: 114].
* [cite_start]**Causal Hypothesis:** This performance drop is likely due to [Insert reasons, e.g., lower resolution features, smaller object bounding boxes, or severe class imbalance in the training data][cite: 114].

#### 2. Safety Criticality Optimization: Precision vs. Recall
* [cite_start]**Pedestrian Subsystem (Recall Priority):** For pedestrian detection, **Recall** is the critical safety metric[cite: 115]. [cite_start]A false negative means a pedestrian is present but undetected, suppressing the emergency braking command and leading to a catastrophic collision[cite: 22, 115]. [cite_start]A false positive merely triggers an unnecessary deceleration event[cite: 21, 115].
* [cite_start]**Vehicle Subsystem (Recall Priority):** Similar to the pedestrian model, **Recall** is prioritized to prevent high-speed collisions with leading vehicles within speed-dependent thresholds[cite: 22, 115].
* [cite_start]**Traffic Light Subsystem (Balanced/Recall Priority):** While a false positive causes unnecessary stops at empty intersections, a false negative causes the autonomous vehicle to run a red light into cross-traffic[cite: 22, 115]. [cite_start]Thus, **Recall** remains the primary target for safety assurance[cite: 115].

---

## 4. ODD Gap Analysis

### Exercise 3.7: Training Space vs. Real-World Domain Matching

#### 1. Empirical Training Set Coverage
[cite_start]The empirical training space is exclusively composed of data captured under pristine environmental bounds[cite: 7, 17, 90, 121]:
* [cite_start]**Atmospheric State:** Clear skies, zero precipitation[cite: 7, 17, 90, 122].
* [cite_start]**Illumination Vector:** Direct daytime solar tracking, stable lux profiles[cite: 7, 17, 90, 122].
* [cite_start]**Infrastructural Scene Types:** Clean urban asphalt tracks within simulated daytime boundaries[cite: 7, 17, 90, 122].

#### 2. Discovered ODD Dimensional Gaps
[cite_start]Comparing these parameters to the system safety goals established in Sheet 2 reveals significant gaps across critical operational dimensions[cite: 119, 120, 123]:

| ODD Dimension | Sheet 2 Target Definition | Sheet 3 Empirical Coverage | Identified Operational Gap |
| :--- | :--- | :--- | :--- |
| **Weather Condition** | [cite_start]Dry, Sudden Rain, Fog Onset [cite: 34] | [cite_start]Dry Only [cite: 7, 17, 90] | [cite_start]No precipitation profiles present [cite: 123] |
| **Lighting State** | [cite_start]Daytime, Low Solar Angles [cite: 33, 34] | [cite_start]Mid-day Clear Sun Only [cite: 7, 17, 90] | [cite_start]Extreme solar glare/shadow transitions missing [cite: 123] |
| **Temporal State** | [cite_start]Dynamic Operational Hours [cite: 33, 34] | [cite_start]Static Daytime [cite: 7, 17, 90] | [cite_start]Nighttime or dawn/dusk transitions missing [cite: 123] |

#### 3. Safety Implications of Discovered ODD Gaps
[cite_start]Because the three baseline networks lack exposure to these environmental variations, the system cannot guarantee safe or reliable performance outside of clear daytime conditions[cite: 124, 125]. 

[cite_start]Transitions into out-of-distribution states—such as a sudden rainstorm or low-angle sun glare—will likely cause unpredictable classification errors, increasing false negative rates[cite: 34, 124]. [cite_start]Since the rule-based planner relies entirely on these binary outputs, these environmental gaps introduce severe system-level collision risks that must be detected and mitigated in later sheets[cite: 23, 124, 126].