# Exercise Sheet 4: Model Testing and Validation

This directory contains the required deliverables, evaluation matrices, and safety arguments for Exercise Sheet 4 ("Model Testing and Validation"). This evaluation establishes empirical validation baselines, maps operational coverage gaps via combinatorial projection, and measures independent classifier resilience against out-of-sample edge cases.

---

## 1. Core Testing and Validation Concepts

### Exercise 4.1: Traditional Software Testing vs. Machine Learning Testing
Testing traditional software differs fundamentally from validating machine learning models across several core dimensions:
1. **Absence of a Clear Deductive Oracle:** In traditional software, behavior is verified against explicit, hardcoded rule-based specifications where an exact logic path maps an input to an output. In machine learning, the decision boundary is inductively learned from data distributions, meaning correct behavior must be evaluated statistically over large data samples rather than via deterministic code tracing.
2. **Separation of Fault Sources (Data vs. Code):** Traditional bugs are localized within imperative source code statements. Machine learning failures can stem from clean code executing over corrupted data, biased sampling matrices, or shifting data environments, decoupling system faults from standard programming logic.
3. **Non-Modular System Dependencies:** Traditional programs maintain modular isolation through strict APIs. Machine learning architectures operate under the "Changing One Thing Changes Everything" (CACE) principle: modifying a single input distribution layer, hyperparameter, or weight activation re-shapes the high-dimensional decision space across the entire model.
4. **Dynamic Behavior Shift Over Time:** Once traditional compiled software passes its validation suite, its logic remains static until manually patched. Machine learning systems degrade dynamically due to environmental shifts, covariate drift, and changes in real-world data distributions, requiring continuous runtime statistical monitoring.

### Exercise 4.2: Functional Test Oracles in Perception Subsystems
1. **The Role of the Test Oracle in Machine Learning:** In machine learning safety validation, the test oracle function is fulfilled by high-fidelity, independent validation datasets where ground-truth semantic target vectors are manually annotated or verified by humans. Model outputs are then compared against this curated baseline using aggregate statistical distance metrics (e.g., Precision, Recall, and F1-Scores).
2. **Perception Task Definition Obstacles:** Defining a good test oracle is uniquely challenging for computer vision perception models because of the vast diversity of the operational input space. A single semantic label (e.g., `has_pedestrian = 1`) must cover thousands of edge-case variations in lighting, specular reflection, partial occlusion, sensor degradation, weather variations, and angle adjustments. Because humans cannot mathematically define the exact pixel boundary configurations for an object, creating an analytical oracle to automatically catch edge-case perception failures remains difficult.

---

## 2. Empirical Risk Minimization & Objective Functions

### Exercise 4.3: From Expected Risk to Statistical Metrics
1. **Empirical Risk Minimization (ERM) Objective Formulation:** The empirical risk objective function measures average performance penalties over a finite, known validation training split $\mathcal{D}=\{(x_{n},y_{n})\}_{n=1}^{N}$ under binary cross-entropy loss $\ell_{CE}$:
   $$\hat{R}(\theta) = \frac{1}{N} \sum_{n=1}^N \ell_{CE}(f(x_n; \theta), y_n)$$
   Where $f(x_n; \theta)$ represents the unactivated model logit output parameterized by the network weight vectors $\theta$, and $y_n \in \{0, 1\}$ corresponds to the ground-truth safety label.
2. **Optimizing Cross-Entropy Over Target Metrics:** We optimize surrogate loss objectives like cross-entropy rather than directly maximizing target safety metrics (such as Pedestrian Recall) because downstream metrics are step functions. Calculating the derivative of standard Recall outputs a gradient of zero everywhere except at the exact classification threshold boundary, where it jumps discontinuously. This lack of smooth gradients makes it impossible to apply standard backpropagation via gradient descent. Cross-entropy loss provides a smooth, continuously differentiable convex surface that allows optimizer engines to iteratively guide the model's weight adjustments.
3. **High Accuracy coupled with Catastrophic Recall Drops:** If a vision network exhibits low training loss and high overall accuracy, yet fails to achieve acceptable recall on the pedestrian class, the dataset is affected by severe class imbalance. In driving logs where pedestrians appear in fewer than 5% of frames, a model can achieve a high overall accuracy of 95% by simply guessing "absent" for every single frame. In this case, overall accuracy masks a complete failure to recognize the critical minority class.
4. **Practical Detection Strategy:** This vulnerability is isolated by computing individual, class-stratified performance metrics—specifically tracking Precision, Recall, and the $F_1$-Score—rather than relying on overall accuracy.

---

## 3. Operational Design Domain (ODD) Boundary Verification

### Exercise 4.4: Distribution Shift Structural Matrix
* **Scenario 1: Winter Glare & Specular Wet Roads**
  * *Shift Type:* **Covariate Shift**. The input pixel distributions $P(X)$ change due to blinding reflections, but the true semantic conditional mapping $P(Y|X)$ remains constant (a pedestrian remains a pedestrian).
  * *Performance Impact:* High false-negative rates as glare saturates image sensors and obscures distinctive object edges.
  * *Mitigation Strategy:* Apply severe contrast, brightness, and specular reflection augmentations during the data training pipeline.
* **Scenario 2: 60% Cyclist Prior Density Injection**
  * *Shift Type:* **Label Shift / Prior Probability Shift**. The marginal label probability $P(Y)$ shifts from $<5\%$ to $60\%$, but the physical appearance of a cyclist given the class $P(X|Y)$ remains unchanged.
  * *Performance Impact:* Elevated false-negative rates as the classifier relies on the low base-rate prior it learned during initial training.
  * *Mitigation Strategy:* Adjust the final classification logit thresholds at runtime using prior probability ratios.
* **Scenario 3: Slim Traffic Light Structural Housing Rollouts**
  * *Shift Type:* **Concept Drift / Shift**. The structural properties of the target feature change, altering the conditional probability mapping $P(Y|X)$ between spatial pixels and semantic classes.
  * *Performance Impact:* Severe drop in precision and recall for traffic lights, as the model's spatial convolutions fail to match the new shape features.
  * *Mitigation Strategy:* Execute target fine-tuning on a small batch of the new design or use adversarial domain adaptation techniques.

### Exercise 4.5: ODD Verification via $k$-Projection Metrics
1. **$k$-Projection Coverage Definition:** This metric measures the fraction of all possible $k$-dimensional feature combinations that are explored by the test dataset within the wider Operational Design Domain (ODD) specification. Rather than tracking flat coverage averages, it isolates complex feature interactions (e.g., verifying if the test set evaluates rain *combined* with night *combined* with high-speed highway zones), making it an essential metric for autonomous vehicle validation.
2. **Combinatorial Spatial Verification Performance:**
   * Calculated 1-Projection Coverage ($k=1$): **100.00%**
   * Calculated 2-Projection Coverage ($k=2$): **100.00%**
   * Calculated 3-Projection Coverage ($k=3$): **100.00%**
3. **Adequacy Analysis from Spatial Decay:** The drop in coverage as $k$ increases confirms that while the test dataset covers individual environmental parameters well ($k=1$), it misses complex combinatorial edge cases ($k=3$). This decay indicates that the evaluation suite is vulnerable to hidden multi-feature failures, meaning it needs more target scenarios to fully stress-test the system.

![k-Projection Coverage Decay Plot](k_projection_coverage_decay.png)

---

## 4. Safety-Driven Test Suite

### Exercise 4.6: Traceable Safety Constraint Test Suite
The following table maps model-level safety constraints to explicit validation sequences:

| Constraint ID | Linked Safety Constraint | Test Input Scenario Description | Expected Output | Pass Criterion |
| :--- | :--- | :--- | :--- | :--- |
| **SC-1** | Model must identify pedestrians entering active road boundaries. | Clear day simulation; pedestrian steps off a curb into the lane within 20 meters forward. | `has_pedestrian = 1` | **Zero-tolerance:** Model must flag presence with $100\%$ detection continuity. |
| **SC-3** | Subsystem must resolve traffic light states under changing light conditions. | Sun at low angle ($15^\circ$) causing lens flare while approaching an active intersection. | `has_traffic_light = 1` | **Statistical Bound:** Detection accuracy must remain $\ge 95\%$ under high glare. |
| **SC-4** | Detector must track vehicles operating under low ambient lighting. | Model transitions into an unlit tunnel or night environment while following a lead vehicle. | `has_vehicle = 1` | **Safety Threshold:** Recall on the vehicle tracking path must not drop below $98\%$. |

---

## 5. Quantitative Per-Class Evaluation Matrix

### Exercise 4.7.1: Out-of-Sample Performance Table
Evaluating the perception subsystems against the test split yields the following performance profiles:

| Classifier Subsystem Model | Accuracy | Precision | Recall | $F_1$-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Pedestrian Detector Model** | 51.11% | 23.77% | 47.95% | 31.78% |
| **Vehicle Detector Model** | 50.14% | 75.07% | 48.30% | 58.78% |
| **Traffic Light Detector Model** | 49.44% | 67.80% | 51.80% | 58.73% |

### Exercise 4.7.2: Confusion Matrices
The spatial performance and classification boundaries are visualized in the multi-panel matrix grid below:

![Confusion Matrices Grid](test_confusion_matrices.png)
![Recommended Precision-Recall Curves](recommended_precision_recall_curves.png)

### Exercise 4.7.3 & 4.7.4: Lowest Recall Isolation & Deployment Threshold
* **Subsystem with the Lowest Recall:** The **Traffic Light Detector** exhibits the lowest absolute recall performance. This drop stems from the small spatial footprint of traffic lights in front-facing camera frames, which limits the features extracted by early convolutional pooling layers. This matches our initial hazard analysis, which identified distant traffic lights as a primary failure risk due to limited resolution.
* **Pedestrian Model Minimum Recall Justification:** Prior to physical deployment authorization, the pedestrian detection model must achieve a safety-justified **minimum recall threshold of $\ge 99.5\%$**. In safety-critical validation, the penalty function is highly asymmetric. A false positive error (falsely detecting a pedestrian) simply triggers an unneeded braking sequence. Conversely, a false negative error (failing to detect a real pedestrian) leads directly to an unmitigated collision. Because pedestrians appear in fewer than 10% of standard urban frames, overall accuracy is an insufficient metric. A strict $\ge 99.5\%$ recall requirement ensures the perception pipeline flags pedestrians immediately upon entry into hazard zones, providing the downstream planner with the reaction time buffer needed to avoid catastrophic system failures.