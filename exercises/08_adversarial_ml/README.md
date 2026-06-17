# Exercise Sheet 8: Adversarial Machine Learning & Robustness Verification

This directory contains the analytical derivations, empirical vulnerability profiles, and system-theoretic safety constraints for Exercise Sheet 8 ("Adversarial Machine Learning"). The primary focus of this security audit is to evaluate the susceptibility of our frozen CARLA binary perception subsystems to white-box gradient-based exploits. We implement the Fast Gradient Sign Method (FGSM) to map model degradation under strict perturbation budgets and trace these vulnerabilities up to system-level safety fallbacks.

---

## 1. Theoretical Foundations & Attack Formulations

### Exercise 8.1: Adversarial Examples vs. Out-of-Distribution Data
* **Adversarial Examples:** These are maliciously optimized input vectors engineered by adding a worst-case, imperceptible perturbation noise mask $\delta$ to a clean training sample $x$. The perturbation vector is specifically calculated to maximize the network's loss function, shifting intermediate layer activations across localized classification hyperplanes to trigger an intentional misclassification.
* **Out-of-Distribution (OOD) Examples:** These samples represent natural covariate shifts or environmental alterations (such as heavy fog, night cycles, or novel city layouts) that violate the model's training distribution boundaries.

#### Core Machine Learning Safety Distinctions
Unlike OOD data—which results from natural, non-malicious environmental factors that spread across all feature vectors—adversarial examples are explicitly optimized threats. They leverage high-dimensional geometric vulnerabilities to force severe model failures while remaining completely identical to nominal data under standard statistical distribution checks.

### Exercise 8.2: Gradient-Based Attack Optimization Mechanics
A basic iterative gradient-based attack updates the input image iteratively using the following optimization rule:
$$x_{i+1} = x_{i} + lpha
abla_{x}\mathcal{L}(y, f(x_{i}))$$

#### 1. Term Characterization Breakdown
* $x_i$: The current spatial state of the input image matrix at iteration step $i$.
* $lpha$: The optimization step size (learning rate), controlling the spatial magnitude of the pixel adjustments applied per iteration.
* $
abla_{x}$: The Jacobian gradient operator computed with respect to the input image coordinates $x$, rather than the model weights.
* $\mathcal{L}(y, f(x_i))$: The classification objective function (e.g., Binary Cross-Entropy with Logits Loss), evaluating the error between the true ground truth label $y$ and the model's forward prediction $f(x_i)$.

#### 2. Targeted vs. Untargeted Attack Topologies
An **untargeted attack** is a loss-maximization problem where the adversary seeks to dissolve model correctness by pushing the image away from its true label, maximizing $\mathcal{L}(y, f(x))$. Conversely, a **targeted attack** forces the model to confidently output an explicit, incorrect target label $y_{	ext{target}}$ chosen by the adversary. To implement a targeted architecture, the update rule minimizes the loss for the fake target class, switching the sign to step *down* the gradient:
$$x_{i+1} = x_{i} - lpha
abla_{x}\mathcal{L}(y_{	ext{target}}, f(x_{i}))$$

#### 3. Perturbation Budget Constraints & Projection Modifications
The unconstrained update rule fails to respect a tight perturbation budget $\|x_{0}-x_{t}\|\le\epsilon$ because the gradient vector can step infinitely across the input domain, corrupting the image until it is visually unrecognizable. To restrict the noise within an $\ell_{\infty}$ bounds budget, the formula must integrate a **Projected Gradient Descent (PGD)** operator that clips the accumulated error back into the valid neighborhood:
$$x_{i+1} = 	ext{Proj}_{x_0 + \mathcal{B}(\epsilon)} \left( x_{i} + lpha \cdot 	ext{sign}(
abla_{x}\mathcal{L}(y, f(x_{i}))) ight)$$
Where $	ext{Proj}$ clamps the modified matrix back within the strict $\epsilon$-ball radius surrounding the original clean image $x_0$, while also safeguarding valid $[0, 1]$ pixel intensity ranges.

### Exercise 8.3: Adversarial Training Defense & Empirical Trade-Offs
Adversarial training treats robustness as a minimax optimization problem, injecting dynamically generated adversarial examples directly back into the training loop:
$$\min_{	heta} \mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ \max_{\|\delta\|\le\epsilon} \mathcal{L}(y, f(x + \delta; 	heta)) ight]$$
The inner maximization discovers the most destructive local perturbation for the current model parameters $	heta$, while the outer minimization adjusts the network weights to minimize that adversarial loss.

#### The Accuracy-Robustness Trade-Off
While adversarial training effectively hardens the network's decision hyperplanes against high-frequency gradient exploits, it introduces a significant trade-off: **a reduction in clean classification accuracy**. By forcing decision boundaries to remain smooth and regularized within an $\epsilon$-radius across all training coordinates, the network loses its ability to fit complex, high-frequency features. This smoother boundary degrades its generalization capability on standard, clean inputs.

---

## 2. Practical Robustness Auditing (Exercises 8.4 & 8.5)

We evaluated our three pre-trained binary classification heads (Pedestrian, Vehicle, and Traffic Light detectors) under white-box Fast Gradient Sign Method (FGSM) attacks:
$$x_{adv} = x + \epsilon \cdot 	ext{sign}(
abla_{x}\mathcal{L}(y, f(x)))$$

### Exercise 8.4.3: Perceptual Human-Inspection Log
The empirical transformations are compiled across our evaluation splits within the visualization matrix below:

![FGSM Qualitative Matrix Grid: Pedestrian](fgsm_qualitative_grid_pedestrian.png)

* **At $\epsilon = 0.01$:** The adversarial noise mask is entirely imperceptible to a human auditor. The semantic content of the scene remains perfectly clear, yet the underlying gradient vectors successfully deceive the intermediate activation layers.
* **At $\epsilon = 0.05$:** High-frequency visual distortion becomes subtly visible as light background grain or texturing. However, the core objects, lane lines, and safety-critical road assets remain completely identifiable to a human operator.
* **At $\epsilon = 0.10$:** The mathematical perturbations become highly visible as a distinct high-frequency noise overlay across all color channels. While humans can still isolate the background semantic layout, the visual quality is deeply degraded.

### Exercise 8.5: Quantitative Safety Performance Matrix
The table below tracks model performance across 100 randomly sampled test frames, mapping the exact **Recall Drop** observed as the adversarial perturbation budget increases:

| Evaluation Target Model | Clean Recall Baseline | $\epsilon = 0.01$ Recall / Drop | $\epsilon = 0.05$ Recall / Drop | $\epsilon = 0.10$ Recall / Drop |
| :--- | :---: | :---: | :---: | :---: |
| **Pedestrian Detector** | 96.00% | 81.00% ($-15.00\%$) | 44.00% ($-52.00\%$) | 11.00% ($-85.00\%$) |
| **Vehicle Detector** | 98.00% | 84.00% ($-14.00\%$) | 49.00% ($-49.00\%$) | 14.00% ($-84.00\%$) |
| **Traffic Light Detector** | 94.00% | 79.00% ($-15.00\%$) | 41.00% ($-53.00\%$) | 08.00% ($-86.00\%$) |

#### Vulnerability Analysis
The empirical results reveal that single-step FGSM attacks cause massive safety degradation across all three primary perception models. Even at an imperceptible budget of $\epsilon = 0.01$, the systems experience an average recall drop of **$14.67\%$**. This vulnerability escalates rapidly; at $\epsilon = 0.05$, over half of all safety-critical targets vanish from object tracking loops, causing the system to fail silently under minor adversarial variations.

---

## 3. System-Theoretic Process Analysis (STPA) Extension (Exercise 8.6)

To protect the vehicle loop against adversarial vulnerabilities, we extend our STPA framework to incorporate adversarial risk vectors.

### Exercise 8.6.1: Refined System Hazards Matrix
* **H-4 (Expanded Perception Hazard):** The vehicle operates outside its validated performance envelope due to unmonitored, corrupted, or adversarially manipulated perception inputs.
* **System-Level Direct Effect:** The primary classifiers fail silently under adversarial attack, causing object tracking dropouts that lead directly to collisions with pedestrians, vehicles, or infrastructure.

### Exercise 8.6.2: Unsafe Control Actions (UCAs) Extension
* **UCA-8:** The automated vehicle trajectory planner continues to execute nominal high-speed cruise velocity commands when camera inputs are subjected to adversarial perturbations and the primary pedestrian classifier has been fooled into outputting a false negative.

### Exercise 8.6.3: Derived Safety Constraints
* **Model-Level Robustness Constraint:** The primary classification networks must implement defensive regularizations (e.g., adversarial training) ensuring that under any white-box perturbation budget bounded by $\epsilon \le 0.01$, the model recall drop cannot exceed **$\le 5.0\%$** relative to the clean dataset baseline.
* **System-Level Fallback Constraint:** If downstream tracking logic detects a sudden, high-frequency drop in object detection confidence or a high-entropy state transition while the vehicle is moving, the trajectory planner must abort nominal driving modes within **100 milliseconds** and initiate a safe fallback maneuver.

### Exercise 8.6.4: Structural Residual Risk Analysis
Even if we achieve robust adversarial training that fully satisfies our model-level constraints, significant **residual risk** remains inside our system safety architecture.

Adversarial training is highly budget-specific. Hardening a network against an $\ell_{\infty}$ attack bound of $\epsilon = 0.01$ provides no protection against an attacker utilizing a slightly larger budget ($\epsilon = 0.03$) or transitioning to alternative mathematical structures like $\ell_2$ or unbounded geometric spatial attacks.

Furthermore, adversarial training cannot protect the vehicle against structural blindspots or edge cases that occur naturally inside the nominal clean distribution. If a pedestrian is physically occluded by urban infrastructure, the primary models will fail to detect them regardless of how robustly they are trained against gradient noise. Because of these limitations, robust training cannot serve as a standalone safety guarantee; it must be coupled with independent system-level fallbacks to manage unavoidable residual risks.