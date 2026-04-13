# R&D Report: Imitation Learning for Dual-Arm Manipulation via Diffusion Policy
**(Including System Architecture & Open-Source Community Optimizations)**

## 1. Project Overview

This project implements a robust imitation learning framework for dual-arm robotic manipulation, built on the Diffusion Policy architecture. Through multiple iterations, the system has been engineered to support and benchmark two parallel input modalities:

* **State-only Pipeline:** Relies entirely on low-dimensional proprioceptive state vectors (32-50 dimensions). It acts as a fast-converging baseline with minimal computational overhead.
* **Vision+State Pipeline:** Fuses multi-camera visual feeds with low-dimensional states. While computationally heavier, it is designed to achieve higher task success rates and better spatial generalization.

---

## 2. Theoretical Primer: Why Diffusion Policy?

Before detailing our architectural optimizations, it is important to establish why we migrated from standard Behavior Cloning (BC) to a **Diffusion Policy** paradigm. 



Traditionally, imitation learning models (like simple MLPs or CNNs trained with MSE loss) map observations directly to a single optimal action. This creates the **"multimodal action problem"**: if an expert demonstrates going *left* around an obstacle in one trajectory and *right* in another, a standard BC model averages these out, causing the robot to crash straight into the obstacle.

**Diffusion Policy solves this by reframing action generation as a conditional denoising process:**
1.  **Learning a Distribution:** Instead of predicting a deterministic action, it learns a probability distribution of valid action sequences.
2.  **Forward Process (Training):** We take a sequence of expert actions and iteratively add Gaussian noise until it becomes pure random noise. 
3.  **Reverse Process (Inference):** Starting from pure noise, a neural network (typically a Conditional U-Net) is trained to iteratively denoise the signal, *conditioned* on the current observations (images and robot states). Over multiple steps (e.g., 100 diffusion steps), it "sculpts" the noise back into a coherent, expert-like trajectory.

This approach inherently supports multimodal demonstrations, provides exceptional training stability, and allows the model to predict long sequences of actions (Action Chunking) rather than just the next immediate step.

---

## 3. Core Optimizations & Architectural Upgrades

Early in development, our baseline model suffered from severe bottlenecks: loss convergence was agonizingly slow, and the model frequently collapsed. By analyzing an excellent **third-party open-source `diffusion_policy` repository** and debugging through multiple iterations, we overhauled the training pipeline and network mechanics.

### 3.1 Overhauling the Training Pipeline
A line-by-line comparison with the referenced open-source implementation revealed critical gaps in our initial setup. We implemented the following upgrades to stabilize the training dynamics:

| Configuration | Baseline BC | Upgraded Open-Source Implementation Strategy |
| :--- | :--- | :--- |
| **Optimizer** | Adam (weight_decay=1e-5) | **AdamW (weight_decay=1e-4 / 1e-6)** to significantly improve generalization. |
| **LR Scheduler** | Constant | **CosineAnnealingLR** for smoother convergence in later epochs. |
| **Gradient Clipping**| None | **max_norm=1.0** added to prevent training crashes caused by sudden gradient spikes. |
| **Epoch Scale** | 100 Epochs | Scaled up to **200–2000 Epochs**, aligning with the heavy step requirements of diffusion models. |

### 3.2 The "Double Warmup" & Dynamic EMA Mechanism
Diffusion models are highly susceptible to "noise locking" early in training. To bypass this, we adopted a dual-warmup strategy found in the referenced codebase:

1.  **Learning Rate Warmup:** Following the open-source logic, the learning rate scales linearly for the first **500 steps** before handing control over to the cosine annealer.
2.  **Dynamic EMA (Exponential Moving Average) Warmup:** A static high decay rate (e.g., 0.999) stalls early weight updates. We implemented a dynamic, power-based decay calculator to smoothly ramp up the EMA rate, capped by a maximum threshold:

    $$\text{decay} = \min\left( \text{max\_decay}, 1 - \left(1 + \frac{\text{step}}{\text{inv\_gamma}}\right)^{-\text{power}} \right)$$

    *Engineering Tweak:* We bumped the `power` parameter from the reference's 0.75 up to **0.8** (with `inv_gamma` typically set to 1.0). This structural adjustment allowed the decay rate to accelerate more effectively through the critical initial 100 steps, measurably speeding up the early convergence phase without sacrificing late-stage stability.

### 3.3 Network Compatibility Fix: GroupNorm vs. BatchNorm
Integrating the EMA mechanism exposed a hidden bug: the running mean and variance of standard `BatchNorm` layers weren't being correctly tracked by the EMA. 
* **The Fix:** Adopting best practices from the community, we wrote a recursive script to forcefully replace every `BatchNorm` layer across the entire architecture—including the ResNet18 vision backbone—with `GroupNorm`. This not only fixed the EMA tracking but also stabilized training under the smaller batch sizes required for multi-view image processing.

---

## 4. Algorithm Deep Dive & Multimodal Fusion

With the above fixes in place, the underlying diffusion process operates smoothly.

* **Mathematical Foundation:** The forward noising process is defined as $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon$. The Conditional U-Net, $\epsilon_\theta(x_t, t, \text{obs})$, is trained to predict and subtract this noise, optimized via the objective $L = \text{MSE}(\epsilon_\theta, \epsilon)$.
* **Multimodal Topology:**
    * *Vision Pathway:* Processes `frontview` and `agentview` feeds using a GroupNorm-modified ResNet18 (pretrained weights discarded). A Global Average Pooling (GAP) layer extracts a 256-dim feature vector.
    * *State Pathway:* A standard MLP extracts a 128-dim embedding from the proprioceptive data.
    * *Fusion & Injection:* The vision, state, and a 128-dim sinusoidal timestep embedding are concatenated into a 512-dim global condition variable. This is injected into the `ConditionalUNet1D` residual blocks using **FiLM (Feature-wise Linear Modulation)**.

---

## 5. Engineering Challenges & Troubleshooting

Moving from algorithm to deployment exposed several environment-specific hurdles:

* **Multiprocessing Collisions (Segmentation Faults):** The C++ backend of RoboSuite/MuJoCo clashed violently with PyTorch's Multiprocessing DataLoaders. 
    * *Workaround:* We hardcoded `num_workers=0` for single-machine training and evaluation. It sacrifices data-loading throughput but guarantees 100% memory stability.
* **PyTorch Serialization Policies:** When building the resume-training function, PyTorch 2.6 triggered an `UnpicklingError` due to new security defaults. 
    * *Fix:* We bypassed this by setting `weights_only=False` and restructured the checkpointing logic. Now, a valid checkpoint must bundle the model weights, optimizer states, and the EMA state dictionary to ensure mathematically seamless resumption.

---

## 6. Inference Strategy & Future Roadmap

### 6.1 Action Chunking Implementation
To prevent jerky robotic movements and account for execution latency, we implemented Receding Horizon control:
* **Observation Horizon:** 2 frames
* **Prediction Horizon:** 16 frames
* **Execution (Action) Horizon:** 8 frames

### 6.2 Next Steps & Open Problems
1.  **Concurrency Optimization:** Investigate PyTorch's `spawn` multiprocessing context to safely bypass the `num_workers=0` bottleneck in RoboSuite.
2.  **Inference Acceleration:** Test compressing the DDPM sampling steps from 100 down to 50. We need to evaluate the precision tradeoff versus the 2x gain in robotic control frequency.
3.  **Ablation Studies:** Run rigorous benchmarks to quantify the exact success-rate delta between the Vision+State and State-only modalities on this specific dual-arm task.