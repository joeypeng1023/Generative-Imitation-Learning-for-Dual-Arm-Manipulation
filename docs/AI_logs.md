# Robotics and Embodied Intelligence Project: AI Assistance Log

**Project Name**: Generative Imitation Learning for Dual-Arm Manipulation  
**Team Members**: 
* Ruiqi YIN (59861168) - Student 1
* Chenxi YANG (59901029) - Student 2
* Rongjie ZHAO (59363026) - Student 3
* Zhongyu PENG (59553020) - Student 4

This log details the interactive process between team members and AI tools (primarily Large Language Models) during environment configuration, baseline development, algorithm implementation, data analysis, and theoretical synthesis.

---

## [Student 1 - Ruiqi YIN] Environment Setup & Data Pipeline
**AI Model Used**: Doubao-1.5-Pro

### Problem Solving Record - Chronological Order

#### Issue 1: Difficulty in acquiring datasets, attempts with three data collection methods
* **User Prompt**: Need to acquire a dual-arm robot manipulation dataset to train the diffusion policy model.
* **Cause of Issue**: No suitable datasets found online; expert data generated using RL algorithms like PPO was of poor quality.
* **AI Solution**: Recommended abandoning pre-existing datasets and RL generation in favor of manual collection. Guided the setup of the `robosuite` environment, custom keyboard controllers, and off-screen rendering to fix HDF5 file saving issues.
* **Result**: Successfully established a robust manual data collection pipeline.

#### Issue 2: Library installation failures during environment setup
* **User Prompt**: Various dependency conflicts and installation failures occurred when installing libraries.
* **Cause of Issue**: Dependency version conflicts, platform-specific dependencies (Windows x64), and Python version incompatibilities.
* **AI Solution**: Created a detailed `environment.yml` file explicitly specifying all dependency versions, suggested using `--no-cache-dir` to avoid caching issues, and recommended installing dependencies in batches.
* **Result**: Environment successfully configured.

#### Issue 3: Binding error between robosuite and MuJoCo
* **User Prompt**: Running scripts containing `robosuite` threw the error `Could not find module 'mujoco.dll'`.
* **AI Solution**: Modified `robosuite/utils/binding_utils.py` to hardcode the absolute path for `mujoco.dll`.
* **Result**: Successfully imported `robosuite` and ran the physics engine.

#### Issue 4: Inability to save data in HDF5 format during collection
* **User Prompt**: The official `gather_demonstrations` script failed to convert the `episodes` directory into an `.hdf5` file.
* **AI Solution**: Wrote a custom `gather_demonstrations_as_hdf5` function to read `.npz` and `.xml` files individually and construct a compliant hierarchical structure using `h5py`.

#### Issue 5: Conflict between keyboard controller and MuJoCo's default controls
* **User Prompt**: Attempting to control the robotic arms triggered MuJoCo's default behaviors (e.g., camera view switching).
* **AI Solution**: Created a `CustomKeyboardOp` class inheriting from `Keyboard` to override or intercept specific key events.

#### Issue 6: Screen freezing or crashing during data collection
* **User Prompt**: When using `env.render()`, the screen froze after collecting a few trajectories.
* **AI Solution**: Set `has_renderer` to `False`, switched to off-screen rendering (`has_offscreen_renderer=True`), and displayed the camera feed via OpenCV (`cv2.imshow`).

#### Issue 7: Converted HDF5 file missing crucial datasets
* **User Prompt**: Training the Diffusion Policy threw an error indicating `data/demo_0/obs/robot0_proprio-state` does not exist.
* **AI Solution**: Modified the data saving logic to extract and save all necessary keys (especially `proprio-state`) from the `obs` dictionary returned by `env.step(action)`.

#### Issue 8: Fixed initial position of expert data, lacking generalizability
* **User Prompt**: The object was in the exact same position every time the environment started.
* **AI Solution**: Called `base_env.reset()` before collecting each new demonstration, and utilized a Wrapper to automatically save `_current_task_instance_state` and `xml`.

#### Issue 9: Inconsistent initial positions when visualizing the same demo
* **User Prompt**: The object's initial position during playback was different from when the data was collected.
* **AI Solution**: When visualizing, the saved XML model must be loaded first, followed by setting the saved initial state, and finally executing `sim.forward()` to synchronize the physics state.

---

## [Student 2 - Chenxi YANG] Baseline Model Development & Evaluation (BC & DAgger)
**AI Model Used**: DeepSeek-V3

### 1. Environment Configuration Phase
* **Prompt 1**: `conda` is not recognized as an internal or external command, how to install `conda`?
    * **AI Response/Application**: Provided installation steps for Anaconda and environment variable configuration. Successfully configured and activated `conda`.
* **Prompt 2**: After creating the `dual_arm_diffusion` environment, `requirements.txt` is missing during dependency installation, and there is a `mujoco` version conflict.
    * **AI Response/Application**: Recommended manually installing core dependencies and specifying `mujoco==3.3.0`.
* **Prompt 3**: `numpy` compilation failed due to a missing C++ compiler.
    * **AI Response/Application**: Downloaded and installed Visual C++ Build Tools; `numpy` installed successfully.
* **Prompt 4**: Importing `robosuite` threw the error `Could not find module 'mujoco.dll'`.
    * **AI Response/Application**: Resolved the issue by modifying the hardcoded path in `binding_utils.py`.
* **Prompt 5**: The interpreter does not show `dual_arm_diffusion` in PyCharm, how to add it?
    * **AI Response/Application**: Manually specified the `python.exe` path, successfully switched the interpreter.

### 2. Data Processing & Network Construction
* **Prompt 6**: Loading the HDF5 file threw a `TypeError`.
    * **AI Response/Application**: Resolved issues caused by mismatched data types; successfully read the expert trajectories.
* **Prompt 7**: What are the exact dimensions of the input state space for BC, and what does it contain?
    * **AI Response/Application**: Confirmed the observation space is `robot0_proprio-state` and extracted it as a 50-dimensional vector.
* **Prompt 8**: How to construct a BC network for a 50-dimensional input?
    * **AI Response/Application**: Built a 3-layer MLP (50->256->256->14) with ReLU and MSE loss; successfully ran 100 epochs of training.

### 3. Baseline Testing
* **Prompt 9**: Running `eval_bc.py` threw `ModuleNotFoundError: No module named 'h5py'`.
    * **AI Response/Application**: Executed `pip install h5py`.
* **Prompt 10**: The model loaded successfully during BC evaluation, but all episodes failed (success rate 0.00).
    * **AI Response/Application**: AI assisted in analyzing the cause: the single-arm state could not perceive the spatial relationship between the dual arms and the object, leading to random generalization failure. Recorded as a 0% baseline result.

### 4. DAgger Implementation
* **Prompt 11**: How to implement the DAgger algorithm? The task lacks a closed-form expert solution.
    * **AI Response/Application**: AI provided an implementation scheme for a "nearest neighbor expert" based on a KDTree, retrieving the closest expert action when the agent deviated, and provided the DAgger main loop code framework.
* **Prompt 12**: DAgger execution stuck at "Collecting new trajectories..." with no progress logs.
    * **AI Response/Application**: Added a `max_steps_per_episode` limit and suppressed redundant logs. Successfully ran 2 iterations.

### 5. Covariate Shift Analysis
* **Prompt 13**: How to quantify Covariate Shift?
    * **AI Response/Application**: AI recommended using Maximum Mean Discrepancy (MMD) with an RBF kernel and provided the implementation code.
* **Prompt 14**: MMD results showed the distance for Expert vs. DAgger was actually larger than for BC, why?
    * **AI Response/Application**: Analysis revealed that lacking object pose data (partial observability), DAgger's attempts to correct the state led to even broader invalid exploratory drift.

---

## [Student 3 - Rongjie ZHAO] Diffusion Policy Implementation & Optimization
**AI Model Used**: Moonshot-v1 (Kimi)

### Problem Solving Record & Technical Decisions

#### Interaction 1: Troubleshooting Slow Loss Convergence
* **User Prompt**: Reported that the training loss was decreasing too slowly and requested a comparison with the official Diffusion Policy training strategy.
* **AI Assistance**: Analyzed differences between the custom BC-like pipeline and the official repo. Recommended switching to the AdamW optimizer (weight_decay=1e-4), adding a CosineAnnealingLR scheduler, implementing Exponential Moving Average (EMA), adding gradient clipping, and increasing training epochs.

#### Interaction 2: EMA Warmup Deep Dive
* **User Prompt**: Needed clarification on how to implement EMA and LR warmup correctly for diffusion models.
* **AI Assistance**: Explained the difference between EMA Warmup (prevents "noise locking" during early noisy epochs) and LR Warmup (stabilizes early gradients). Provided PyTorch implementation guidance for setting up a dynamic EMA decay starting from 0 to 0.9999 with a power of 0.8.

#### Interaction 3: Action Chunking & Model Architecture
* **User Prompt**: How to implement Action Chunking to prevent jerky robotic movements and account for execution latency?
* **AI Assistance**: Guided the setup of Receding Horizon Control. Recommended specific horizon windows (Observation Horizon: 2, Prediction Horizon: 16, Execution Horizon: 8) and assisted in refactoring the Conditional 1D U-Net to support these sequence lengths smoothly.

#### Interaction 4: Resolving Hardware & Concurrency Bottlenecks
* **User Prompt**: Encountered `UnpicklingError` during checkpoint loading in PyTorch 2.6 and Segmentation Faults (`SIGSEGV`) during DataLoader multiprocessing with RoboSuite.
* **AI Assistance**: Diagnosed the C++ backend clash between MuJoCo and PyTorch Multiprocessing. Recommended hardcoding `num_workers=0` as a safe workaround. Solved the checkpoint error by setting `weights_only=False` and restructuring the checkpointing logic to bundle model weights, optimizer states, and the EMA state dictionary to ensure mathematically seamless resumption.

#### Interaction 5: Vision Modality Integration
* **User Prompt**: How to fuse multi-camera visual feeds with the existing state-only pipeline?
* **AI Assistance**: Recommended using a ResNet18 encoder (without pretrained weights to avoid domain gap issues) paired with GroupNorm instead of BatchNorm to maintain consistency with EMA. Guided the fusion of visual embeddings with low-dimensional state vectors into the conditional input of the U-Net.

#### Key Technical Decisions Summary
1. **EMA Strategy**: Dynamic warmup with power=0.8 for faster convergence.
2. **LR Strategy**: Step-based warmup (500 steps) following official implementations.
3. **DataLoader**: Single-process for stability (`num_workers=0`) to avoid MuJoCo crashes.
4. **Normalization**: Replaced BatchNorm with GroupNorm throughout the network for EMA compatibility.
5. **Resume Training**: Full state restoration including EMA and optimizer states.

---

## [Student 4 - Zhongyu PENG] Project Synthesis, Visualization & Academic Review
**AI Model Used**: Gemini 1.5 Pro

### Key Records & Workflow

#### Interaction 1: Project Milestone Planning & Task Breakdown
* **User Prompt**: Reviewing current progress, list the remaining deliverables required.
* **AI Assistance**: AI provided a structured project management checklist, explicitly pointing out remaining tasks: obtaining the final Diffusion success rate for the Scaling Law chart, compiling the AI Prompt Log, editing the side-by-side demo video, and structuring a standardized GitHub repository.

#### Interaction 2: GitHub Repository Standardization
* **User Prompt**: How to organize the Github repo to meet academic course submission standards?
* **AI Assistance**: AI recommended a standard robotics learning project structure (dividing into `models/`, `data/`, `ai_logs/`), and guided the drafting of a high-quality `README.md` (including execution commands, environment dependencies, and result screenshots).

#### Interaction 3: Data Visualization Strategy & Python Automated Plotting
* **User Prompt**: Received the packaged data from Student 2, how should I visualize it to demonstrate theoretical depth?
* **AI Assistance**: AI suggested extracting three core charts: the Training Loss curve, the MMD bar chart, and **the most critical PCA state distribution scatter plot**. AI provided a one-click plotting Python script based on `seaborn` and `sklearn.decomposition.PCA` to visually demonstrate Covariate Shift.

#### Interaction 4: Academic Validation & Review of Experimental Images
* **User Prompt**: Check if the generated Loss curves and PCA scatter plots are up to standard.
* **AI Assistance**: AI confirmed the validity of the images and provided chart interpretations: pointing out the stark contrast between the "compact manifold" of the expert data and the "divergent drift" of the BC/DAgger trajectories in the PCA plot, perfectly corroborating the $O(T^2)$ compounding error theory.

#### Interaction 5: Cross-Literature & Data Gap Analysis
* **User Prompt**: Review the drafts from Student 1 and Student 2, pointing out data gaps that need to be excavated from the code.
* **AI Assistance**: After cross-referencing, AI found the report lacked "hard data support". Guided me to extract: specific randomization boundaries (e.g., $\pm 0.15m$), the precise physical meaning of the 50-dimensional state vector, specific MMD values, and pointed out the absence of "trajectory smoothness (Jerk/Acceleration)" calculations.

#### Interaction 6: Quantitative Data Analysis & Evaluation
* **User Prompt**: Provide the newly generated JSON data for MMD and smoothness, requesting evaluation.
* **AI Assistance**: AI validated that DAgger indeed outperformed BC in smoothness (Avg Jerk) and distribution matching. Meanwhile, AI generated a design concept for an interactive visualization component to dynamically demonstrate the comprehensive error penalties of the algorithms during the defense presentation.

#### Interaction 7: Academic Blind Spot Review & Theoretical Deepening
* **User Prompt**: Review the finalized PDF report to identify logical loopholes or theoretical deficiencies.
* **AI Assistance**: AI provided in-depth revision suggestions after reviewing. Pointed out the report should further explain "why DAgger failed despite lowering the MMD," and introduced the advanced concepts of **"Causal Confusion"** and **"Precision Bottleneck"**, elevating the "Failure Analysis" section to the academic depth of a top-tier conference.
