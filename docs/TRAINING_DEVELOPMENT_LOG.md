# Diffusion Policy Training Development Log

**Project:** Dual-Arm Manipulation with Diffusion Policy  
**Date:** April 10, 2026  
**Status:** Active Development

---

## Session 1: Initial Problem - Slow Loss Convergence

### Issue Reported
The user reported that training loss was decreasing too slowly and requested a comparison with the official Diffusion Policy training strategy.

### Analysis Performed
Reviewed three key files:
- `diffusion_policy.py` - Original Diffusion Policy implementation
- `train_diffusion.py` - Training script
- `train.py` - Basic training pipeline

### Key Findings

| Strategy | Diffusion Policy (Official) | BC (Current) |
|----------|---------------------------|--------------|
| Optimizer | AdamW (weight_decay=1e-4) | Adam (weight_decay=1e-5) |
| LR Scheduler | CosineAnnealingLR | None |
| EMA | ✅ Yes (ema_decay=0.999) | ❌ No |
| Gradient Clipping | ✅ Yes (max_norm=1.0) | ❌ No |
| DataLoader | num_workers=4, pin_memory=True | num_workers=0 |

### Recommendations Given
1. Switch to AdamW optimizer
2. Add CosineAnnealingLR scheduler
3. Implement EMA (Exponential Moving Average)
4. Add gradient clipping
5. Increase training epochs (100 → 200-500)

---

## Session 2: EMA Warmup Deep Dive

### Concept Explained
Two types of warmup are crucial for Diffusion Policy:

1. **EMA Warmup**: Gradually increases EMA decay from 0 to 0.9999
   - Prevents "noise locking" during early training
   - Formula: `decay = 1 - (1 + step/inv_gamma)^(-power)`

2. **Learning Rate Warmup**: Gradually increases LR from 0 to base_lr
   - Prevents gradient explosion during early steps
   - Default: 500 steps for Diffusion Policy

### Official Implementation Reference
```python
# From zhaorj/diffusion_policy
ema:
  update_after_step: 0
  inv_gamma: 1.0
  power: 0.75        # 0.75 for <1M steps, 2/3 for >1M steps
  min_value: 0.0
  max_value: 0.9999

training:
  lr_warmup_steps: 500
  lr_scheduler: cosine
```

### Critical Insight
- EMA warmup prevents early random noise from being "locked" into the model
- Power=0.8 (user's choice) reaches high decay faster than power=0.75
- Both warmups should be measured in **steps**, not epochs

---

## Session 3: Code Implementation - train_with_vision.py

### Features Added

#### 1. Dynamic EMA with Power=0.8
```python
def get_ema_decay(self, step):
    effective_step = max(0, step - self.update_after_step - 1)
    if effective_step <= 0:
        return 0.0
    value = 1 - (1 + effective_step / self.inv_gamma) ** (-self.power)
    return max(self.min_value, min(value, self.max_value))
```

**Decay Curve (power=0.8):**
| Step | Decay | Status |
|------|-------|--------|
| 0 | 0.0 | Pure new model |
| 10 | 0.48 | Fast adaptation |
| 100 | 0.88 | Stabilizing |
| 1000 | 0.99 | Near convergence |

#### 2. Learning Rate Warmup (Steps-based)
```python
def get_lr_with_warmup(self, step):
    if step < self.warmup_steps:
        return self.base_lr * (step / self.warmup_steps)
    else:
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

#### 3. Checkpoint Organization
Warmup-enabled checkpoints are saved to separate subdirectories:
- `checkpoints_v2/ema_warmup_0.8_lr_warmup_500steps/`
- `checkpoints_v2/ema_warmup_0.75/` (EMA only)
- `checkpoints_v2/lr_warmup_500steps/` (LR only)

---

## Session 4: Training Interruptions & Solutions

### Issue: DataLoader Segmentation Fault
**Error:** `RuntimeError: DataLoader worker (pid 3885186) exited unexpectedly`

**Root Cause:** 
- RoboSuite/MuJoCo C++ backend conflicts with Python multiprocessing
- `num_workers > 0` causes memory corruption

**Solution:**
```python
# Must use num_workers=0 for stability
train_loader = DataLoader(
    ..., num_workers=0, pin_memory=False
)
```

**Performance Impact:**
- `num_workers=0`: ~10-30% slower but stable
- `num_workers=2`: Faster but risks crashes
- Optimal: 2 for training, 0 for validation

### Issue: PyTorch 2.6 Compatibility
**Error:** `_pickle.UnpicklingError: Weights only load failed`

**Solution:**
```python
checkpoint = torch.load(path, map_location=device, weights_only=False)
```

---

## Session 5: Resume Training Implementation

### Load Checkpoint Function
```python
def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, ..., weights_only=False)
    
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if self.ema_model:
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    
    # Restore training progress
    self.epoch = checkpoint['epoch'] + 1
    self.global_step = checkpoint['global_step']
    self.ema_optimization_step = checkpoint['ema_optimization_step']
```

### Usage
```bash
python train_with_vision.py \
    --resume checkpoints_v2/.../latest_model.pt \
    --use_ema_warmup \
    --use_lr_warmup
```

**Key Insight:** When resuming, must include `--use_ema_warmup` and `--use_lr_warmup` flags to maintain consistency with saved configuration.

---

## Session 6: Training Performance Issues

### Observation
Training slowed from 2.56 it/s to 1.32 it/s after modifications.

### Investigation
1. **num_workers=8 tried** → Too many workers caused overhead
2. **num_workers=0 set** → Single process, slower but stable
3. **Optimal: num_workers=2** → Best balance for this setup

### Recommendation
For RoboSuite environments:
- Training: `num_workers=2`
- Validation: `num_workers=0` (to avoid multiprocessing issues)

---

## Session 7: Architecture Comparison

### Two Training Pipelines

| Component | State-Only | Vision+State |
|-----------|-----------|--------------|
| **Script** | `train_new.py` | `train_with_vision.py` |
| **Network** | `ConditionalUNet1D` | `ConditionalUNet1DWithVision` |
| **Encoder** | MLP State Encoder | ResNet18 + MLP State |
| **Input** | State vector (50-dim) | Multi-camera images + state |
| **Dataset** | `RobosuiteDiffusionDataset` | `VisionDiffusionDataset` |
| **Batch Size** | 128 | 128 (memory intensive) |
| **GPU Memory** | ~2GB | ~10-16GB |
| **Speed** | 2.56 it/s | 1.3-2.5 it/s |
| **Success Rate** | Medium | Higher |

### Network Details

**VisionEncoder (ResNet18-based):**
- Replaces BatchNorm with GroupNorm (for EMA compatibility)
- Multi-camera fusion: independent encoders + concatenation
- Output: 256-dim feature vector

**StateEncoder (MLP):**
- Input: flattened state sequence (obs_horizon × state_dim)
- Hidden: 256-dim
- Output: 128-dim embedding

**ConditionalUNet1D:**
- Down dims: [256, 512, 1024]
- GroupNorm throughout (no BatchNorm)
- FiLM conditioning for observation injection

---

## Session 8: Normalization Strategy

### GroupNorm vs BatchNorm
**Decision:** Use GroupNorm exclusively

**Reason:**
- BatchNorm running statistics conflict with EMA
- GroupNorm is computation-graph friendly
- Official Diffusion Policy also recommends GroupNorm

### Implementation
```python
# VisionEncoder replaces BN with GN
for module in resnet.modules():
    if isinstance(module, nn.BatchNorm2d):
        gn = nn.GroupNorm(num_groups, module.num_features)
        replace_module(model, name, gn)
```

---

## Key Technical Decisions Summary

1. **EMA Strategy**: Dynamic warmup with power=0.8 for faster convergence
2. **LR Strategy**: Step-based warmup (500 steps) following official implementation
3. **DataLoader**: Single-process for stability (`num_workers=0`)
4. **Normalization**: GroupNorm throughout for EMA compatibility
5. **Checkpointing**: Separate directories for different warmup configurations
6. **Resume Training**: Full state restoration including EMA and optimizer states

---

## Current Configuration (Recommended)

```bash
python train_with_vision.py \
    --data_dir expert_data_with_images \
    --checkpoint_dir checkpoints_v2 \
    --use_ema_warmup \
    --ema_warmup_power 0.8 \
    --use_lr_warmup \
    --lr_warmup_steps 500 \
    --num_epochs 2000 \
    --batch_size 128 \
    --eval_every 100
```

---

## Open Questions for Future Work

1. **Performance Optimization:** Can we use spawn multiprocessing safely with RoboSuite?
2. **Vision Ablation:** How much does vision improve over state-only for this task?
3. **EMA Power Tuning:** Is 0.8 optimal, or should we use 0.75 for longer training?
4. **Inference Speed:** Can we reduce diffusion steps from 100 to 50 without quality loss?

---

**Log End**

*This log documents the development and optimization of Diffusion Policy training for dual-arm manipulation tasks, with specific focus on EMA warmup strategies and multi-modal (vision+state) learning.*
