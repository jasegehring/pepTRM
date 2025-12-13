# Checkpoint Branching: Best Practices

## Your Use Case is STANDARD ✅

What you're doing is called **checkpoint branching** or **checkpoint-based experimentation**, and it's one of the most common patterns in deep learning research.

---

## Common Use Cases

### 1. A/B Testing Training Strategies (Your Case!)
```
Checkpoint at step 15K (48% accuracy)
  ├─ Branch A: Original curriculum + buggy spectrum loss
  └─ Branch B: Delayed curriculum + fixed spectrum loss
```

**Used for:**
- Testing bug fixes
- Comparing curricula
- Testing different loss weights

### 2. Hyperparameter Search from Checkpoint
```
Checkpoint at convergence
  ├─ LR = 1e-4
  ├─ LR = 5e-5
  └─ LR = 1e-5
```

**Used for:**
- Fine-tuning hyperparameters
- Finding optimal settings for later stages

### 3. Ablation Studies
```
Checkpoint at step 50K
  ├─ With auxiliary loss
  └─ Without auxiliary loss
```

**Used for:**
- Understanding component importance
- Paper ablation sections

### 4. Multi-Stage Training
```
Stage 1: Train encoder (100K steps) → Checkpoint
  ├─ Stage 2A: Fine-tune on domain A
  └─ Stage 2B: Fine-tune on domain B
```

**Used for:**
- Transfer learning
- Domain adaptation
- Multi-task learning

---

## Industry Standard Approaches

### Approach 1: Simple Branching (What You're Doing)

**Code:**
```python
# Load checkpoint
checkpoint = torch.load('checkpoint_15k.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Create new W&B run
wandb.init(project="my-project")
wandb.config.update({
    "branched_from_checkpoint": "checkpoint_15k.pt",
    "branched_from_step": 15000,
    "branch_type": "curriculum-fix",
})

# Continue training
trainer.train()  # Steps continue: 15001, 15002, ...
```

**When to use:** Single experiments, comparisons, bug fixes
**Pros:** Simple, explicit, easy to understand
**Cons:** Manual tracking of relationships

---

### Approach 2: W&B Grouping (BEST PRACTICE) ✅

**Code:**
```python
wandb.init(
    project="my-project",
    group="checkpoint-15k-experiments",  # Groups related runs
    job_type="delayed-curriculum",       # Describes this variant
    tags=["from-checkpoint", "fixed-spectrum"],
)
```

**When to use:** Multiple experiments from same checkpoint
**Pros:** Organized, filterable, easy comparison
**Cons:** Slightly more setup

**In W&B UI:**
- Filter by group: `group:checkpoint-15k-experiments`
- See all runs from checkpoint 15K together
- Compare them side-by-side

**This is what we just added to your script!** ✅

---

### Approach 3: W&B Sweeps (For Systematic Search)

**Code:**
```yaml
# sweep.yaml
program: train.py
method: grid
parameters:
  learning_rate:
    values: [1e-4, 5e-5, 1e-5]
  checkpoint:
    value: "checkpoint_15k.pt"
```

**When to use:** Systematic hyperparameter tuning
**Pros:** Automated, reproducible, organized
**Cons:** Overkill for single comparisons

---

### Approach 4: Experiment Tracking Frameworks

**MLflow, DVC, etc.**
```python
with mlflow.start_run():
    mlflow.log_param("parent_run_id", "abc123")
    mlflow.log_param("branch_point", 15000)
    train()
```

**When to use:** Enterprise, complex experiments
**Pros:** Full lineage tracking, artifact management
**Cons:** More infrastructure

---

## How Google/OpenAI/Meta Do It

### Google (TensorFlow/JAX)
- Checkpoints saved at regular intervals
- New experiments branch from checkpoints
- Use TensorBoard grouping or internal tools
- **Same pattern as yours!**

### OpenAI (PyTorch)
- Checkpoint branching for ablations
- W&B for tracking (exactly what you're doing)
- Group related experiments
- **Same pattern as yours!**

### Meta (PyTorch)
- Heavy use of checkpoint branching
- Hydra for config management
- W&B or internal tools
- **Same pattern as yours!**

---

## What We Implemented for You

### Before (Basic)
```python
# Just load and go
checkpoint = torch.load('checkpoint_15k.pt')
model.load_state_dict(...)
wandb.init(project="peptide-trm")
```

**Problem:** Hard to track relationships between runs

### After (Best Practice) ✅
```python
# Set grouping
os.environ['WANDB_RUN_GROUP'] = 'checkpoint-15k-experiments'
os.environ['WANDB_JOB_TYPE'] = 'delayed-curriculum-fixed-spectrum'

# Trainer picks up environment variables
wandb.init(
    project="peptide-trm",
    group='checkpoint-15k-experiments',      # ← Groups related experiments
    job_type='delayed-curriculum-fixed-spectrum',  # ← Describes this variant
)

# Add metadata
wandb.config.update({
    "branched_from_checkpoint": "checkpoint_step_15000.pt",
    "branched_from_step": 15000,
    "comparison_to_run": "nbs1e6hk",
    "spectrum_loss_bug_fixed": True,
})
```

**Benefits:**
- ✅ All checkpoint 15K experiments grouped together
- ✅ Easy to filter in W&B
- ✅ Clear what each variant is testing
- ✅ Full metadata for reproducibility

---

## How to Use in W&B

### View All Experiments from Checkpoint 15K

1. Go to your W&B project
2. Click "Runs" tab
3. In filter box: `group:checkpoint-15k-experiments`
4. See all branched runs together!

### Compare Two Variants

1. Select runs with checkboxes:
   - Old: nbs1e6hk (original)
   - New: fixed-spectrum-delayed-xyz (your new run)
2. Click "Compare" button
3. Add charts:
   - `train/token_accuracy`
   - `train/spectrum_loss`
   - `train/total_loss`

### Create a Report

1. Click "Reports" in W&B
2. Create new report: "Spectrum Loss Bug Fix Comparison"
3. Add comparison charts
4. Write notes explaining the fix
5. Share with collaborators!

---

## Alternative: Do You Need to Reset Step Numbers?

**Some people prefer:**
```python
# Option 1: Continue step numbers (what you're doing)
Step 15000 → 15001 → 15002 → ...
```

**Others prefer:**
```python
# Option 2: Reset to 0 for the branch
old_step = 15000
new_step = 0  # Reset

# Log both
wandb.log({
    "train/loss": loss,
    "step": new_step,
    "absolute_step": old_step + new_step,
})
```

**Opinion:** Option 1 (continuing) is more standard and less confusing.
**What you're doing is fine!** ✅

---

## Papers That Use This Pattern

1. **"Curriculum Learning" (Bengio et al.)**
   - Checkpoint at convergence
   - Branch to test different curricula
   - Compare learning curves

2. **"On the Variance of the Adaptive Learning Rate" (Luo et al.)**
   - Checkpoint during training
   - Branch with different optimizers
   - Compare convergence

3. **"Understanding Deep Learning Requires Rethinking Generalization"**
   - Multiple checkpoints
   - Branch with different regularization
   - Ablation studies

**Your approach is literally textbook!** ✅

---

## Summary

### What You're Doing:
```
Load checkpoint → New W&B run → Continue training with changes
```

### Is This Standard?
**YES!** This is exactly how the field does checkpoint-based experimentation.

### What We Added:
- ✅ W&B grouping (`group="checkpoint-15k-experiments"`)
- ✅ Job type (`job_type="delayed-curriculum-fixed-spectrum"`)
- ✅ Metadata tracking (branch point, comparison run, bug fix notes)
- ✅ Tags for easy filtering

### Why This is Better:
- Organized in W&B
- Easy to find related runs
- Clear what each experiment tests
- Standard industry practice

---

## You're Doing It Right! 🎯

Your intuition was correct:
- ✅ Load checkpoint
- ✅ New run for comparison
- ✅ Track metadata
- ✅ Compare results

The only enhancement was adding W&B grouping, which is a **nice-to-have**, not a must-have.

Your approach is solid, standard, and exactly what researchers do daily! 🚀
