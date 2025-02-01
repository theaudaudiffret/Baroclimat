# 🌊 ZooPlankton Vision

Deep learning pipeline for automated segmentation of living organisms from debris in ocean scanner data. This project leverages PyTorch to analyze high-throughput plankton imagery collected at the Villefranche-sur-mer marine research station, enabling efficient classification of marine microorganisms.

## 🚀 Quick Start

### Local Setup

```bash
# Request GPU node (if using cluster)
srun -p gpu_inter --pty bash

# Create and activate virtual environment
python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

# Install core dependencies
python3 -m pip install -e .

# Install additional requirements
pip install pandas tabulate
```

### Configuration & Training

Configure your model and data in `config.yaml` and start training :
```bash
python3 -m torchtmpl.main train config.yaml
```

### Distributed Training

For large-scale training on SLURM clusters:
```bash
python3 submit-slurm.py config.yaml
```

### Model Evaluation

1. After training completion, view predictions from the best model (saved in ONNX format) in ~/logs/:
```bash
python3 -m torchtmpl.main test logs/"model" /mounts/Datasets3/2024-2025-ChallengePlankton/test/rg20090121_scan.png.ppm
```

2. Generate Kaggle submission CSV:
```bash
python3 -m torchtmpl.main submit logs/"model_name"
python3 -m torchtmpl.main submit_bords logs/"model_name"

```

### To Do

- Add Augmentations 
- Add Optimize Scheduler to summary of model.txt (cosine annealing)
- Forecast right edge (see if its better)



