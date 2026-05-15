# Final-Project
# Adaptive Runtime Strategies for Image Processing Pipelines

# Overview

This project implements a simplified adaptive image-processing pipeline that compares different execution strategies for image filtering. The system evaluates:

* Spatial convolution
* FFT-based convolution
* Adaptive runtime execution

The adaptive strategy dynamically selects the faster execution method based on workload characteristics and previous runtime measurements.

The project focuses on:

* runtime-system behavior
* execution strategy selection
* performance tradeoffs
* adaptive decision-making
* workload-dependent optimization


# Dependencies and Installation

Install the required Python libraries:

```bash
pip install numpy scipy matplotlib pillow pandas torch
```

---

# Project Directory Structure

project/
│
├── images/
│   ├── xray1.png
│   ├── xray2.png
│   └── xray3.png
│
├── results/
│   ├── history.csv
│  
│
├── pipeline.py
├── filters.py
├── runtime.py
├── metrics.py
├── visualizations.py
├── experiments.py
├── jit_experiments.py
└── README.md
```

---

# Running the Experiments

Run the main experiment file:

```bash
python experiments.py
```

This executes:

* spatial pipeline experiments
* FFT pipeline experiments
* adaptive runtime experiments
* runtime crossover analysis
* memory comparison
  
* Note: JIT optimization experiment will be run seperately under jit_experiments.py

---

# Reproducing the Experiments

The project includes the following experiments:

## 1. Runtime Comparison

Measures runtime across:

* multiple chest X-ray images
* multiple image sizes
* repeated runs

## 2. Kernel Crossover Analysis

Compares spatial and FFT filtering across different kernel sizes.

## 3. Memory Tradeoff Experiment

Estimates memory usage for:

* spatial filtering
* FFT filtering
* adaptive execution

## 4. JIT Optimization Experiment

Run the jit_experiments.py file

Tests:

```python
torch.compile()
```

using the PyTorch eager backend.

---

# Expected Outputs

Running the experiments generates:

* runtime statistics
* adaptive execution decisions
* runtime history logs
* output comparison images
* performance graphs

Generated graphs include:

* runtime vs image size
* crossover analysis
* runtime vs memory tradeoff
* JIT runtime comparison

Runtime history is stored in:

results/history.csv



# Notes

* The project was tested on CPU workloads.
* Results may vary slightly between runs due to system overhead and caching behavior.
* Adaptive execution uses previous runtime measurements to improve future execution decisions.

University of Wisconsin–Milwaukee
Domain Specific Programming for AI Final Project
