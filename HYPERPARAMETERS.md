# Hyperparameter Exploration Report

**Generated:** 2025-11-02 21:37:49

---
## Executive Summary
This document logs the hyperparameter exploration process for both MLP and CNN models trained on the MNIST dataset using the tinygrad framework.

**Key Findings:**
- **Best MLP Configuration:**  Extended training achieving **[98.78]%** test accuracy

- **Best CNN Configuration:** Better convergence achieving **[99.53]%** test accuracy

---

## 🧠 MLP (Multi-Layer Perceptron)

### Methodology
All MLP experiments were conducted using the `python mnist_explorer.py mlp` command with JIT compilation enabled (`JIT=1`). 

### Configurations Tested

| #  | Category | Description                            | LR     | BS | Steps | Accuracy | 
|----|----------|----------------------------------------|--------|----|-------|----------|
| 1  | baseline | Baseline - Adam with standard params   | 0.001  | 128 | 100   | 94.54%   | 
| 2  | learning_rate | Higher LR - faster convergence test    | 0.003  | 128 | 100   | 94.54%   |
| 3  | learning_rate | Lower LR - stability test              | 0.0005 | 128 | 100   | 94.75%   |
| 4  | batch_size | Larger batch - smoother gradients      | 0.001  | 256 | 100   | 94.73%   |
| 5  | batch_size | Smaller batch - more updates per epoch | 0.001  | 64 | 100   | 94.94%   |
| 6  | training_duration | Extended training - convergence test   | 0.001  | 128 | 1000  | 98.78%   |
| 7  | combo | Aggressive - high LR + large batch     | 0.002  | 256 | 500   | 98.18%   |
| 8  | combo | Conservative - balanced approach       | 0.0008 | 128 | 500   | 97.81%   |
| 9  | combo | Best parameters                        | 0.0008 | 128 | 1000  | 98.45%   |
| 10 | combo | Best parameters                        | 0.0005 | 128 | 1000  | 98.55%   |
| 11 | combo | Best parameters balanced               | 0.0005 | 128 | 500   | 98.14%   |
### Analysis

### Base Line
**94.54%** accuracy with LR=0.001 BS=128 STEPS=100 

Good starting point but under-trained model

#### Impact of Learning Rate
- **High LR (0.002-0.003):** No significant improvement; causing instability
- **Medium LR (0.001):** Already optimal for 100 steps.
- **Low LR (0.0005):** Slight improvement but would require more steps

**Conclusion:** Prefer a lower LR

#### Impact of Batch Size
- **Small batches (64):** Best result for 100 steps ; More updates per epoch
- **Medium batches (128):** Most balanced solution 
- **Large batches (256):** Smoother gradients but fewer updates

**Conclusion:** For a short workout, choose a smaller batch; otherwise, choose a more balanced batch (128).

#### Impact of Training Duration
- **100 steps:** Under-trained model
- **500 steps:** Good compromise between training time and performance 
- **1000 steps:** Massive improvement in accuracy but very long training time ; complete convergence

**Conclusion:** Need to increase the number of steps for a well-trained model. STEPS=500 is a good compromise between time and performance.

### 🏆 Best Configurations

- **Accuracy:** 98.78% : Target reached (>= 95%)
- **Description:** Extended training - convergence test
- **Parameters:**
  - `LR`: 0.001
  - `BS`: 128
  - `STEPS`: 1000

Best ratio performance/time
- **Accuracy:** 98.18% : Target reached (>= 95%)
- **Description:** Aggressive - high LR + large batch
- **Parameters:**
  - `LR`: 0.002
  - `BS`: 256
  - `STEPS`: 500

## **SUMUP** ##

1. **Training Duration is Critical**: Increasing from 100 to 1000 steps 
   improved accuracy by +4.24% (94.54% → 98.78%). This was the single 
   most impactful factor.

2. **Batch Size Impact**: Smaller batches (64) outperformed larger ones 
   (256) for short training runs due to more frequent weight updates 
   (94.94% vs 94.73%).

3. **Learning Rate Stability**: The baseline LR=0.001 proved optimal. 
   Higher LR (0.003) showed no improvement, while lower LR (0.0005) 
   provided marginal gains but would require longer training.

4. **Optimal Combination**: LR=0.001, BS=128, STEPS=1000 achieved 98.78%, 
   well above the 95% target.


## 🎯 CNN (Convolutional Neural Network)

### Configurations Tested

| # | Category | Description                       | LR     | BS  | Steps | Accuracy | Time |
|---|----------|-----------------------------------|--------|-----|-------|----------|------|
| 1 | baseline | Baseline CNN - standard params    | 0.001  | 128 | 100 | 98.40%   | 450.89s |
| 2 | learning_rate | Lower LR - CNNs converge slower   | 0.0005 | 128 | 100 | 98.09%   | 417.52s |
| 3 | learning_rate | Higher LR - risk of instability   | 0.002  | 128 | 100 | 98.20%   | 369.62s |
| 4 | batch_size | Smaller batch - better for CNNs   | 0.001  | 64  | 100 | 98.42%   | 359.64s |
| 5 | batch_size | Larger batch - faster training    | 0.001  | 256 | 100 | 98.47%   | 384.01s |
| 6 | training_duration | More steps - better convergence   | 0.001  | 128 | 1000 | 99.53%   | ?s |
| 7 | combo | Mini-batch - fine-grained updates | 0.001  | 32  | 500 | 99.31%   | ?s |
| 8 | combo | Balanced - optimal CNN settings   | 0.0008 | 64  | 500 | 99.30%   | ?s |
| 9 | combo | Best balanced parameters          | 0.001  | 256 | 500 | 99.42%   | ?s |
### Base Line
**98.40%** accuracy with LR=0.001 BS=128 STEPS=100 

Good starting point.

#### Impact of Learning Rate
- **High LR (0.002):** Slight overshoot, no improvement
- **Medium LR (0.001):** Already optimal for 100 steps.
- **Low LR (0.0005):** Incomplete convergence in 100 steps

**Conclusion:** LR=0.001

#### Impact of Batch Size
- **Small batches (64):** Almost no variation
- **Medium batches (128):** Almost no variation
- **Large batches (256):** Better accuracy and faster

**Conclusion:** Low sensitivity related to batch size

#### Impact of Training Duration
- **100 steps:** Already good convergence
- **500 steps:** Good compromise between training time and performance 
- **1000 steps:** Improvement in accuracy but very long training time ; complete convergence
 

### 🏆 Best Configuration

- **Accuracy:** 99.53% : Target reached! (>= 98%)
- **Description:** training_duration | More steps - better convergence
- **Parameters:**
  - `LR`: 0.001
  - `BS`: 128
  - `STEPS`: 1000
  
Best ratio performance/time
- **Accuracy:** 99.42% : Target reached! (>= 98%)
- **Description:** Best balanced parameters
- **Parameters:**
  - `LR`: 0.001
  - `BS`: 256
  - `STEPS`: 500

## Unexplored Hyperparameters

In this exploration, I focused only on three hyperparameters: **learning rate (LR)**, **batch size (BS)**, and **number of steps/epochs** for both models (MLP and CNN).

Other important hyperparameters that were not tested could have had a significant impact on performance:

- **Optimizers** (Adam, SGD, RMSprop): Different optimizers can accelerate convergence or stabilize training. For instance, Adam often outperforms SGD on deeper architectures. 


- **Activation functions** (ReLU, Tanh, LeakyReLU): Certain functions improve gradient flow and can boost final accuracy, especially for the MLP.  


- **Number of layers and neurons / filters**: Increasing model depth or width may enhance learning capacity but could also lead to overfitting.  


- **Dropout / Batch Normalization**: Regularization techniques that help stabilize training and improve generalization.  


- **Data augmentation parameters (angle, scale)**: Rotating or scaling input images could help the model generalize better by making it more robust to variations in handwriting.


- **Kernel size for CNNs**: Different kernel sizes affect how the model captures patterns in the images.

Exploring these parameters could have potentially led to slightly higher performance or faster convergence, while providing more stable test accuracy.


## Comparative Analysis: MLP vs CNN

### Performance Summary

| Model | Best Config | Accuracy | Target | 
|-------|-------------|----------|--------|
| MLP | LR=0.001, BS=128, Steps=1000 | 98.78% | ≥95% |
| CNN | LR=0.001, BS=128, Steps=1000 | 99.53% | ≥98% |

### Key Differences

**MLP:**
- Sensitive to batch size (BS=64 best for short training)
- Requires 1000 steps to reach 98%+
- Struggles with spatial patterns (treats pixels independently)

**CNN:**
- Robust to batch size (BS=256 works well)
- Reaches 98%+ in only 100 steps
- Exploits spatial structure via convolutions

### Why CNN Outperforms MLP

1. **Spatial Inductive Bias**: Convolutions preserve 2D structure
2. **Parameter Efficiency**: Shared weights reduce overfitting
3. **Translation Invariance**: Detects features anywhere in image
4. **Hierarchical Features**: Learns edges → shapes → digits


**Conclusion**: CNNs converge much faster and reach higher accuracy, 
making them the clear choice for image classification tasks.