# TinyGrad MNIST Classifier

## Overview

This project implements a **digit classifier** using **TinyGrad**, a minimal deep learning framework, and deploys it as a **WebGPU-based web application**.  
It aims to explore lightweight deep learning inference directly in the browser, combining **Python model training** with **WebGPU acceleration** for frontend prediction visualization.

![App Screenshot](screenshot application.png)  

### 🔗 **Live Demo**

LIEN 


---

## Features

- **TinyGrad MLP and CNN training** on the MNIST dataset   
- **WebGPU-powered inference** directly in the browser
- **Model export and loading** from Python to JavaScript   
- **Hyperparameter exploration tracking** via `HYPERPARAMETERS.md`


- **Responsive UI/UX** built with TailwindCSS 
- **Dropdown** to easily switch between MLP and CNN models
- **Interactive drawing canvas**  with working **Pen**, **Eraser**, and **Clear** tools for real-time digit prediction 
- **Bar chart** to display the model's confidence

---

## Model Summary

### Best accuracy
| Model | Best Config | Accuracy | Target | 
|-------|-------------|----------|--------|
| MLP | LR=0.001, BS=128, Steps=1000 | 98.78% | ≥95% |
| CNN | LR=0.001, BS=128, Steps=1000 | 99.53% | ≥98% |

### Best accuracy/time
| Model | Best Config                 | Accuracy | Target | 
|-------|-----------------------------|----------|--------|
| MLP | LR=0.002, BS=256, Steps=500 | 98.18%   | ≥95% |
| CNN | LR=0.001, BS=256, Steps=500 | 99.42%   | ≥98% |

**All training details and tuning experiments are documented here:**  
➡️ [HYPERPARAMETERS.md](HYPERPARAMETERS.md)

## Project Retrospective

# A completer

---

## Credits

- **Author:** Clara Fadda  
- **Frameworks:** [TinyGrad](https://github.com/tinygrad/tinygrad), [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)  
- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/)

---


