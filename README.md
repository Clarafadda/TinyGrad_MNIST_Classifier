# TinyGrad MNIST Classifier

## Overview

This project implements a **digit classifier** using **TinyGrad**, a minimal deep learning framework, and deploys it as a **WebGPU-based web application**.  
It aims to explore lightweight deep learning inference directly in the browser, combining **Python model training** with **WebGPU acceleration** for frontend prediction visualization.

![App Screenshot](screen.png)  

### 🔗 [**Live Demo here !**](https://clarafadda.github.io/TinyGrad_MNIST_Classifier/)

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

---

## Local Setup and Testing Instructions

To verify your local environment and test the models before deployment, follow these steps:

### Prerequisites
 - Python 3.10+ (3.12 recommended)
 - Modern web browser with WebGPU support: Chrome/Edge 113+
 - Git (for cloning the repository)

1. **Clone the Repository**
    ```bash
   git clone https://github.com/Clarafadda/TinyGrad_MNIST_Classifier.git
   cd TinyGrad_MNIST_Classifier
2. **Create and activate your virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
3. **Import all the libraries you will need**
   ```bash
   pip install -r requirements.txt
4. **Install WebGPU Support**
5. **Train a quick model to check the setup**
    ```bash
    STEPS=100 JIT=1 python mnist_mlp.py
6. **Train and import models with the best parameters**
- MLP :
  ```bash
  LR=0.001 BS=128 Steps=1000 JIT=1 python mnist_mlp.py
- CNN :
    ```bash
  LR=0.001, BS=128, Steps=1000 JIT=1 python mnist_convnet.py
7. **Run the Web Application** : Start server from project root
    ```bash
    python -m http.server 8000
8. **Navigate to http://localhost:8000/webapp/** and:

- Select a model (MLP or CNN)
- Draw a digit on the canvas
- See real-time prediction with confidence scores

--- 
## Project Retrospective

During development, I encountered several technical and hardware-related challenges that shaped my understanding of TinyGrad and WebGPU integration.

- **Backend compatibility issues:**  
  At first, the models would not run properly due to backend errors caused by my operating system version.  
  I initially suspected that the issue came from my **Python** or **TinyGrad** versions, which led to a lot of debugging before discovering that the actual problem was the **GPU backend configuration**.


- **Heavy computation and performance limits:**  
  My local machine struggled with the computational load during training. Compilations took a very long time, and running large models like the **CNN** required disabling JIT (`JIT=0`) since my hardware could not handle just-in-time compilation efficiently.

  
- **WebGPU integration:**  
  Translating model weights and tensor operations into browser-friendly shaders was non-trivial and required multiple testing iterations.


- **Optimization:**  
  Reducing latency in the canvas-to-prediction pipeline required WebAssembly-like optimizations and fine-tuning of model exports.


**Key Insight:**  
Even minimalist frameworks like **TinyGrad** can bridge the gap between Python-based machine learning and in-browser GPU inference.  
Despite the technical hurdles, this project demonstrated that lightweight and interpretable frameworks can achieve surprisingly strong results when integrated with modern web technologies such as **WebGPU**.


---

## Credits

- **Author:** Clara Fadda  
- **Frameworks:** [TinyGrad](https://github.com/tinygrad/tinygrad), [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)  
- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/)

---


