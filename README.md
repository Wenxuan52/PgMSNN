# PgMSNN: Physics-guided Multi-Stage Neural Network

**Authors: Wenxuan Yuan and Rui Guo**

## Overview

**PgMSNN** is a novel framework that integrates a physics-informed multi-stage neural network to address complex evolution problems. It incorporates a new Dispersion Factor in the classic Physics-Informed Neural Network (PINN) architecture, significantly improving the model’s ability to handle both forward evolution and parameter inversion problems for nonlinear partial differential equations, specifically the generalized Gardner equation.

## Highlights

1. **Innovative Physics-Guided Framework:**
The PgMSNN introduces a Dispersion Factor into the standard Physics-Informed Neural Network (PINN), coupled with an advanced deep Runge-Kutta method. This combination improves the handling of initial conditions in the generalized Gardner equation, making the framework more flexible and powerful.

2. **Advanced Multi-Stage Training Strategy:**
The model employs a multi-stage training approach to enhance the learning of complex dynamics. This strategy improves the model’s accuracy and efficiency in both forward evolution and parameter inversion tasks, outperforming traditional PINN-based approaches.

3. **Superior Performance in Numerical Experiments:**
In comparative studies, the PgMSNN outperforms alternative neural network models, such as mPINN, PINN, and PeRCNN, particularly in handling dispersive phenomena and high-frequency components.

4. **Outstanding Predictive Accuracy:**
The model demonstrates exceptional predictive accuracy with minimal error across various experimental conditions, ensuring reliable performance in both stability assessments and parameter inversion problems.

5. **Robust Performance under Noisy Conditions:**
Even when exposed to noisy data and complex parameter variations, PgMSNN retains its robustness, showcasing its ability to accurately model the dynamics of the generalized Gardner equation in challenging scenarios.

## Example result heatmap

Running PgMSNN_main.py will draw the following image:

<img src="https://github.com/Wenxuan52/PgMSNN/blob/master/figures/PgMSNN%20Prediction%20heatmap.png" alt="PgMSNN Prediction heatmap" style="zoom: 33%;" />

Additionally, an `.npy file` containing the final predictions of PgMSNN will be created for further analysis.

## Model Usage

The core code for the PgMSNN model is encrypted due to confidentiality reasons. However, an example API for calling the PgMSNN model is provided in the `PgMSNN_main.py` file. The model can be run directly with GPU support. The full model code will be made available soon.

## Requirements

- matplotlib
- numpy
- pyDOE
- scipy
- torch==1.8.0 (recommend)

## Installation

To get started with PgMSNN, you can clone the repository and install the required dependencies:

``` bash
git clone https://github.com/Wenxuan52/PgMSNN.git
cd PgMSNN
pip install -r requirements.txt
```

## Citation
If you use PgMSNN in your research, please cite the arxiv preprint first, and the latest citation information will be updated after the official publication:


