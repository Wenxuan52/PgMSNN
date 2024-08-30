# PgMSNN

Physics-guided Multi-Stage Neural Network

by Wenxaun Yuan and Rui Guo

## Highlights

1. **Innovative Approach**: Introduced a novel Dispersion Factor into the original Physics-Informed Neural Network (PINN) and developed a flexible deep Runge-Kutta method for solving the evolution of step initial conditions in the generalized Gardner equation.
2. **Advanced Model Development**: Proposed a Physics-guided Multi-Stage Neural Network (PgMSNN) model using a multi-stage training strategy, enhancing the model's performance in handling both forward evolution and parameter inversion problems.
3. **Comparative Analysis**: Demonstrated the superior performance of the PgMSNN model over other neural networks like mPINN, PINN, and PeRCNN in numerical experiments, especially in scenarios involving dispersive phenomena and high-frequency components.
4. **Outstanding Predictive Accuracy**: PgMSNN exhibited excellent predictive accuracy with minimal error across different conditions, showing reliable performance in both stability and parameter inversion experiments.
5. **Robust Performance**: The PgMSNN model maintained effectiveness even under noisy data conditions and complex parameter scenarios, indicating its robust representation and reasoning capabilities for the generalized Gardner equation.

## Example result heatmap

Running PgMSNN_main.py will draw the following image:

<img src="https://github.com/Wenxuan52/PgMSNN/blob/master/figures/PgMSNN%20Prediction%20heatmap.png" alt="PgMSNN Prediction heatmap" style="zoom: 33%;" />

In addition, after running PgMSNN_main.py, an npy containing the final prediction of PgMSNN is generated to aid further analysis.

## Models

The code details of the PgMSNN model are encrypted due to confidentiality. But in the PgMSNN_main.py file, we give an example of the model API. The PgMSNN model can be called directly through the code (GPU required), and the detailed model code will be given soon. 

## Requirements

- matplotlib==3.7.5
- sys
- numpy
- pyDOE==0.3.8
- scipy==1.10.1
- torch==1.8.0 (recommend)
