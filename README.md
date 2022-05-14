# HowPredictableIsTraffic-UncertaintyQuantification
The source code of the the manuscript "How predictable is traffic: quantifying predictive uncertainty in network-level speed forecasting"

Model structure is stacked spatio-temporal graph convolution based attention modules. It can be found in ```custom_layers.py```(```STGA``` class). The backbone is adapted from Attention based spatial-temporal graph convolutional networks for traffic flow forecasting (Guo et al. 2019 https://ojs.aaai.org/index.php/AAAI/article/view/3881) and our previous dynamic graph convolution (Li et al. 2021, https://www.sciencedirect.com/science/article/pii/S0968090X21002011). Model details, parameters setting can be found in ```STA.ipynb``` and files in ```custom_model```. Pretrained models are also provided to reproduce the results in the paper. 

For used dataset, please send an emails to G.Li-5@tudelft.nl and we will share the link.

More details will be provided after getting published
