# MastersThesis

## Setup

For stochastic volatility models, we use Template Model Builder (TMB, https://github.com/kaskr/adcomp). You need:   
- a working installation of R, see https://www.r-project.org for download.  
- the TMB package, see https://github.com/kaskr/adcomp/wiki/Download.  

Once this is installed, you can clone the repository
```
git clone https://github.com/NinoCourtecuisse/MastersThesis.git
```
and install the Python requirements (ideally in a virtual environment to avoid dependency conflicts)
```
pip install -r requirements.txt
```

We use PyMB (https://github.com/kforeman/PyMB) to wrap TMB in Python. Due to versions incompatibility, it had to be revised and is therefore directly included in my repo at  [`./PyMB/PyMB`](./PyMB/PyMB).  
To check that the R/TMB/PyMB installation was successful, run

```
from PyMB.model import check_R_TMB
assert check_R_TMB()
```

## Run scripts
All the scripts can be found in [`./experiments/scripts`](./experiments/scripts) and can be run by executing
```
./run.sh experiments/scripts/[name-of-the-file].py [arguments of needed]
```
For example, to reproduce the results of MLE on Black-Scholes the following command is used
```
./run.sh experiments/scripts/mle.py --model bs --seed 42 --verbose
```

## Models
All statistical models are implemented as PyTorch neural networks. The root class is defined in [`./models/model_class.py`](./models/model_class.py) and inherits from [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html). The `forward` method, usually representing a forward pass over the neural network, here computes the negative log-likelihood of the model, i.e. the loss function to minimize.

All models are then implemented as children of this root class. In particular they all have a `params` attribute of type [`torch.nn.ParameterDict`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html) and representing all trainable parameters.  
See for example the Black-Scholes model implementation at [`./models/bs_class.py`](./models/bs_class.py) for a simple case.

Currently the following models are implemented:
- Black-Scholes, Constant Elasticity of Variance, Stochastic Volatility, Normal Inverse-Gaussian