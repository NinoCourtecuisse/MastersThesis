# MastersThesis
## Setup
You will need a working Python installation, we use Python 3.11.5.

Start by cloning the repository
```
git clone https://github.com/NinoCourtecuisse/MastersThesis.git
```
then move to the root
```
cd MastersThesis
```
To avoid dependency conflicts, we recommend to create a virtual environment
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
Finally, install the Python requirements
```
pip install -r requirements.txt
```

## Reproducing experiments

#### Data
We use close prices from the S&P500 Index (ticker: SPX) from 2006-01-03 to 2023-08-31 (both dates included). The dataset has a total of 4446 rows.  
We obtained the data from Optionmetrics via Wharton Research Data
Services (WRDS).

The data needs to be saved as a comma separated CSV at [`./data/spx.csv`](./data/spx.csv) with two columns named 'date' and 'close'. An empty file with only the header is given as reference.

#### Running the scripts
The scripts to reproduce all the plots in the Master's thesis report can be found in [`./experiments`](./experiments).  
The script where our proposed method is implemented is [`./experiments/proposed_method.py`](./experiments/proposed_method.py).

At the top of each script file, there is a comment of the form
```
"""
Description of what the script does.

Usage:
    ./run.sh experiments/name_of_script_file.py [arguments if needed]
"""
```
To run a script, just run the command below usage from the root of the repository.

## About the repository

The source code can be found in [`./src`](./src).  
We describe some important components below.

#### Model pool
The class "ModelPool" defined [`./src/models/ModelPool.py`](./src/models/ModelPool.py) is the main object of interest. It represents several model classes with several parameters (a.k.a particles).  
Two important methods are implemented:
- estimate_nothing: Follow "Estimate Nothing" on this model pool
- update_particles: Move the particles towards regions a-posteriori more likely.

More information can be found with this implementation.

#### Model classes
The model classes are defined in [`./src/models`](./src/models).
Each model class has
-   A prior on its parameter space
-   A transform to map the prior's support to an unconstrained space
-   A (log) transition density

The base logic is implemented in the parent class Model [`./src/models/Model.py`](./src/models/Model.py). All model classes inherits from Model.  

The models currently implemented are Black-Scholes, CEV and NIG. 

#### Priors and Optimization
In [`./src/utils/priors.py`](./src/utils/priors.py) we define different structured priors.  
Each prior has a corresponding transform defined in [`./src/utils/optimization.py`](./src/utils/optimization.py). A transform is a map from the support of the prior to an unconstrained space (in which we perform the optimization).

#### SGLD
Stochastic Gradient Langevin Dynamics (SGLD) is implemented in [`./src/inference/Sgld.py`](./src/inference/Sgld.py).  
It inherits from torch.optim.Optimizer so that it can be used as any other PyTorch optimizer.

#### Distributions
Some distributions were not implemented in PyTorch, our implementation can be found in [`./src/utils/distributions.py`](./src/utils/distributions.py).
It concerns the Inverse Gaussian, Normal Inverse Gaussian and Scaled Beta distributions.

#### Special functions
The (exponentially scaled) Bessel function of order 1 is implemented in 
[`./src/utils/special_functions.py`](./src/utils/special_functions.py).  
An implementation is available in PyTorch via the function
```
torch.special.scaled_modified_bessel_k1
```
however the gradients were somehow unstable ruining the optimization.

## Special case: SV models

For stochastic volatility models, we use Template Model Builder (TMB, https://github.com/kaskr/adcomp).  

In the end we don't use SV models for the main experiments so that this section can be skipped.  
The only scripts that require TMB are 
- [`./experiments/mle_sp500.py`](./experiments/mle_sp500.py)
- [`./experiments/latent_vol_sp500.py`](./experiments/latent_vol_sp500.py)

If you want to run these scripts the following additional setup is required:
- a working installation of R, see https://www.r-project.org for download.  
- the R TMB package, see https://github.com/kaskr/adcomp/wiki/Download.  

We use PyMB (https://github.com/kforeman/PyMB) to wrap TMB in Python. Due to versions incompatibility, it had to be revised and is therefore directly included in the repository at  [`./PyMB/model.py`](./PyMB/model.py).

To check that the R/TMB/PyMB installation was successful, run

```
from PyMB.model import check_R_TMB
assert check_R_TMB()
```

To use the Laplace approximation from TMB, only the implementation of the joint log likelihood in C++ is required. This is done in [`./PyMB/likelihoods`](./PyMB/likelihoods).  
The final step is to compile these files: Just run the script [`./PyMB/compile.py`](./PyMB/compile.py).