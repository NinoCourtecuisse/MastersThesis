from PyMB.model import check_R_TMB
from PyMB.model import model as PyMB_model

"""
Script to compile TMB custom models.
See https://github.com/kaskr/adcomp/wiki/Tutorial#writing-the-c-function for a tutorial.
"""

def compile(name):
    check_R_TMB()

    m = PyMB_model(name=name)
    filepath = f'PyMB/likelihoods/{name}.cpp'
    m.compile(filepath=filepath,
                output_dir='PyMB/likelihoods/tmb_tmp',
                use_R_compiler=True)

compile('sv')