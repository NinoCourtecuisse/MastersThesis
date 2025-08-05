import argparse

from PyMB.model import check_R_TMB
from PyMB.model import model as PyMB_model

"""
Script to compile TMB custom models after writing the joint likelihood.
See https://github.com/kaskr/adcomp/wiki/Tutorial#writing-the-c-function for a tutorial.

Usage:
    ./run.sh PyMB/compile.py
"""

def compile():
    check_R_TMB()

    file_names = ['sv', 'sabr']
    for name in file_names:
        m = PyMB_model(name=name)
        filepath = f'PyMB/likelihoods/{name}.cpp'
        m.compile(filepath=filepath,
                    output_dir='PyMB/likelihoods/tmb_tmp',
                    use_R_compiler=True)

if __name__ == "__main__":
    compile()
