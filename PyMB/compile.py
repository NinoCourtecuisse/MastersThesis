import argparse

from PyMB.model import check_R_TMB
from PyMB.model import model as PyMB_model

"""
Script to compile TMB custom models after writing the joint likelihood.
See https://github.com/kaskr/adcomp/wiki/Tutorial#writing-the-c-function for a tutorial.

Usage:
    ./run.sh PyMB/compile.py --file_name sv
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, choices=['sv', 'sabr'],
                        help='name of the .cpp file to compile, containing the model likelihood.')
    return parser.parse_args()

def compile(file_name):
    check_R_TMB()

    m = PyMB_model(name=file_name)
    filepath = f'PyMB/likelihoods/{file_name}.cpp'
    m.compile(filepath=filepath,
                output_dir='PyMB/likelihoods/tmb_tmp',
                use_R_compiler=True)

if __name__ == "__main__":
    args = parse_args()
    compile(args.file_name)
