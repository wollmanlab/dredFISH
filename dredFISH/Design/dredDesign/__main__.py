#!/usr/bin/env python

"""Design contains 
- PNMF
    - DPNMF
    - DPNMF-tree
- Neural network
    - different settings and versions 
- pre-processing
- post-processing (generate sequences)
"""

import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        prog='python dredDesign',
        description='dredFISH Design', 
        )

    parser.add_argument('method', choices=['PNMF', 'NNET',], 
                        help="choose PNMF or NNET",
                        )
    pnmf = parser.add_argument_group("PNMF options")
    pnmf.add_argument('aa', type=str, 
                        help="xxx",
                        )

    nnet = parser.add_argument_group("Neural network options")

    return parser 

if __name__ == "__main__":
    parser = create_parser() 
    args = parser.parse_args()

