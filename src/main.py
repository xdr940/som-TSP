from sys import argv

import numpy as np
#iner
from neuron import generate_network, get_neighborhood, get_route
from plot import plot_neuron_chain, plot_route,plot_loss
from opts import OPT
from dataloader import dataloader
import pandas as pd
from path import Path
from som import SOM

import matplotlib.pyplot as plt





def main():
    pass
if __name__ == '__main__':
    args = OPT().args()
    SOM(args)

    plot_loss(input_dir=args.out_dir)
    plot_route(input_dir=args.out_dir)
    plot_neuron_chain(input_dir=args.out_dir)




