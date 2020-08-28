
from sys import argv

import numpy as np
#iner
from neuron import init_neurons, get_neighborhood, get_route
from distance import select_closest, euclidean_distance,route_distance
from plot import plot_network, plot_route,plt_traj_p,plot_loss,plt_traj_np,plt_mtsp
from opts import OPT
import math
from dataloader import dataloader
import json
import pandas as pd
from path import Path
import matplotlib.pyplot as plt
from neuron import normalize,save_neuron_chain

from tqdm import  tqdm

def SOM(args):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtain the normalized set of cities (w/ coord in [0,1])
    iteration = args.iteration
    learning_rate = args.learning_rate
    decay = args.decay

    out_dir = Path(args.out_dir)
    out_dir.mkdir_p()
    cities = pd.read_csv(Path(args.data_dir)/'data1.csv')
    cities.to_csv(out_dir/'cities.csv')

    cities_nm = cities.copy()
    cities_nm[['x', 'y']] = normalize(cities_nm[['x', 'y']])
    cities_nm.to_csv(out_dir/'cities_nm.csv')


    # The population size is 8 times the number of cities
    n = cities_nm.shape[0] * 8

    # Generate an adequate network of neurons:
    neuron_chain = init_neurons(n)
    print('--> Network of {} neurons created. Starting the iterations:'.format(n))
    best_route=np.array([0])


    best_id=0
    min_loss=0
    losses={}
    losses_decay = {}

    for i in tqdm(range(iteration)):

        # Choose a random city
        city = cities_nm.sample(1)[['x', 'y']].values#随机抽样 random  sampling
        winner_idx = select_closest(neuron_chain, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(center=winner_idx, radix=n//10, domain=neuron_chain.shape[0])
        # Update the network's weights (closer to the city)
        neuron_chain += gaussian[:,np.newaxis] * learning_rate * (city - neuron_chain)
        # Decay the variables
        learning_rate = learning_rate * decay
        n = n * decay



        if i % args.evaluate_freq==0:
            route = get_route(cities_nm, neuron_chain)

            cities_od = cities.reindex(route)
            loss = route_distance(cities_od)
            losses[i] = loss

            if  min_loss==0 or min_loss > loss:
                min_loss=loss
                best_route = list(route.astype(np.float64))
                best_id = i
                losses_decay[i] = loss
                cities_od.to_csv(out_dir / 'route_{:04d}.csv'.format(i))
                save_neuron_chain(neuron_chain, out_dir / "neuron_chain_{:04d}.npy".format(i))
    #end for

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break


    print('Completed {} iterations.'.format(iteration))

    results={}
    results['min_loss'] = min_loss
    results['best_id'] = best_id
    results['best_route'] = best_route
    results['losses_decay'] = losses_decay
    results['losses'] = losses

    p = Path(out_dir / 'results.json')
    with open(p, 'w') as fp:
        json.dump(results, fp)
        print('ok')

    return results
