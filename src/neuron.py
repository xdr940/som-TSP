import numpy as np

from distance import select_closest

def init_neurons(size):

    theta = np.linspace(0,2*np.pi,size)
    x = np.cos(theta)
    y = np.sin(theta)
    x = np.expand_dims(x,axis=1)
    y = np.expand_dims(y,axis=1)
    return np.concatenate([x,y],axis=1)+0.5




def save_neuron_chain(neuron_chain,path):
    neuron_chain_np = np.array(neuron_chain)
    np.save(path,neuron_chain_np)


def normalize(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)

def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)

def get_neighborhood(center, radix, domain):
    '''

    :param center: winner idx
    :param radix: 周围几个神经元
    :param domain: 总神经元个数
    :return:
    '''
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))

def get_route(cities, network):
    """Return the route computed by a network."""
    winners = cities[['x', 'y']].apply(
        lambda c: select_closest(network, c),
        axis=1, raw=True)
    cities['winner'] = winners
    ret = cities.sort_values('winner').index
    return ret


