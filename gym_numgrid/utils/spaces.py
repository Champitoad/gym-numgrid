import numpy as np

def total_discrete_mapping(multi_discrete):
    """
    Returns a total discrete mapping over a MultiDiscrete space,
    associating a unique number to each element of the sub-spaces cartesian product.
    """
    xi = [np.arange(space.n) for space in multi_discrete.spaces]
    cartesian_product = np.array(np.meshgrid(*xi, indexing='ij')).T.reshape(-1,len(xi))[:,::-1]
    return dict(enumerate(cartesian_product))
