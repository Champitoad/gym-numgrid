import numpy as np

def total_discrete_mapping(multi_discrete):
    """
    Returns a total discrete mapping over a MultiDiscrete space,
    associating a unique number to each element of the sub-spaces cartesian product.
    """
    xi = [np.arange(n+1) for n in multi_discrete.high]
    cartesian_product = np.array(np.meshgrid(*xi, indexing='ij')).T.reshape(-1,len(xi))
    return dict(enumerate(cartesian_product.tolist()))
