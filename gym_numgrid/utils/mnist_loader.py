import gzip
import numpy as np
from functools import reduce

def get_idx_metadata(idx_file):
    """
    Retrieves the metadata contained in the header of an IDX file.
    """
    magic_number = idx_file.read(4)
    dtype = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64
            }[magic_number[2]]
    ndims = magic_number[3]
    shape = tuple(int.from_bytes(idx_file.read(4), byteorder='big')
            for i in range(ndims))

    return (dtype, shape)

def load_idx_data(path, outer_shape=None):
    """
    Returns an ndarray of data loaded from an IDX gzipped file,
    with a shape of outer_shape + data_shape.
    """
    idx_file = gzip.open(path)

    metadata = get_idx_metadata(idx_file)
    dtype = metadata[0]
    data_len = metadata[1][0]
    if outer_shape == None:
        outer_shape = (data_len,)
    outer_shape = tuple(outer_shape)
    data_shape = metadata[1][1:]

    assert reduce(lambda a,b: a*b, outer_shape) <= data_len, \
        'Outer shape size must be smaller than %i' % data_len

    shape = outer_shape + data_shape
    size = reduce(lambda a,b: a*b, shape)

    dtype_len = np.dtype(dtype).itemsize
    data = np.array([int.from_bytes(idx_file.read(dtype_len), byteorder='big')
                    for i in range(size)], dtype)
    
    return data.reshape(shape)
