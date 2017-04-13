import gzip
import numpy as np

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

def load_idx_data(path, outer_shape=None, pos=None):
    """
    Returns an ndarray of data loaded from an IDX gzipped file,
    with a shape of outer_shape + data_shape.

    It is also possible to specify with pos the indices of the examples to retrieve.
    There are 4 different usecases, depending on the args specification:

    outer_shape == None, pos == None -- Load the entire dataset with outer_shape = (num_examples,)
    outer_shape != None, pos == None -- Load the first n examples, with n the size of outer_shape
    outer_shape == None, pos != None -- Load indices specified in pos with outer_shape = (len(pos),)
    outer_shape != None, pos != None -- Load indices specified in pos with specified outer_shape
    """
    idx_file = gzip.open(path)

    metadata = get_idx_metadata(idx_file)
    dtype = metadata[0]
    data_len = metadata[1][0]
    data_shape = metadata[1][1:]
    data_size = int(np.prod(data_shape))

    if pos is not None:
        assert len(pos) <= data_len, \
            'Number of indices must be smaller than %i' % data_len
        assert sorted(pos)[-1] <= data_len, \
            'Indices must be smaller than %i' % data_len
        
    if outer_shape is None:
        if pos is None:
            outer_shape = (data_len,)
            pos = np.arange(data_len)
        else:
            outer_shape = (len(pos),)
    else:
        outer_size = np.prod(outer_shape)
        assert outer_size <= data_len, \
            'Outer shape size must be smaller than %i' % data_len
        if pos is None:
            pos = np.arange(outer_size)
        else:
            assert outer_size == len(pos), \
                'Outer shape size must be equal to the number of indices'

    dtype_len = np.dtype(dtype).itemsize
    data = []
    for i in range(len(pos)):
        next_pos = pos[i]
        cur_pos = 0 if i == 0 else pos[i-1] + 1
        dist = (next_pos - cur_pos) * (dtype_len * data_size)
        idx_file.seek(dist, 1)
        data += [int.from_bytes(idx_file.read(dtype_len), byteorder='big')
                for j in range(data_size)]

    idx_file.close()

    return np.array(data, dtype).reshape(outer_shape + data_shape)
