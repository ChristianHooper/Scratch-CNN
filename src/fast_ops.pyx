# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as cnp
import cython

ctypedef cnp.float32_t f32


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[f32, ndim=4] max_pool4d(cnp.ndarray[f32, ndim=4] x, int reduction):
    """
    Max-pools a 4D tensor of shape (batch, channels, height, width) with a square window.

    Parameters
    ----------
    x : float32 array
        Input activations.
    reduction : int
        Pooling window/stride (e.g., 2 for 2x downsample).

    Returns
    -------
    float32 array
        Downsampled tensor.
    """
    cdef Py_ssize_t batch = x.shape[0]
    cdef Py_ssize_t channels = x.shape[1]
    cdef Py_ssize_t height = x.shape[2]
    cdef Py_ssize_t width = x.shape[3]
    cdef Py_ssize_t stride = reduction
    cdef Py_ssize_t out_h = height // stride
    cdef Py_ssize_t out_w = width // stride

    cdef cnp.ndarray[f32, ndim=4] out = np.empty((batch, channels, out_h, out_w), dtype=np.float32)

    cdef Py_ssize_t b, c, oh, ow, r, col
    cdef f32 current, candidate

    for b in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    current = x[b, c, oh * stride, ow * stride]
                    for r in range(stride):
                        for col in range(stride):
                            candidate = x[b, c, oh * stride + r, ow * stride + col]
                            if candidate > current:
                                current = candidate
                    out[b, c, oh, ow] = current

    return out

    #cpdef cnp.ndarray[f32, ndim=4] forward(cnp.ndarray[f32, ndim=4] x):
