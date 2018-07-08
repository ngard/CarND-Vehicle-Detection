import cython
cimport numpy as np

@cython.locals(
hist_features=np.ndarray)
cpdef np.ndarray[np.float32_t] color_hist(np.ndarray[np.float32_t, ndim=3] img, int nbins=?, tuple bins_range=?)