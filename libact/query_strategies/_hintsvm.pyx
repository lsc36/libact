#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.stdlib cimport free
from . cimport _hintsvm

cdef extern from *:
    ctypedef struct svm_parameter:
        pass

np.import_array()

LIBSVM_KERNEL_TYPES = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

def hintsvm_query(
    np.ndarray[np.float64_t, ndim=2, mode='c'] X,
    np.ndarray[np.float64_t, ndim=1, mode='c'] y,
    np.ndarray[np.float64_t, ndim=1, mode='c'] w,
    np.ndarray[np.float64_t, ndim=2, mode='c'] X_pool,
    svm_params):

    # libsvm parameters
    cdef int svm_type=5
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] sample_weight=w,

    cdef str kernel=svm_params.pop('kernel', 'linear')
    cdef int degree=svm_params.pop('degree', 3)
    cdef double gamma=svm_params.pop('gamma', 0.1)
    cdef double coef0=svm_params.pop('coef0', 0.)
    cdef double tol=svm_params.pop('tol', 1e-3)
    cdef int shrinking=svm_params.pop('shrinking', 1)
    cdef double cache_size=svm_params.pop('cache_size', 100.)
    cdef double C=svm_params.pop('C', 0.1) # cl --> for hintsvm
    cdef int verbose=svm_params.pop('verbose', 0)

    # not used for now
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] class_weight=np.empty(0),
    cdef double epsilon=0.1
    cdef double nu=0.5
    cdef int probability=0
    cdef int max_iter=-1
    cdef int random_seed=0

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)

    cdef svm_parameter param
    cdef svm_problem problem
    cdef svm_model *model
    cdef const char *error_msg
    cdef np.npy_intp SV_len
    cdef np.npy_intp nr

    kernel_index = LIBSVM_KERNEL_TYPES.index(kernel)
    set_problem(
        &problem, X.data, y.data, w.data, X.shape, kernel_index)
    if problem.x == NULL:
        raise MemoryError("Seems we've run out of memory")

    set_parameter(
        &param, svm_type, kernel_index, degree, gamma, coef0, nu, cache_size,
        C, tol, epsilon, shrinking, probability, <int> class_weight.shape[0],
        class_weight_label.data, class_weight.data, max_iter, random_seed)
    set_verbosity(verbose)

    error_msg = svm_check_parameter(&problem, &param)
    if error_msg:
        # for SVR: epsilon is called p in libsvm
        error_repl = error_msg.decode('utf-8').replace("p < 0", "epsilon < 0")
        raise ValueError(error_repl)

    with nogil:
        model = svm_train(&problem, &param)

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] dec_values
    try:
        dec_values = np.empty((X_pool.shape[0], 1), dtype=np.float64)
        with nogil:
            rv = copy_predict_values(X_pool.data, model, X_pool.shape, dec_values.data, 1)
        if rv < 0:
            raise MemoryError("We've run out of memory")
    finally:
        svm_free_and_destroy_model(&model)
        free(problem.x)

    return dec_values
