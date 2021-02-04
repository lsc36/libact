#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>

/* DGESVD prototype */
extern void LAPACK_dgesvd( char const* jobu, char const* jobvt, lapack_int const* m, lapack_int const* n, double* a,
                    lapack_int const* lda, double* s, double* u, lapack_int const* ldu, double* vt, lapack_int const* ldvt,
                    double* work, lapack_int const* lwork, lapack_int* info );

double** An(double *pi, double *x, int labs, int dims);
double** A(double **PI, double **X, int labs, int dims, int n_pool);
double** Fisher(double *pi, double *x, double sigma, int labs, int dims);

static char estVar_docstring[] =
    "Calculate the A and Fisher matrix.";

static PyObject *varRedu_estVar(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"estVar", varRedu_estVar, METH_VARARGS, estVar_docstring},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_variance_reduction",  /* m_name */
    "This module provides calculate A and Fisher matrix using C.",  /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

PyMODINIT_FUNC PyInit__variance_reduction(void){
    PyObject *m = PyModule_Create(&moduledef);
    if(m==NULL){
        return NULL;
    }

    /* Load 'numpy' */
    import_array();

    return m;
}
#else
PyMODINIT_FUNC init_variance_reduction(void){
    PyObject *module = Py_InitModule("_variance_reduction", module_methods);
    import_array();
}
#endif

double* matrix_mul(double* a, double* b, int m1, int n1, int m2, int n2){
    double *ret = (double*) malloc(m1 * n2 * sizeof(double));
    if(n1 != m2){
        return NULL;
    }
    for(int i=0; i<m1; i++)
        for(int j=0; j<n2; j++){
            double temp = 0.0;
            for(int p=0; p<n1; p++)
                temp += a[i*n1 + p] * b[p*n2 + j];
            ret[i*n2 + j] = temp;
        }
    return ret;
}


void pinv(double** X, int labs, int dims){
    int m = labs*dims, n = labs*dims, lda = labs*dims, ldu = labs*dims, 
        ldvt = labs*dims, lwork, info;
    double wkopt;
    double *work;
    double *s  = (double*) malloc(labs*dims * sizeof(double));
    double *u  = (double*) malloc(labs*dims * labs*dims * sizeof(double));
    double *vt = (double*) malloc(labs*dims * labs*dims * sizeof(double));
    double *a  = (double*) malloc(labs*dims * labs*dims * sizeof(double));

    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            a[i*labs*dims + j] = X[i][j];

     /* Query and allocate the optimal workspace */
    lwork = -1;
    LAPACK_dgesvd("All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
        &info);
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    /* Compute SVD */
    LAPACK_dgesvd("All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
        &info);
    /* Check for convergence  */
    if(info > 0) {
        printf("The algorithm computing SVD failed to converge. %d\n", info);
    }
    if(info < 0) {
        printf("Has illegal value. %d\n", info);
    }

    int numSigular = 0;
    double *si  = (double*) malloc(labs*dims * labs*dims * sizeof(double));
    memset(si, 0, labs*dims * labs*dims * sizeof(double));

    for(int i=0; i<ldu; i++){
        if(s[i] > 1e-30){
            si[i*ldu + i] = 1.0 / s[i];
            numSigular += 1;
        }else{
            si[i+ldu + i] = 0.0;
        }
    }

    /* calculating transpose */
    double *ret = matrix_mul(vt, si, labs*dims, numSigular, numSigular, numSigular);
    double *ret_pinv = matrix_mul(ret, u, labs*dims, numSigular, numSigular, labs*dims);

    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            X[i][j] = ret_pinv[i*n + j];

    free(ret);
    free(ret_pinv);
    free(work);
    free(a);
    free(s);
    free(vt);
    free(u);
    free(si);

    return;
}

static PyObject *varRedu_estVar(PyObject *self, PyObject *args)
{
    int dims, n_pool, labs, sigma;
    PyObject *PI_obj, *X_obj, *ePI_obj, *eX_obj;

    if (!PyArg_ParseTuple(args, "dOOOO", &sigma, &PI_obj, &X_obj, &ePI_obj, &eX_obj))
        return NULL;

    PyArrayObject *PI_array  =  (PyArrayObject*)PyArray_FROM_OTF(PI_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array   =  (PyArrayObject*)PyArray_FROM_OTF(X_obj,   NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ePI_array =  (PyArrayObject*)PyArray_FROM_OTF(ePI_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *eX_array  =  (PyArrayObject*)PyArray_FROM_OTF(eX_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (PI_array == NULL || X_array == NULL || ePI_array == NULL || eX_array == NULL) {
        Py_XDECREF(PI_array);
        Py_XDECREF(X_array);
        Py_XDECREF(ePI_array);
        Py_XDECREF(eX_array);
        return NULL;
    }

    labs   = (int)PyArray_DIM(PI_array, 1);
    n_pool = (int)PyArray_DIM(X_array, 0);
    dims   = (int)PyArray_DIM(X_array, 1);

    double **PI  =  (double**) malloc(n_pool * sizeof(double*));
    double **X   =  (double**) malloc(n_pool * sizeof(double*));
    for(int i=0; i<n_pool; i++){
        PI[i] = (double*) malloc(labs * sizeof(double));
        X[i]  = (double*) malloc(dims * sizeof(double));
    }
    for(int i=0; i<n_pool; i++){
        for(int j=0; j<labs; j++){
            PI[i][j] = *(double*)PyArray_GETPTR2(PI_array, i, j);
        }
        for(int j=0; j<dims; j++){
            X[i][j] = *(double*)PyArray_GETPTR2(X_array, i, j);
        }
    }
    double *ePI =  (double*) PyArray_DATA(ePI_array);
    double *eX  =  (double*) PyArray_DATA(eX_array);

    double **retF = Fisher(ePI, eX, sigma, labs, dims);
    double **retA = A(PI, X, labs, dims, n_pool);

    pinv(retF, labs, dims);

    /* calculates the trace of the multiply of pinv(F) and A  */
    double score = 0.0;
    for(int i=0; i<dims*labs; i++){
        for(int k=0; k<dims*labs; k++){
            score += retA[i][k] * retF[k][i];
        }
    }

    Py_DECREF(PI_array);
    Py_DECREF(X_array);
    Py_DECREF(ePI_array);
    Py_DECREF(eX_array);

    PyObject* ret = Py_BuildValue("d", score);

    for(int i=0; i<n_pool; i++){
        free(PI[i]);
        free(X[i]);
    }
    free(PI);
    free(X);

    for(int i=0; i<labs*dims; i++){
        free(retF[i]);
        free(retA[i]);
    }
    free(retF);
    free(retA);

    return ret;
}

double** An(double *pi, double *x, int labs, int dims){
    double **g = (double**) malloc(labs*dims * sizeof(double*));
    for(int i=0; i<labs*dims; i++){
        g[i] = (double*) malloc(labs * sizeof(double));
        memset(g[i], 0, labs * sizeof(double));
    }
    
    for(int p=0; p<labs; p++)
        for(int i=0; i<dims; i++){
            for(int c=0; c<labs; c++)
                if(p == c) g[p*dims + i][c] = pi[p] * (1.0-pi[p]) * x[i];
                else g[p*dims + i][c] = -1.0 * pi[p] * pi[c] * x[i];
        }

    double **an = (double**) malloc(labs*dims * sizeof(double*));
    for(int i=0; i<labs*dims; i++){
        an[i] = (double*) malloc(labs*dims * sizeof(double));
        memset(an[i], 0, labs*dims * sizeof(double));
    }
    
    for(int p=0; p<labs; p++)
        for(int i=0; i<dims; i++)
            for(int q=0; q<labs; q++)
                for(int j=0; j<dims; j++){
                    /* inner product */
                    double tmp = 0.0;
                    for(int k=0; k<labs; k++){
                        tmp += g[p*dims + i][k] * g[q*dims + j][k];
                    }
                    an[p*dims + i][q*dims + j] = tmp;
                }

    for(int i=0; i<labs*dims; i++)
        free(g[i]);
    free(g);

    return an;
}

double** A(double **PI, double **X, int labs, int dims, int n_pool){
    double **ret = (double**) malloc(labs*dims * sizeof(double*));
    for(int i=0; i<labs*dims; i++){
        ret[i] = (double*) malloc(labs*dims * sizeof(double));
        memset(ret[i], 0, labs*dims * sizeof(double));
    }

    for(int n=0; n<n_pool; n++){
        double **an = An(PI[n], X[n], labs, dims);

        for(int p=0; p<labs; p++)
            for(int i=0; i<dims; i++)
                for(int q=0; q<labs; q++)
                    for(int j=0; j<dims; j++)
                        ret[p*dims + i][q*dims + j] += an[p*dims + i][q*dims + j];
        for(int i=0; i<labs*dims; i++)
            free(an[i]);
        free(an);
    }
    return ret;
}

double** Fisher(double *pi, double *x, double sigma, int labs, int dims){
    double **ret = (double**) malloc(labs*dims * sizeof(double*));
    for(int i=0; i<labs*dims; i++){
        ret[i] = (double*) malloc(labs*dims * sizeof(double*));
        memset(ret[i], 0, labs*dims * sizeof(double));
    }

    for(int p=0; p<labs; p++)
        for(int i=0; i<dims; i++)
            for(int q=0; q<labs; q++)
                for(int j=0; j<dims; j++){
                    if(p == q && i == j)
                        ret[p*dims + i][q*dims + j] = x[i]*x[i]*pi[p]*(1.0-pi[p]) + 1.0/sigma*sigma;
                    else if(p == q && i != j)
                        ret[p*dims + i][q*dims + j] = x[i]*x[j]*pi[p]*(1.0-pi[p]);
                    else
                        ret[p*dims + i][q*dims + j] = x[i]*x[j]*pi[p]*pi[q];
                }
    return ret;
}
