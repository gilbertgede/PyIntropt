# Adapted from:
# CHOLMOD wrapper for scikits.sparse
# to now include a
# SuiteSparseQR wrapper

# Original license:
# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials
#   provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

import warnings
from libc cimport stdlib
cimport cpython as py
import numpy as np
cimport numpy as np
from scipy import sparse

np.import_array()

cdef extern from "numpy/arrayobject.h":
    # Cython 0.12 complains about PyTypeObject being an "incomplete type" on
    # this line:
    #py.PyTypeObject PyArray_Type
    # So use a hack:
    struct MyHackReallyPyTypeObject:
        pass
    MyHackReallyPyTypeObject PyArray_Type
    object PyArray_NewFromDescr(MyHackReallyPyTypeObject * subtype,
                                np.dtype descr,
                                int nd,
                                np.npy_intp * dims,
                                np.npy_intp * strides,
                                void * data,
                                int flags,
                                object obj)
    # This is ridiculous: the description of PyArrayObject in numpy.pxd does
    # not mention the 'base' member, so we need a separate wrapper just to
    # expose it:
    ctypedef struct ndarray_with_base "PyArrayObject":
        void * base

    # In Cython 0.14.1, np.NPY_F_CONTIGUOUS is broken, because numpy.pxd
    # claims that it is of a non-existent type called 'enum requirements', and
    # new versions of Cython attempt to stash it in a temporary variable of
    # this type, which then annoys the C compiler.
    enum:
        NPY_F_CONTIGUOUS

cdef inline np.ndarray set_base(np.ndarray arr, object base):
    cdef ndarray_with_base * hack = <ndarray_with_base *> arr
    py.Py_INCREF(base)
    hack.base = <void *> base
    return arr

cdef extern from "cholmod.h":
    cdef enum:
        CHOLMOD_INT
        CHOLMOD_PATTERN, CHOLMOD_REAL, CHOLMOD_COMPLEX
        CHOLMOD_DOUBLE
        CHOLMOD_AUTO, CHOLMOD_SIMPLICIAL, CHOLMOD_SUPERNODAL
        CHOLMOD_OK, CHOLMOD_NOT_POSDEF
        CHOLMOD_A, CHOLMOD_LDLt, CHOLMOD_LD, CHOLMOD_DLt, CHOLMOD_L
        CHOLMOD_Lt, CHOLMOD_D, CHOLMOD_P, CHOLMOD_Pt

    ctypedef struct cholmod_common:
        int supernodal
        int status
        int print_ "print"
        void (*error_handler)(int status, char * file, int line, char * msg)

    # int versions
    int cholmod_start(cholmod_common *) except? 0
    int cholmod_finish(cholmod_common *) except? 0
    int cholmod_check_common(cholmod_common *) except? 0
    int cholmod_print_common(char *, cholmod_common *) except? 0
    # long versions
    int cholmod_l_start(cholmod_common *) except? 0
    int cholmod_l_finish(cholmod_common *) except? 0
    int cholmod_l_check_common(cholmod_common *) except? 0
    int cholmod_l_print_common(char *, cholmod_common *) except? 0

    ctypedef struct cholmod_sparse:
        size_t nrow, ncol, nzmax
        void * p # column pointers
        void * i # row indices
        void * x
        int stype # 0 = regular, >0 = upper triangular, <0 = lower triangular
        int itype # type of p, i, nz
        int xtype
        int dtype
        int sorted
        int packed


    cholmod_sparse *cholmod_allocate_sparse (size_t, size_t, size_t, int, int,
                                             int, int, cholmod_common *)
    cholmod_sparse *cholmod_l_allocate_sparse (size_t, size_t, size_t, int,
                                               int, int, int, cholmod_common *)

    # int versions
    int cholmod_free_sparse(cholmod_sparse **, cholmod_common *) except? 0
    int cholmod_check_sparse(cholmod_sparse *, cholmod_common *) except? 0
    int cholmod_print_sparse(cholmod_sparse *, char *, cholmod_common *) except? 0
    # long versions
    int cholmod_l_free_sparse(cholmod_sparse **, cholmod_common *) except? 0
    int cholmod_l_check_sparse(cholmod_sparse *, cholmod_common *) except? 0
    int cholmod_l_print_sparse(cholmod_sparse *, char *, cholmod_common *) except? 0

    ctypedef struct cholmod_dense:
        size_t nrow, ncol, nzmax
        size_t d
        void * x
        int xtype, dtype

    # int versions
    int cholmod_free_dense(cholmod_dense **, cholmod_common *) except? 0
    int cholmod_check_dense(cholmod_dense *, cholmod_common *) except? 0
    int cholmod_print_dense(cholmod_dense *, char *, cholmod_common *) except? 0
    # long versions
    int cholmod_l_free_dense(cholmod_dense **, cholmod_common *) except? 0
    int cholmod_l_check_dense(cholmod_dense *, cholmod_common *) except? 0
    int cholmod_l_print_dense(cholmod_dense *, char *, cholmod_common *) except? 0

    cholmod_dense * cholmod_allocate_dense(size_t, size_t, size_t, int,
                                           cholmod_common *)
    cholmod_dense * cholmod_l_allocate_dense(size_t, size_t, size_t, int,
                                             cholmod_common *)


cdef extern from "SuiteSparseQR_C.h":
    long SuiteSparseQR_C_QR(int o, double tol, long m, cholmod_sparse *,
                            cholmod_sparse **, cholmod_sparse **, long **,
                            cholmod_common *)# except? 0

    cholmod_dense * SuiteSparseQR_C_backslash(int o, double tol,
                                              cholmod_sparse *,
                                              cholmod_dense *,
                                              cholmod_common *)# except? NULL
    cholmod_dense * SuiteSparseQR_C_backslash_default(cholmod_sparse *,
                                                      cholmod_dense *,
                                                      cholmod_common *)

#cdef class Common

class CholmodError(Exception):
    pass

cdef object _integer_py_dtype = np.dtype(np.int64)
#assert sizeof(int) == _integer_py_dtype.itemsize == 8

cdef _require_1d_integer(a):
    a = np.ascontiguousarray(a, dtype=_integer_py_dtype)
    assert a.ndim == 1
    return a

cdef object _real_py_dtype = np.dtype(np.float64)
assert sizeof(double) == _real_py_dtype.itemsize == 8
cdef object _complex_py_dtype = np.dtype(np.complex128)
assert _complex_py_dtype.itemsize == 2 * sizeof(double) == 16

##########
# Cholmod -> Python conversion:
##########

cdef np.dtype _np_dtype_for(int xtype):
    if xtype == CHOLMOD_COMPLEX:
        py.Py_INCREF(_complex_py_dtype)
        return _complex_py_dtype
    elif xtype == CHOLMOD_REAL:
        py.Py_INCREF(_real_py_dtype)
        return _real_py_dtype
    else:
        raise CholmodError, "cholmod->numpy type conversion failed"


def QR(A, P=False):
    m, n = A.shape
    cdef int order = 4
    cdef long * perm = <long *>stdlib.malloc(n * sizeof(long))
    if P is False:
        order = 0
        perm = NULL
    cdef long tol = -2
    cdef cholmod_common Com, *cc
    cc = &Com
    cholmod_l_start(cc)
    cdef cholmod_sparse * c_Q
    cdef cholmod_sparse * c_R

    """    Code for 'viewing' the sparse matrix    """
    A.sort_indices()
    cdef cholmod_sparse * c_A = cholmod_l_allocate_sparse(<size_t>A.shape[0], <size_t>A.shape[1], <size_t>A.nnz, 1, 1, 0, CHOLMOD_REAL, cc)
    cdef np.ndarray i1 = _require_1d_integer(A.indptr)
    cdef np.ndarray i2 = _require_1d_integer(A.indices)
    cdef np.ndarray i3 = np.asfortranarray(A.data, dtype=_real_py_dtype)
    c_A.p = i1.data
    c_A.i = i2.data
    c_A.x = i3.data
    """    end code for allocating sparse matrix    """

    rank = SuiteSparseQR_C_QR(order, tol, A.shape[0], c_A, &c_Q, &c_R, &perm, cc)

    """    Code to get a scipy sparse matrix    """
    shape = (c_Q.nrow, c_Q.ncol)
    py.Py_INCREF(_integer_py_dtype)
    cdef np.npy_intp ncol_plus_1 = c_Q.ncol + 1
    indptr = PyArray_NewFromDescr(&PyArray_Type, _integer_py_dtype, 1,
                                  &ncol_plus_1, NULL, c_Q.p, NPY_F_CONTIGUOUS,
                                  None)
    py.Py_INCREF(_integer_py_dtype)
    cdef np.npy_intp nzmax = c_Q.nzmax
    indices = PyArray_NewFromDescr(&PyArray_Type, _integer_py_dtype, 1, &nzmax,
                                   NULL, c_Q.i, NPY_F_CONTIGUOUS, None)
    data_dtype = _np_dtype_for(c_Q.xtype)
    data = PyArray_NewFromDescr(&PyArray_Type, data_dtype, 1, &nzmax, NULL,
                                c_Q.x, NPY_F_CONTIGUOUS, None)
    Q = sparse.csc_matrix((data, indices, indptr), shape=shape)
    """    end code to get a scipy sparse matrix    """

    """    Code to get a scipy sparse matrix    """
    shape = (c_R.nrow, c_R.ncol)
    py.Py_INCREF(_integer_py_dtype)
    ncol_plus_1 = c_R.ncol + 1
    indptr = PyArray_NewFromDescr(&PyArray_Type, _integer_py_dtype, 1,
                                  &ncol_plus_1, NULL, c_R.p, NPY_F_CONTIGUOUS,
                                  None)
    py.Py_INCREF(_integer_py_dtype)
    nzmax = c_R.nzmax
    indices = PyArray_NewFromDescr(&PyArray_Type, _integer_py_dtype, 1, &nzmax,
                                   NULL, c_R.i, NPY_F_CONTIGUOUS, None)
    data_dtype = _np_dtype_for(c_R.xtype)
    data = PyArray_NewFromDescr(&PyArray_Type, data_dtype, 1, &nzmax, NULL,
                                c_R.x, NPY_F_CONTIGUOUS, None)
    R = sparse.csc_matrix((data, indices, indptr), shape=shape)
    """    end code to get a scipy sparse matrix    """

    """    Get Permutation info    """
    cdef np.npy_intp nn = n
    if P:
        py.Py_INCREF(_integer_py_dtype)
        perm_ret = PyArray_NewFromDescr(&PyArray_Type, _integer_py_dtype, 1, &nn,
                                        NULL, perm, NPY_F_CONTIGUOUS, None)
    """    End get permutation info """

    cholmod_l_finish(cc)

    if P:
        return Q, R, rank, perm_ret
    else:
        return Q, R, rank


def qr_solve(A, b):
    cdef int order = 0
    cdef double tol = -2
    cdef cholmod_common Com, *cc
    cdef cholmod_dense * c_x

    cc = &Com
    cholmod_l_start(cc)

    """    Code for 'viewing' the sparse matrix    """
    A.sort_indices()
    cdef cholmod_sparse * c_A = cholmod_l_allocate_sparse(<size_t>A.shape[0], <size_t>A.shape[1], <size_t>A.nnz, 1, 1, 0, CHOLMOD_REAL, cc)
    cdef np.ndarray i1 = _require_1d_integer(A.indptr)
    cdef np.ndarray i2 = _require_1d_integer(A.indices)
    cdef np.ndarray i3 = np.asfortranarray(A.data, dtype=_real_py_dtype)
    c_A.p = i1.data
    c_A.i = i2.data
    c_A.x = i3.data
    """    end code for allocating sparse matrix    """

    """    Code for 'viewing' the dense matrix    """
    cdef cholmod_dense * c_b = cholmod_l_allocate_dense(<size_t>b.shape[0], <size_t>b.shape[1], <size_t>b.shape[0], CHOLMOD_REAL, cc)
    cdef np.ndarray i4 = np.asfortranarray(b.data, dtype=_real_py_dtype)
    c_b.x = i4.data
    """    end code for allocating dense matrix    """

    c_x = SuiteSparseQR_C_backslash_default(c_A, c_b, cc)

    """    Code to get a numpy dense matrix    """
    cdef np.dtype dtype = _np_dtype_for(c_x.xtype)
    cdef np.npy_intp dims[2]
    dims[0] = c_x.nrow
    dims[1] = c_x.ncol
    out = PyArray_NewFromDescr(&PyArray_Type, dtype, 2, dims, NULL, c_x.x,
                               NPY_F_CONTIGUOUS, None)
    cholmod_l_free_dense(&c_x, cc)
    """    end code for allocating numpy matrix    """

    cholmod_l_finish(cc)
    return out
