/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : matrix.h
Last Update : Mar 28, 2016
Description : Provide interfaces of general purpose of matrix in CUDA
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */


#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include "stacktrace.h"

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#ifdef XT
    #include <cublasXt.h>
#endif

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cassert>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;


#define CU1DBLOCK 256
#define CU2DBLOCK 16

#define CUDA_CHECK( error, name )       __cudaCheck( error, __FILE__, __LINE__, name )
#define DRIVER_CHECK( result, name )    __driverCheck( result, __FILE__, __LINE__, name )
#define KERNEL_CHECK( name )            __kernelCheck( __FILE__, __LINE__, name )
#define CUBLAS_CHECK( status, name )    __cublasCheck( status, __FILE__, __LINE__, name )
#define CURAND_CHECK( status, name )    __curandCheck( status, __FILE__, __LINE__, name )
#define CUSPARSE_CHECK( status, name )  __cusparseCheck( status, __FILE__, __LINE__, name )
#define ASSERT( status )                __assert( status, __FILE__, __LINE__ );



inline string CurrentTime() {
    time_t t = time( NULL );
    char* str = asctime( localtime( &t ) );
    str[ strlen(str) -1 ] = 0;
    return string( str );
}



inline void __assert( bool status, const char* file, int line ) 
{                               
    if ( !status ) 
    {                
        fprintf( stderr, "FAIL at Line %4d in %10s\n", line, file );                     
        print_stacktrace();   
        exit( EXIT_FAILURE );                       
    }                                               
}      


inline void __cusparseCheck
(
            cusparseStatus_t    status,
    const   char*               file,
            int                 line,
    const   char*               name
) 
{
#ifdef DEBUG
    switch( status ) {
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << endl; 
            break;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            cout << "CUSPARSE_STATUS_ALLOC_FAILED" << endl; 
            break;
        case CUSPARSE_STATUS_INVALID_VALUE:
            cout << "CUSPARSE_STATUS_INVALID_VALUE" << endl; 
            break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << endl; 
            break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            cout << "CUSPARSE_STATUS_EXECUTION_FAILED" << endl; 
            break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << endl; 
            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << endl; 
            break;
    }
#endif
    if ( status != CUSPARSE_STATUS_SUCCESS ) 
    {
        fprintf( stderr, "%10s %4d: %s failed\n", file, line, name );
        print_stacktrace();
        exit( EXIT_FAILURE );
    }
}                         



inline void __cudaCheck
(
            cudaError_t     error,
    const   char*           file,
            int             line,
    const   char*           name
)
{
    if ( error != cudaSuccess )
    {
        fprintf( stderr, "!!! %s !!!  ", cudaGetErrorString( error ) );
        fprintf( stderr, "%10s %4d: %s failed\n", file, line, name );
        print_stacktrace();
        exit( EXIT_FAILURE );
    }
}   // end of __cudaCheck



inline void __driverCheck
(
            CUresult        error,
    const   char*           file,
            int             line,
    const   char*           name
)
{
    if ( error != CUDA_SUCCESS )
    {
        fprintf( stderr, "%10s %4d: %s failed\n", file, line, name );
        print_stacktrace();
        exit( EXIT_FAILURE );
    }
}   // end of __cudaCheck



inline void __curandCheck
(
            curandStatus_t  status,
    const   char*           file,
            int             line,
    const   char*           name
)
{
    if ( status != CURAND_STATUS_SUCCESS )
    {
        fprintf( stderr, "%10s %4d: %s failed\n", file, line, name );
        print_stacktrace();
        exit( EXIT_FAILURE );
    }
}   // end of __curandCheck



inline void __cublasCheck
(
            cublasStatus_t  status,
    const   char*           file,
            int             line,
    const   char*           name
)
{
#ifdef DEBUG
    switch ( status ) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            fprintf( stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n" );
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            fprintf( stderr, "CUBLAS_STATUS_INVALID_VALUE\n" );
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            fprintf( stderr, "CUBLAS_STATUS_ARCH_MISMATCH\n" );
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            fprintf( stderr, "CUBLAS_STATUS_EXECUTION_FAILED\n" );
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            fprintf( stderr, "CUBLAS_STATUS_ALLOC_FAILED\n" );
            break;
        case CUBLAS_STATUS_LICENSE_ERROR:
            fprintf( stderr, "CUBLAS_STATUS_LICENSE_ERROR\n" );
            break;
    }
#endif
    if ( status != CUBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "%10s %4d: %s failed\n", file, line, name );
        print_stacktrace();
        exit( EXIT_FAILURE );
    }
}   // end of __cublasCheck



inline void __kernelCheck
(
    const   char*   file,
            int     line,
    const   char*   name
)
{
// waiting for a kernel's return may significantly degrade the performance
#ifdef KERNEL_DEBUG
    cudaError_t error = cudaGetLastError();
    if ( error !=  cudaSuccess )
    {
        fprintf( stderr, "%s  ", cudaGetErrorString( error ) );
        fprintf( stderr, "%10s %4d: %s failed\n", file, line, name );
        print_stacktrace();
        exit( EXIT_FAILURE );
    }

    // a kernel may returns asynchronously
    error = cudaDeviceSynchronize();
    if ( error !=  cudaSuccess )
    {
        fprintf( stderr, "%s  ", cudaGetErrorString( error ) );
        fprintf( stderr, "%10s %4d: %s failed\n", file, line, name );
        print_stacktrace();
        exit( EXIT_FAILURE );
    }
#endif
}   // end of __kernelCheck


// ==========================================================================================


inline int GridDim( int size, int blockSize ) {
    return size / blockSize + (size % blockSize ? 1 : 0);
}


struct MatrixShape {
    int row;
    int column;
    int stride;
};


struct MatrixRange {
    int startRow;
    int endRow;
    int startColumn;
    int endColumn;
    MatrixRange( int inStartRow, int inEndRow, int inStartColumn, int inEndColumn ) :
        startRow(inStartRow), endRow(inEndRow), 
        startColumn(inStartColumn), endColumn(inEndColumn) {}
};


enum FunctionType {
    kSigmoid,   // 0
    kRelu,
    kExp,
    kReciprocal,
    kLog,
    kFill,
    kScale,
    kShift,
    kClip,

    kSetZero,   // 9
    kUndefined,

    kDiffSigmoid, // 11
    kDiffReLU,
    kDiffNCE,
    kLowerBound,

    kGetRows,
    kGetColumns,
    kAddRows,
    kAddColumns, 

    kMaxReduce,
    kMinReduce,
    kSumReduce
};

// ==========================================================================================

/* CSR format is used since all cuSPARSE function takes CSR. */
class MatrixBase;
class Matrix;
class SparseMatrix {
private:
    int     m_nnz;
    int        m_n_row;
    int     m_n_column;
    float*     m_data;
    int*     m_indices;
    int*    m_indptr;
    int     m_n_byte;
    
protected:
    static cusparseHandle_t handle;
    static cusparseMatDescr_t descriptor;
    static int n_matrix;
    static Matrix* buffer;      // also makes the GPU assignment consistent
    
public:
    friend class MatrixBase;
    friend class Matrix;
    friend class SubMatix;

    SparseMatrix();

    SparseMatrix( const vector<float>& data, 
                  const vector<int>& indices, 
                  const vector<int>& indptr,
                  int n_column = 0 );
                  
    virtual ~SparseMatrix();
    
    void SetData( const vector<float>& data, 
                  const vector<int>& indices, 
                  const vector<int>& indptr,
                  int n_column = 0 );
    
    inline int Rows() const {
        return m_n_row;
    }


    inline int Columns() const {
        return m_n_column;
    }
};    // end of SparseMatrix


// ==========================================================================================

class MatrixBase {
private:
    float* m_data;
    size_t m_n_row;
    size_t m_n_column;
    size_t m_n_stride;
    size_t m_n_allocated;
    
    
protected:
    static float* ones;
    static float* zeros;
    static float* buffer;
    static int buffer_size;
    static int n_matrix;
    static cublasHandle_t handle;
    static curandGenerator_t generator;
    static cudaStream_t stream;

#ifdef XT
    static cublasXtHandle_t xt_handle;
#endif


    MatrixBase()  : m_data( NULL ), m_n_row( 0 ), m_n_column( 0 ), m_n_stride( 0 ), m_n_allocated( 0 ) {
    }   // end of MatrixBase

    void m_clear_buffer();

public:
    friend class Matrix;
    friend class SubMatrix;
    friend class SparseMatrix;

    virtual ~MatrixBase() {
    }   // end of ~MatrixBase


    inline int Rows() const {
        return m_n_row;
    }


    inline int Columns() const {
        return m_n_column;
    }


    inline int Stride() const {
        return m_n_stride;
    }


    bool ValidateRange( MatrixRange range ) const;
    
    void GetData( vector<float>* data ) const;
    
    void SetData( const vector<float>& data );

    void SetDataAsync( const vector<float>& data );

    string ToString() const;
    
    MatrixShape Shape() const;

    // this = beta * this + alpha * dot( A, B )
    // actual computation: C.T = B.T * A.T
    void Sgemm( float beta, float alpha,
                const MatrixBase& A, cublasOperation_t transa,
                const MatrixBase& B, cublasOperation_t transb );
                
    void Sgemm( float beta, float alpha,
                const MatrixBase& A, cusparseOperation_t transa,
                const SparseMatrix& B, cusparseOperation_t transb );

    void Sgemm( float beta, float alpha,
                const SparseMatrix& A, cusparseOperation_t transa,
                const MatrixBase& B, cusparseOperation_t transb );

    void Strmm( float alpha,
                const MatrixBase& A, cublasSideMode_t side, cublasFillMode_t uplo,
                const MatrixBase& B, cublasOperation_t trans = CUBLAS_OP_N );

    // this = value
    void Fill( float value );

    // this *= value
    void Scale( float value );

    // this += value
    void Shift( float value ); 

    // this = max(0, value)
    void ReLU( const MatrixBase& value );

    // this = (value > 0) * diff
    void DiffReLU( const MatrixBase& value, const MatrixBase& diff );

    // this = 1 / (1 + exp(-value))
    void Sigmoid( const MatrixBase& value );

    // this = value * (1 - value) * diff
    void DiffSigmoid( const MatrixBase& value, const MatrixBase& diff );

    // this = exp(value)
    void Exp( const MatrixBase& value );

    // this = exp(value)
    // this /= this.sum(0).reshape((-1, 1))
    void Softmax( const MatrixBase& value );

    // this = soure[index]
    void GetRows( const MatrixBase& source, const MatrixBase& index );

    // for i in xrange(len(index)):
    //     this[index[i]] += alpha * update[i] 
    void AddRows( const MatrixBase& update, const MatrixBase& index, float alpha );

    // for i in xrange(len(index)):
    //     this[:,i] = source[:,index[i]]
    void GetColumns( const MatrixBase& source, const MatrixBase& index );

    // for i in xrange(len(index)):
    //     this[:, index[i]]
    void AddColumns( const MatrixBase& source, const MatrixBase& index, float alpha );

    // for i in xrange(len(target)):
    //     this[i,target[i]] -= 1
    void DiffXent( const MatrixBase& value, const MatrixBase& target );

    void Random( float lower, float upper );

    // this = beta * beta + alpha + other
    // 
    // for i in xrange(Rows()):
    //     this[i,:] = beta * this[i,:] + alpha * row
    // effectively: this = beta * this + alpha * dot(ones((m_n_row, 1)), row)
    // 
    // for i in xrange(Columns()):
    //     this[i] = column
    // effectively: this = beta * this + alpha * dot(column, ones(1, m_n_column))
    void Add( float beta, float alpha, const MatrixBase& other, cublasOperation_t = CUBLAS_OP_N );
    void Add( float alpha, const MatrixBase& other );

    // this = beta * this + alpha * matrix.sum(0)
    void SumRowsOf( const MatrixBase& matrix, float beta = 0.0f, float alpha = 1.0f );

    // this = beta * this + alpha * matrix.sum(1)
    void SumColumnsOf( const MatrixBase& matrix, float beta = 0.0f, float alpha = 1.0f );

    // this = deepcopy(other);
    void Copy( const MatrixBase& other );

    // this = value.argmax(1)
    void ArgMax( const MatrixBase& value );

    // this = value[arange(value.Rows()), index];
    void LookUp( const MatrixBase& value, const MatrixBase& index );

    // this = log(value)
    void Log( const MatrixBase& value );

    // return -log(this[arange(m_n_row), index]).sum()
    float Xent( const MatrixBase& index ) const;

    // acclen is the accumulative length of the sentence length in a minibatch
    // alpha is forgetting factor
    // for i in xrange(len(acclen) - 1):
    //     startIdx, endIdx = acclen[i], acclen[i + 1]
    //     for j in xrange(startIdx, endIdx):
    //         for k in xrange(startIdx, endIdx):
    //             (*this)[j, k] = alpha ** (j - k) if j >= k else 0.0f
    void FOFE( const MatrixBase& acclen, float alpha );

    // for i in xrange(len(acclen) - 1):
    //     startIdx, endIdx = acclen[i], acclen[i + 1]
    //     for j in xrange(startIdx, endIdx):
    //         for k in xrange(startIdx, endIdx):
    //             (*this)[j, k] = filter[j - k] if j >= k and j - k < order else 0.0f
    void SfsmnBlockDiaongal( const MatrixBase& acclen, const MatrixBase& filter );

    // for i in xrange(len(acclen) - 1):
    //     startIdx, endIdx = acclen[i], acclen[i + 1]
    //     for j in xrange(startIdx, endIdx):
    //         for k in xrange(startIdx, endIdx):
    //             if j >= k and j - k < order:
    //                 (*this)[j - k] += gradient[j, k]
    void UpdateSfsmnFilter( const MatrixBase& acclen, const MatrixBase& gradient );

    // for i in xrange(shape[0]):
    //     for j in xrange(shape[1]):
    //         if j > i:
    //             (*this)[i, j] = -unigram[j] / (value[i, j] + unigram[j])
    //         else if i == j:
    //             (*this)[i, j] = 1 - unigram[j] / (value[i, j] + unigram[j])
    //         else:
    //             (*this)[i, j] = 0
    void DiffNCE( const MatrixBase& value, const MatrixBase& unigram );

    // please refers to LowerBound in STL
    void LowerBound( const MatrixBase& value, const MatrixBase& bound );

    void ClearNanOrInf();

    bool HasNanOrInf();

    void VfsmnMemory( const MatrixBase& hidden, 
                      const MatrixBase& filter, 
                      const MatrixBase& position );

    void ComputeVfsmnHiddenDiff( const MatrixBase& memoryDiff, 
                                 const MatrixBase& filter,
                                 const MatrixBase& position );

    void UpdateVfsmnFilter( const MatrixBase& memoryDiff,
                            const MatrixBase& hidden,
                            const MatrixBase& position,
                            float alpha );


    void Clip( float lower, float upper );

    float Min();

    float Max();

    float Sum();

    float L2Norm();
    
    void Sparse2Dense( const SparseMatrix& other );
};  // end of MatrixBase


// ==========================================================================================

ostream& operator << ( ostream& out, const MatrixBase& matrix );

// ==========================================================================================

class Matrix : public MatrixBase {
private:
    Matrix& operator = ( const Matrix& other ); // { return *this }

public:
    friend class SubMatrix;
    Matrix();
    Matrix( int n_row, int n_column, FunctionType type = kSetZero );
    ~Matrix();
    void Reshape( int row, int column, FunctionType type = kUndefined );
};  // end of Matrix


// ==========================================================================================

// Once the parent matrix is reshaped, the SubMatrix becomes invalid.
class SubMatrix : public MatrixBase {
public:
    SubMatrix() {}
    SubMatrix( const MatrixBase& matrix, MatrixRange range );
    SubMatrix( const SubMatrix& other );
    SubMatrix& operator = ( const SubMatrix& other );
    ~SubMatrix() {}
};  // end of SubMatrix


#endif
