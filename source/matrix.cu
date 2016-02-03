/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : matrix.cu
Last Update : Feb 2, 2016
Description : All low-level implementation in CUDA is hidden in this file
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// nvcc -Xcompiler -rdynamic -lcurand -lcublas -o ../main matrix.cu -DMATRIX_UNIT_TEST


#include "matrix.h"

struct TestNanOrInf {
    __host__ __device__ bool operator() ( const float x ) const {
        return isnan(x) || isinf(x);
    }
};


__global__ static void DiffXent( float* value, MatrixShape shape, 
                                 const float* target, int stride ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < shape.row ) {
        value[i * shape.stride + (int)target[i * stride]] -= 1.0f;
    }
}   // end of DiffXent


__global__ static void ClearNanOrInf( float* value, MatrixShape shape ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < shape.row && c < shape.column ) {
        float __value = value[r * shape.stride + c];
        if ( isnan(__value) || isinf(__value) ) {
            value[r * shape.stride + c] = 0.0f;
            printf( "(%d, %d) of (%d, %d, %d) \n", 
                      r, c, shape.row, shape.column, shape.stride );
        }
    }
}   // end of ClearNanOrInf


__global__ static void FOFE( float* result, MatrixShape shape, 
                             const float* length, MatrixShape lshape, 
                             float alpha ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < lshape.row - 1 ) {
        const int START_IDX = (float) length[i * lshape.stride];
        const int END_IDX = (float) length[(i + 1) * lshape.stride];
        for ( int j = START_IDX; j < END_IDX; j++ ) {
            for ( int k = START_IDX; k < END_IDX; k++ ) {
                if ( j >= k ) {
                    result[j * shape.stride + k] = powf( alpha, j - k );
                }
            }
        }
    }   // end of if ( i
}   // end of FOFE


__global__ static void SfsmnBlockDiaongal( float* result, MatrixShape shape,
                                           const float* length, MatrixShape lshape,
                                           const float* filter, int order ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < lshape.row - 1 ) {
        const int START_IDX = (float) length[i * lshape.stride];
        const int END_IDX = (float) length[(i + 1) * lshape.stride];
        for ( int j = START_IDX; j < END_IDX; j++ ) {
            result[j * shape.stride + j] = 1.0f;
            for ( int k = START_IDX; k < END_IDX; k++ ) {
                if ( j > k && (j - k < order) ) {
                    result[j * shape.stride + k] = filter[j - k];
                }
            }
        }
    }   // end of if ( i
}   // end of SfsmnBlockDiaongal 


__global__ static void UpdateSfsmnFilter( const float* gradient, MatrixShape shape,
                                          const float* length, MatrixShape lshape,
                                          float* filter, int order ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < lshape.row - 1 ) {
        const int START_IDX = (float) length[i * lshape.stride];
        const int END_IDX = (float) length[(i + 1) * lshape.stride];
        for ( int j = START_IDX; j < END_IDX; j++ ) {
            for ( int k = START_IDX; k < END_IDX; k++ ) {
                if ( j > k && (j - k < order) ) {
                    atomicAdd( filter + (j - k), gradient[j * shape.stride + k] );
                }
            }
        }
    }   // end of if ( i
}   // end of UpdateSfsmnFilter


__global__ static void VfsmnMemroy( const float* hidden, MatrixShape hShape,
                                    const float* filter, MatrixShape fShape,
                                    const float* position, int stride,
                                    float* memory, MatrixShape mShape ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < mShape.row && c < mShape.column ) {
        const int step = fminf(position[r * stride], fShape.row);
        float result = hidden[r * hShape.stride + c];
        for ( int i = 0; i < step; i++ ) {
            result += filter[i * fShape.stride + c] * 
                        hidden[(r - i - 1) * hShape.stride + c];
        }
        memory[r * mShape.stride + c] = result;
    }
}   // end of VfsmnMemroy


__global__ static void ComputeVfsmnHiddenDiff( const float* memoryDiff, MatrixShape mdShape,
                                               const float* position, int stride,
                                               const float* filter, MatrixShape fShape,
                                               float* hiddenDiff, MatrixShape hdShape ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < mdShape.row && c < mdShape.column ) {
        const int step = fminf(position[r * stride], fShape.row);
        for ( int i = 0; i < step; i++ ) {
            atomicAdd( hiddenDiff + (r - i - 1) * hdShape.stride + c,
                       filter[i * fShape.stride + c] * memoryDiff[r * mdShape.stride + c] );
        }
    }
}   // end of UpdateVfsmnFilter


__global__ static void UpdateVfsmnFilter( const float* memoryDiff, MatrixShape mdShape,
                                          const float* position, int stride,
                                          const float* hidden, MatrixShape hShape,
                                          float* filter, MatrixShape fShape,
                                          float alpha ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < mdShape.row && c < mdShape.column ) {
        const int step = fminf(position[r * stride], fShape.row);
        for ( int i = 0; i < step; i++ ) {
            atomicAdd( filter + i * fShape.stride + c,
                       hidden[(r - i - 1) * hShape.stride + c] * 
                            memoryDiff[r * mdShape.stride + c] * alpha );
        }
    }
}   // end of UpdateVfsmnFilter


__global__ static void ArgMax( const float* value, MatrixShape shape, 
                               float* index, int stride ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < shape.row ) {
        float largest = value[i * shape.stride];
        int maxIdx = 0;
        for ( int j = 1; j < shape.column; ++j ) {
            float candidate = value[i * shape.stride + j];
            if ( candidate > largest ) {
                largest = candidate;
                maxIdx = j;
            }
        }
        index[i * stride] = (float) maxIdx;
    }
}   // end of ArgMax


__global__ static void LookUp( const float* value, MatrixShape shape, 
                               const float* index, int stride, 
                               float* result ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < shape.row ) {
        result[i] = value[i * shape.stride + (int)index[i * stride]];
    }
}


template <FunctionType type>
__global__ static void Kernel( const float* inP, MatrixShape inShape, 
                               float* outP, MatrixShape outShape, 
                               float optional = 0.0f ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < outShape.row && c < outShape.column ) {
        int srcIdx = r * inShape.stride + c;
        int dstIdx = r * outShape.stride + c;
        switch ( type ) {
            case kRelu:
                outP[dstIdx] = fmaxf( 0.0f, inP[srcIdx] );
                break;
                
            case kExp: // clip it/ceiling
                outP[dstIdx] = expf( inP[srcIdx] );
                break;
                
            case kReciprocal:
            #ifdef DEBUG
                assert( inP[srcIdx] != 0.0f );
            #endif
                outP[dstIdx] = 1.0f / inP[srcIdx];
                break;
                
            case kSigmoid:
                outP[dstIdx] = 1.0f / ( 1.0f + expf( -inP[srcIdx] ) );
                break;

            case kFill:
                outP[dstIdx] = optional;
                break;

            case kScale:
                outP[dstIdx] = inP[srcIdx] * optional;
                break;

            case kShift:
                outP[dstIdx] = inP[srcIdx] + optional;
                break;

            case kLog:
                float __out = logf(inP[srcIdx]);
                outP[dstIdx] = (isnan(__out) || isinf(__out)) ? -77.0f : __out;
                break;
        }   // end of case
    }   // end of if
}   // end of Kernel


template <FunctionType type> 
__global__ static void Kernel( const float* src1, MatrixShape src1shape,    // value
                               const float* src2, MatrixShape src2shape,    // diff/prob/bound
                               float* dst, MatrixShape dstShape,
                               float optional = 0.0f ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < dstShape.row && c < dstShape.column ) {
        switch ( type ) {
            float __value, __prob, commonTerm, gradient;
            int miniBatch;

            case kDiffReLU:
                __value = src1[r * src1shape.stride + c];
                dst[r * dstShape.stride + c] = (__value > 0.0f ? src2[r * src2shape.stride + c] : 0.0f);
                break;

            case kDiffSigmoid:
                __value = src1[r * src1shape.stride + c];
                dst[r * dstShape.stride + c] = __value * (1.0f - __value) * src2[r * src2shape.stride + c];
                break;

            case kDiffNCE:
                miniBatch = src1shape.row;
                __value = src1[r * src1shape.stride + c];
                __prob = src2[c];
                commonTerm = __prob / (__prob + __value);
                if ( c < miniBatch ) {
                    if ( r == c ) {     // true sample
                        gradient = -commonTerm;
                        gradient = (isnan(gradient) || isinf(gradient)) ? -0.001 : gradient;
                    }
                    else {
                        gradient = 0.0f;
                    }
                }
                else {
                    gradient = 1.0f - commonTerm;
                    gradient = (isnan(gradient) || isinf(gradient)) ? 0.001 : gradient;
                }
                dst[r * dstShape.stride + c] = gradient;
                break;

            case kLowerBound:
                int count = src2shape.column;
                int result = 0;
                int idx, step;
                float target = src1[r * src1shape.stride + c];
                while ( count > 0 ) {
                    idx = result;
                    step = count / 2;
                    idx += step;
                    if ( src2[idx] < target ) {
                        result = ++idx;
                        count -= step + 1;
                    }
                    else {
                        count = step;
                    }
                }
                dst[r * dstShape.stride + c] = (float)result;
                break;

        }
    }   // end of if
}   // end of Kernel


template<FunctionType type> 
__global__ static void Kernel( const float* src, MatrixShape srcShape,
                               const float* index, int stride,
                               float* dst, MatrixShape dstShape ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < dstShape.row && c < dstShape.column ) {
        switch( type ) {
            case kGetRows:
                dst[r * dstShape.stride + c] = \
                    src[(int)index[r * stride] * srcShape.stride + c];
                break;
            case kGetColumns:
                dst[r * dstShape.stride + c] = \
                    src[r * srcShape.stride + (int)index[c * stride]];
                break;
        }
    }
}   // end of Kernel


template<FunctionType type>
__global__ static void Kernel( const float* src, MatrixShape srcShape,
                               const float* index, int stride,
                               float* dst, MatrixShape dstShape,
                               float alpha ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < srcShape.row && c < srcShape.column ) {
        int dstOffset;
        switch ( type ) {
            case kAddRows:
                dstOffset = (int)index[r * stride] * dstShape.stride + c;
                atomicAdd( dst + dstOffset, alpha * src[r * srcShape.stride + c] ); 
                break;
            case kAddColumns:
                dstOffset = r * dstShape.stride + (int)index[c * stride];
                atomicAdd( dst + dstOffset, alpha * src[r * srcShape.stride + c] );
                break;
        }
    }
}



template __global__ void Kernel< kSigmoid >  ( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kRelu >     ( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kExp>       ( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kReciprocal>( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kFill >     ( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kScale >    ( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kShift >    ( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kLog >      ( const float* inP, MatrixShape inShape, 
                                               float* outP, MatrixShape outShape, 
                                               float optional );

template __global__ void Kernel< kDiffReLU    > ( const float* value, MatrixShape valueShape, 
                                                  const float* diff, MatrixShape diffShape,
                                                  float* result, MatrixShape resultShape,
                                                  float optional );

template __global__ void Kernel< kDiffSigmoid > ( const float* value, MatrixShape valueShape, 
                                                  const float* diff, MatrixShape diffShape,
                                                  float* result, MatrixShape resultShape,
                                                  float optional );

template __global__ void Kernel< kDiffNCE > ( const float* value, MatrixShape valueShape, 
                                                  const float* unigram, MatrixShape diffShape,
                                                  float* result, MatrixShape resultShape,
                                                  float optional );

template __global__ void Kernel< kLowerBound > ( const float* value, MatrixShape valueShape, 
                                                 const float* bound, MatrixShape diffShape,
                                                 float* result, MatrixShape resultShape,
                                                 float optional );

template __global__ void Kernel< kGetRows >   ( const float* src, MatrixShape srcShape,
                                                const float* index, int stride,
                                                float* dst, MatrixShape dstShape );

template __global__ void Kernel< kGetColumns >( const float* src, MatrixShape srcShape,
                                                const float* index, int stride,
                                                float* dst, MatrixShape dstShape );

template __global__ void Kernel< kAddRows >   ( const float* src, MatrixShape srcShape,
                                                const float* index, int stride,
                                                float* dst, MatrixShape dstShape,
                                                float alpha );

template __global__ void Kernel< kAddColumns> ( const float* src, MatrixShape srcShape,
                                                const float* index, int stride,
                                                float* dst, MatrixShape dstShape,
                                                float alpha );


// ==========================================================================================


float* MatrixBase::ones = NULL;
float* MatrixBase::zeros = NULL;
float* MatrixBase::buffer = NULL;
int MatrixBase::buffer_size = 0;
int MatrixBase::n_matrix = 0;
cublasHandle_t MatrixBase::handle = NULL;
curandGenerator_t MatrixBase::generator = NULL;


bool MatrixBase::ValidateRange( MatrixRange range ) const  {
    return range.startRow <= range.endRow 
        && range.startColumn <= range.endColumn
        && range.startRow >= 0 
        && range.endRow <= m_n_row
        && range.startColumn >= 0
        && range.endColumn <= m_n_column;
}   // end of ValidateRange



void MatrixBase::GetData( vector<float>* data ) const {
    data->resize( m_n_row * m_n_column );
    CUDA_CHECK( cudaMemcpy2D( &((*data)[0]),
                              m_n_column * sizeof(float),
                              m_data,
                              m_n_stride * sizeof(float),
                              m_n_column * sizeof(float),
                              m_n_row,
                              cudaMemcpyDeviceToHost ),
                "cudaMemcpy2D" );
}   // end of GetData



void MatrixBase::SetData( const vector<float>& data ) {
    ASSERT( data.size() == m_n_row * m_n_column );
    CUDA_CHECK( cudaMemcpy2D( m_data,
                              m_n_stride * sizeof(float),
                              &data[0],
                              m_n_column * sizeof(float),
                              m_n_column * sizeof(float),
                              m_n_row,
                              cudaMemcpyHostToDevice ),
                "cudaMemcpy2D" );
}   // end of SetData



string MatrixBase::ToString() const {
    vector<float> buffer;
    GetData( &buffer );
    stringstream ss;

    for ( int r = 0; r < m_n_row; r++ ) {
        for ( int c = 0; c < m_n_column; c++ ) {
            ss << setw(8) << setprecision(4) 
               << buffer[r * m_n_column + c] << ' ';
        }
        ss << endl;
    }
    return ss.str();
}   // end of ToString



MatrixShape MatrixBase::Shape() const {
    MatrixShape m;
    m.row = m_n_row;
    m.column = m_n_column;
    m.stride = m_n_stride;
    return m;
}   // end of Shape



void MatrixBase::Sgemm( float beta, float alpha,
            const MatrixBase& A, cublasOperation_t transa,
            const MatrixBase& B, cublasOperation_t transb ) {
    CUBLAS_CHECK( cublasSgemm( MatrixBase::handle,
                               transb,
                               transa,
                               m_n_column,
                               m_n_row,
                               (transb == CUBLAS_OP_N ? B.m_n_row : B.m_n_column),
                               &alpha,
                               B.m_data,
                               B.m_n_stride,
                               A.m_data,
                               A.m_n_stride,
                               &beta,
                               m_data,
                               m_n_stride ), 
                  "cublasSgemm" );
}   // end of Sgemm



void MatrixBase::Fill( float value ) {
    MatrixShape shape = Shape();
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    Kernel<kFill><<<grid, block>>>( m_data, shape, m_data, shape, value );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of fill



void MatrixBase::Scale( float value ) {
    MatrixShape shape = Shape();
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    Kernel<kScale><<<grid, block>>>( m_data, shape, m_data, shape, value );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of Scale



void MatrixBase::Shift( float value ) {
    MatrixShape shape = Shape();
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    Kernel<kShift><<<grid, block>>>( m_data, shape, m_data, shape, value );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}



void MatrixBase::ReLU( const MatrixBase& value ) {
    MatrixShape dstShape = Shape();
    MatrixShape srcShape = value.Shape();
    ASSERT( dstShape.row == srcShape.row && dstShape.column == srcShape.column );

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );

    Kernel<kRelu><<<grid, block>>>( value.m_data, srcShape, m_data, dstShape );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of ReLU



void MatrixBase::DiffReLU( const MatrixBase& value, const MatrixBase& diff ) {
    MatrixShape valueShape = value.Shape();
    MatrixShape diffShape = diff.Shape();
    MatrixShape resultShape = Shape();
    ASSERT( resultShape.row == valueShape.row && resultShape.column == valueShape.column );
    ASSERT( resultShape.row == diffShape.row && resultShape.column == diffShape.column ); 

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );

    Kernel<kDiffReLU><<<grid, block>>>( value.m_data, valueShape, 
                                        diff.m_data, diffShape, 
                                        m_data, resultShape );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of DiffReLU



void MatrixBase::Sigmoid( const MatrixBase& value ) {
    MatrixShape dstShape = Shape();
    MatrixShape srcShape = value.Shape();
    ASSERT( dstShape.row == srcShape.row && dstShape.column == srcShape.column );

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );

    Kernel<kSigmoid><<<grid, block>>>( value.m_data, srcShape, m_data, dstShape );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of Sigmoid



void MatrixBase::DiffSigmoid( const MatrixBase& value, const MatrixBase& diff ) {
    MatrixShape valueShape = value.Shape();
    MatrixShape diffShape = diff.Shape();
    MatrixShape resultShape = Shape();
    ASSERT( resultShape.row == valueShape.row && resultShape.column == valueShape.column );
    ASSERT( resultShape.row == diffShape.row && resultShape.column == diffShape.column ); 

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );

    Kernel<kDiffSigmoid><<<grid, block>>>( value.m_data, valueShape, 
                                           diff.m_data, diffShape, 
                                           m_data, resultShape );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of DiffSigmoid


void MatrixBase::DiffNCE( const MatrixBase& value, const MatrixBase& unigram ) {
    ASSERT( unigram.m_n_row == 1 && unigram.m_n_column == m_n_column );
    ASSERT( m_n_row == value.m_n_row && m_n_column == value.m_n_column );
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    Kernel<kDiffNCE><<<grid, block>>>( value.m_data, value.Shape(), 
                                       unigram.m_data, unigram.Shape(), 
                                       m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of DiffNCE


void MatrixBase::LowerBound( const MatrixBase& value, const MatrixBase& bound ) {
    ASSERT( bound.m_n_row == 1 && bound.m_n_column > 0 );
    ASSERT( m_n_row == value.m_n_row && m_n_column == value.m_n_column );
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    Kernel<kLowerBound><<<grid, block>>>( value.m_data, value.Shape(),
                                          bound.m_data, bound.Shape(),
                                          m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of LowerBound



void MatrixBase::Exp( const MatrixBase& value ) {
    MatrixShape dstShape = Shape();
    MatrixShape srcShape = value.Shape();
    ASSERT( dstShape.row == srcShape.row && dstShape.column == srcShape.column );

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );

    Kernel<kExp><<<grid, block>>>( value.m_data, srcShape, m_data, dstShape );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of Exp



void MatrixBase::Softmax( const MatrixBase& value ) {
    const float ONE = 1.0f;
    const float ZERO = 0.0f;

    Exp( value );

    CUBLAS_CHECK( cublasSgemv( MatrixBase::handle,
                               CUBLAS_OP_T,
                               m_n_column,
                               m_n_row,
                               &ONE,
                               m_data,
                               m_n_stride,
                               MatrixBase::ones,
                               1,
                               &ZERO,
                               MatrixBase::buffer,
                               1 ),
                  "cublasSgemv" );

    MatrixShape shape = { m_n_row, 1, 1 };
    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim(m_n_row, CU1DBLOCK), 1 );
    Kernel<kReciprocal><<<grid, block>>>( MatrixBase::buffer, shape,
                                          MatrixBase::buffer, shape );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    CUBLAS_CHECK( cublasSdgmm( MatrixBase::handle,
                               CUBLAS_SIDE_RIGHT,
                               m_n_column,
                               m_n_row,
                               m_data,
                               m_n_stride,
                               MatrixBase::buffer,
                               1,
                               m_data,
                               m_n_stride ),
                  "cublasSdgmm" );
}   // end of Softmax



void MatrixBase::GetRows( const MatrixBase& source, const MatrixBase& index ) {
    ASSERT( m_n_row == index.m_n_row );
    ASSERT( m_n_column == source.m_n_column );
    ASSERT( index.m_n_column == 1 );

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );

    Kernel<kGetRows><<<grid, block>>>( source.m_data, source.Shape(),
                                       index.m_data, index.m_n_stride,
                                       m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}



void MatrixBase::AddRows( const MatrixBase& update, const MatrixBase& index, float alpha ) {
    ASSERT( update.m_n_row == index.m_n_row );
    ASSERT( m_n_column == update.m_n_column );
    ASSERT( index.m_n_column == 1 );

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(update.m_n_row, CU2DBLOCK), GridDim(update.m_n_column, CU2DBLOCK) );

    Kernel<kAddRows><<<grid, block>>>( update.m_data, update.Shape(),
                                       index.m_data, index.m_n_stride,
                                       m_data, Shape(),
                                       alpha );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}



void MatrixBase::GetColumns( const MatrixBase& source, const MatrixBase& index ) {
    ASSERT( m_n_column == index.m_n_row );
    ASSERT( m_n_row == source.m_n_row );
    ASSERT( index.m_n_column == 1 );

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );

    Kernel<kGetColumns><<<grid, block>>>( source.m_data, source.Shape(),
                                          index.m_data, index.m_n_stride,
                                          m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of GetColumns



void MatrixBase::AddColumns( const MatrixBase& update, 
                             const MatrixBase& index, 
                             float alpha ) {
    ASSERT( update.m_n_column == index.m_n_row );
    ASSERT( m_n_row == update.m_n_row );
    ASSERT( index.m_n_column == 1 );

    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(update.m_n_row, CU2DBLOCK), GridDim(update.m_n_column, CU2DBLOCK) );

    Kernel<kAddColumns><<<grid, block>>>( update.m_data, update.Shape(),
                                          index.m_data, index.m_n_stride,
                                          m_data, Shape(),
                                          alpha );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of AddColumns



void MatrixBase::DiffXent( const MatrixBase& value, const MatrixBase& target ) {
    ASSERT( m_n_row == value.m_n_row );
    ASSERT( m_n_row == target.m_n_row );
    ASSERT( m_n_column == value.m_n_column );
    ASSERT( target.m_n_column == 1 );

    if ( this != &value ) {
        CUDA_CHECK( cudaMemcpy2D( m_data,
                                  m_n_stride * sizeof(float),
                                  value.m_data,
                                  value.m_n_stride * sizeof(float),
                                  m_n_column * sizeof(float),
                                  m_n_row,
                                  cudaMemcpyDeviceToDevice ),
                    "cudaMemcpy2D" );
    }

    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim( m_n_row, CU1DBLOCK ), 1 );

    ::DiffXent<<<grid, block>>>( m_data, Shape(), target.m_data, target.m_n_stride );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}






void MatrixBase::Random( float lower, float upper ) {
    int actualSize = m_n_row * m_n_column;
    float* deviceP = NULL;

    if ( MatrixBase::buffer_size < actualSize ) {
        CUDA_CHECK( cudaMalloc( (void**)&deviceP, sizeof(float) * actualSize ),
                    "cudaMalloc" );
    }
    else {
        deviceP = MatrixBase::buffer;
    }

    CURAND_CHECK( curandGenerateUniform( MatrixBase::generator, deviceP, actualSize ),
                  "curandGenerateUniform" );

    CUDA_CHECK( cudaMemcpy2D( m_data,
                              m_n_stride * sizeof(float),
                              deviceP,
                              m_n_column * sizeof(float),
                              m_n_column * sizeof(float),
                              m_n_row,
                              cudaMemcpyDeviceToDevice ),
                "cudaMemcpy2D" );

    if ( MatrixBase::buffer_size < actualSize ) {
        CUDA_CHECK( cudaFree( deviceP ), "cudaFree" );
    }

    float range = upper - lower;
    if ( range != 1.0f ) {
        Scale( range );
    }

    if ( lower != 0.0f ) {
        Shift( lower );
    }
}



void MatrixBase::SumColumnsOf( const MatrixBase& matrix, float beta, float alpha ) {
    ASSERT( m_n_column == 1 );
    ASSERT( m_n_row == matrix.m_n_row );
    CUBLAS_CHECK( cublasSgemv( MatrixBase::handle,
                               CUBLAS_OP_T,
                               matrix.m_n_column,
                               matrix.m_n_row,
                               &alpha,
                               matrix.m_data,
                               matrix.m_n_stride,
                               MatrixBase::ones,
                               1,
                               &beta,
                               m_data,
                               m_n_stride ),
                  "cublasSgemv" );
}


void MatrixBase::SumRowsOf( const MatrixBase& matrix, float beta, float alpha ) {
    ASSERT( m_n_row == 1 );
    ASSERT( m_n_column == matrix.m_n_column );
    CUBLAS_CHECK( cublasSgemv( MatrixBase::handle,
                               CUBLAS_OP_N,
                               matrix.m_n_column,
                               matrix.m_n_row,
                               &alpha,
                               matrix.m_data,
                               matrix.m_n_stride,
                               MatrixBase::ones,
                               1,
                               &beta,
                               m_data,
                               1 ),
                  "cublasSgemv" );
}


void MatrixBase::Add( float beta, float alpha, const MatrixBase& other, cublasOperation_t trans ) {
    if ( trans == CUBLAS_OP_N && m_n_row == other.m_n_row && m_n_column == other.m_n_column ||
         trans == CUBLAS_OP_T && m_n_row == other.m_n_column && m_n_column == other.m_n_row ) {
        // this = beta * this + alpha * other
        CUBLAS_CHECK( cublasSgeam( MatrixBase::handle,
                                   trans,
                                   CUBLAS_OP_N,
                                   m_n_column,
                                   m_n_row,
                                   &alpha,
                                   other.m_data,
                                   other.m_n_stride,
                                   &beta,
                                   m_data,
                                   m_n_stride,
                                   m_data,
                                   m_n_stride ), 
                      "cublasSgeam" );
    }
    else if ( trans == CUBLAS_OP_N && other.m_n_row == 1 && other.m_n_column == m_n_column ||
              trans == CUBLAS_OP_T && other.m_n_column == 1 && other.m_n_row == m_n_column ) {
        // this = beta * this + alpha * dot(ones(m_n_row, 1), row)
        CUBLAS_CHECK( cublasSgemm( MatrixBase::handle,
                                   trans,
                                   CUBLAS_OP_N,
                                   m_n_column,
                                   m_n_row,
                                   1,
                                   &alpha,
                                   other.m_data,
                                   other.m_n_stride,
                                   MatrixBase::ones,
                                   1,
                                   &beta,
                                   m_data,
                                   m_n_stride ), 
                      "cublasSgemm" );
    }
    else if ( trans == CUBLAS_OP_N && other.m_n_column == 1 && other.m_n_row == m_n_row ||
              trans == CUBLAS_OP_T && other.m_n_row == 1 && other.m_n_column == m_n_row ) {
        // this = beta * this + alpha * dot(column, ones(1, m_n_column))
        CUBLAS_CHECK( cublasSgemm( MatrixBase::handle,
                                   CUBLAS_OP_N,
                                   trans,
                                   m_n_column,
                                   m_n_row,
                                   1,
                                   &alpha,
                                   MatrixBase::ones,
                                   m_n_column,
                                   other.m_data,
                                   other.m_n_stride,
                                   &beta,
                                   m_data,
                                   m_n_stride ), 
                      "cublasSgemm" );
    }
    else {
        ASSERT( false );
    }
}   // end of Add


void MatrixBase::Copy( const MatrixBase& other ) {
    if ( this != &other ) {
        ASSERT( m_n_row == other.m_n_row && m_n_column == other.m_n_column );
        CUDA_CHECK( cudaMemcpy2D( m_data,
                                  m_n_stride * sizeof(float),
                                  other.m_data,
                                  other.m_n_stride * sizeof(float),
                                  m_n_column * sizeof(float),
                                  m_n_row,
                                  cudaMemcpyDeviceToDevice ),
                    "cudaMemcpy2D" );
    }
}


void MatrixBase::ArgMax( const MatrixBase& value ) {
    ASSERT( m_n_row == 1 || m_n_column == 1 );
    if ( m_n_row == 1 ) {
        ASSERT( m_n_column == value.m_n_row );
    }
    else {
        ASSERT( m_n_row == value.m_n_row );
    }

    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim(value.m_n_row, CU1DBLOCK), 1 );
    ::ArgMax<<<grid, block>>>( value.m_data, 
                               value.Shape(), 
                               m_data, 
                               m_n_row == 1 ? 1 : m_n_stride );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}


void MatrixBase::LookUp( const MatrixBase& value, const MatrixBase& index ) {
    ASSERT( index.m_n_row == 1 || index.m_n_column == 1 );
    if ( index.m_n_row == 1 ) {
        ASSERT( value.m_n_row == index.m_n_column );
        ASSERT( m_n_column == index.m_n_column );
    }
    else {
        ASSERT( value.m_n_row == index.m_n_row );
        ASSERT( m_n_column == index.m_n_row );
    }

    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim(m_n_row, CU1DBLOCK), 1 );
    ::LookUp<<<grid, block>>>( value.m_data,
                               value.Shape(),
                               index.m_data,
                               index.m_n_row == 1 ? 1 : index.m_n_stride,
                               m_data );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}


void MatrixBase::Log( const MatrixBase& value ) {
    ASSERT( m_n_row == value.m_n_row && m_n_column == value.m_n_column );
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    Kernel<kLog><<<grid, block>>>( value.m_data, value.Shape(), m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}



float MatrixBase::Xent( const MatrixBase& index ) const {
    ASSERT( index.m_n_row == 1 || index.m_n_column == 1 );
    if ( index.m_n_row == 1 ) {
        ASSERT( m_n_row == index.m_n_column );
    }
    else {
        ASSERT( m_n_row == index.m_n_row );
    }

    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim(m_n_row, CU1DBLOCK), 1 );
    ::LookUp<<<grid, block>>>( m_data,
                               Shape(),
                               index.m_data,
                               index.m_n_row == 1 ? 1 : index.m_n_stride,
                               MatrixBase::buffer );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    MatrixShape shape = { 1, m_n_row, m_n_row };
    dim3 _block( CU2DBLOCK, CU2DBLOCK );
    dim3 _grid( GridDim(1, CU2DBLOCK), GridDim(m_n_row, CU2DBLOCK) );
    Kernel<kLog><<<_grid, _block>>>( MatrixBase::buffer, shape, MatrixBase::buffer, shape );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    thrust::device_ptr<float> buffer = thrust::device_pointer_cast( MatrixBase::buffer ); 
    float result = thrust::reduce( buffer, buffer + m_n_row );

    return -result;
}


void MatrixBase::FOFE( const MatrixBase& acclen, float alpha ) {
    ASSERT( m_n_row == m_n_column );
    ASSERT( acclen.m_n_column == 1 );

    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim(acclen.m_n_row - 1, CU1DBLOCK), 1 );



    ::FOFE<<<grid, block>>>( m_data, Shape(), acclen.m_data, acclen.Shape(), alpha );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
} 


void MatrixBase::SfsmnBlockDiaongal( const MatrixBase& acclen, const MatrixBase& filter ) {
    ASSERT( m_n_row == m_n_column );
    ASSERT( acclen.m_n_column == 1 );
    ASSERT( filter.m_n_row == 1 );

    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim(acclen.m_n_row - 1, CU1DBLOCK), 1 );

    ::SfsmnBlockDiaongal<<<grid, block>>>( m_data, 
                                           Shape(), 
                                           acclen.m_data, 
                                           acclen.Shape(), 
                                           filter.m_data, 
                                           filter.m_n_column );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}


void MatrixBase::UpdateSfsmnFilter( const MatrixBase& acclen, const MatrixBase& gradient ) {
    ASSERT( gradient.m_n_row == gradient.m_n_column );
    ASSERT( acclen.m_n_column == 1 );
    ASSERT( m_n_row == 1 );

    dim3 block( CU1DBLOCK, 1 );
    dim3 grid( GridDim(acclen.m_n_row - 1, CU1DBLOCK), 1 );

    ::UpdateSfsmnFilter<<<grid, block>>>( gradient.m_data, 
                                          gradient.Shape(), 
                                          acclen.m_data, 
                                          acclen.Shape(), 
                                          m_data, 
                                          m_n_column );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}


void MatrixBase::ClearNanOrInf() {
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    ::ClearNanOrInf<<<grid, block>>>( m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}


// these few functions are ugly written because they won't be called frequently

bool MatrixBase::HasNanOrInf() {
    float* __buffer = NULL;
    int __size = m_n_row * m_n_column;
    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaMalloc( (void**)&__buffer, sizeof(float) * __size ), "cudaMalloc" );
    }
    else {
        __buffer = MatrixBase::buffer;
    }

    CUDA_CHECK( cudaMemcpy2D( __buffer,
                              m_n_column * sizeof(float),
                              m_data,
                              m_n_stride * sizeof(float),
                              m_n_column * sizeof(float),
                              m_n_row,
                              cudaMemcpyDeviceToDevice ),
                "cudaMemcpy2D" );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( __buffer );

    bool result = thrust::transform_reduce( deviceP, 
                                            deviceP + __size, 
                                            TestNanOrInf(), 
                                            0,
                                            thrust::plus<bool>() );

    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaFree( __buffer ), "cudaFree" );
    }

    return result;
}


float MatrixBase::Min() {
    float* __buffer = NULL;
    int __size = m_n_row * m_n_column;
    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaMalloc( (void**)&__buffer, sizeof(float) * __size ), "cudaMalloc" );
    }
    else {
        __buffer = MatrixBase::buffer;
    }

    CUDA_CHECK( cudaMemcpy2D( __buffer,
                              m_n_column * sizeof(float),
                              m_data,
                              m_n_stride * sizeof(float),
                              m_n_column * sizeof(float),
                              m_n_row,
                              cudaMemcpyDeviceToDevice ),
                "cudaMemcpy2D" );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( __buffer );

    float result = thrust::reduce( deviceP, 
                                  deviceP + __size, 
                                  numeric_limits<float>::max(),
                                  thrust::minimum<float>() );

    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaFree( __buffer ), "cudaFree" );
    }

    return result;
}


float MatrixBase::Max() {
    float* __buffer = NULL;
    int __size = m_n_row * m_n_column;
    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaMalloc( (void**)&__buffer, sizeof(float) * __size ), "cudaMalloc" );
    }
    else {
        __buffer = MatrixBase::buffer;
    }

    CUDA_CHECK( cudaMemcpy2D( __buffer,
                              m_n_column * sizeof(float),
                              m_data,
                              m_n_stride * sizeof(float),
                              m_n_column * sizeof(float),
                              m_n_row,
                              cudaMemcpyDeviceToDevice ),
                "cudaMemcpy2D" );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( __buffer );

    float result = thrust::reduce( deviceP, 
                                  deviceP + __size, 
                                  numeric_limits<float>::min(),
                                  thrust::maximum<float>() );

    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaFree( __buffer ), "cudaFree" );
    }

    return result;
}


float MatrixBase::Sum() {
    float* __buffer = NULL;
    int __size = m_n_row * m_n_column;
    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaMalloc( (void**)&__buffer, sizeof(float) * __size ), "cudaMalloc" );
    }
    else {
        __buffer = MatrixBase::buffer;
    }

    CUDA_CHECK( cudaMemcpy2D( __buffer,
                              m_n_column * sizeof(float),
                              m_data,
                              m_n_stride * sizeof(float),
                              m_n_column * sizeof(float),
                              m_n_row,
                              cudaMemcpyDeviceToDevice ),
                "cudaMemcpy2D" );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( __buffer );

    float result = thrust::reduce( deviceP, 
                                  deviceP + __size, 
                                  0.0f,
                                  thrust::plus<float>() );

    if ( MatrixBase::buffer_size < __size ) {
        CUDA_CHECK( cudaFree( __buffer ), "cudaFree" );
    }

    return result;
}


void MatrixBase::VfsmnMemory( const MatrixBase& hidden, 
                              const MatrixBase& filter, 
                              const MatrixBase& position ) {
    ASSERT( m_n_row == hidden.m_n_row && m_n_column == hidden.m_n_column );
    ASSERT( position.m_n_column == 1 && position.m_n_row == m_n_row );
    ASSERT( m_n_column == filter.m_n_column );
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    ::VfsmnMemroy<<<grid, block>>>( hidden.m_data, hidden.Shape(),
                                    filter.m_data, filter.Shape(),
                                    position.m_data, position.m_n_stride,
                                    m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}



void MatrixBase::ComputeVfsmnHiddenDiff( const MatrixBase& memoryDiff, 
                                         const MatrixBase& filter,
                                         const MatrixBase& position ) {
    ASSERT( m_n_row == memoryDiff.m_n_row && m_n_column == memoryDiff.m_n_column );
    ASSERT( position.m_n_column == 1 && position.m_n_row == m_n_row );
    ASSERT( m_n_column == filter.m_n_column );
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    ::ComputeVfsmnHiddenDiff<<<grid, block>>>( memoryDiff.m_data, memoryDiff.Shape(),
                                               position.m_data, position.m_n_stride,
                                               filter.m_data, filter.Shape(),
                                               m_data, Shape() );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}


void MatrixBase::UpdateVfsmnFilter( const MatrixBase& memoryDiff,
                                    const MatrixBase& hidden,
                                    const MatrixBase& position,
                                    float alpha ) {
    ASSERT( hidden.m_n_row == memoryDiff.m_n_row && 
            hidden.m_n_column == memoryDiff.m_n_column );
    ASSERT( position.m_n_column == 1 && position.m_n_row == hidden.m_n_row );
    ASSERT( m_n_column == hidden.m_n_column );
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(hidden.m_n_row, CU2DBLOCK), 
               GridDim(hidden.m_n_column, CU2DBLOCK) );
    ::UpdateVfsmnFilter<<<grid, block>>>( memoryDiff.m_data, memoryDiff.Shape(),
                                          position.m_data, position.m_n_stride,
                                          hidden.m_data, hidden.Shape(),
                                          m_data, Shape(),
                                          alpha );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}

// ==========================================================================================

Matrix::Matrix() {
    if ( 0 == MatrixBase::n_matrix ) {
        long maxFreeByte = 0;
        int bestChoice = 0;

        int nDevice = 0;
        CUDA_CHECK( cudaGetDeviceCount( &nDevice ), "cudaGetDeviceCount" );
        ASSERT( nDevice > 0 );
        cout << CurrentTime() << ") There are " << nDevice << " GPU(s)" << endl; 

        for ( int i = 0; i < nDevice; i++ ) {
            size_t freeByte, total;
            CUDA_CHECK( cudaSetDevice(i), "cudaSetDevice" );
            CUDA_CHECK( cudaMemGetInfo( &freeByte, &total ), "cudaMemGetInfo" );
            if ( freeByte > maxFreeByte ) {
                bestChoice = i;
                maxFreeByte = freeByte;
            }
            CUDA_CHECK( cudaDeviceReset(), "cudaDeviceReset" );
        }

        CUDA_CHECK( cudaSetDevice( bestChoice ), "cudaSetDevice" );
        cout << CurrentTime() << ") device " << bestChoice << " is picked" << endl; 

        CUBLAS_CHECK( cublasCreate( &MatrixBase::handle ), "cublasCreate" );
        CURAND_CHECK( curandCreateGenerator( &MatrixBase::generator, 
                                             CURAND_RNG_PSEUDO_DEFAULT ), 
                      "curandCreateGenerator" );
        CURAND_CHECK( curandSetPseudoRandomGeneratorSeed( MatrixBase::generator, time(NULL) ), 
                      "curandSetPseudoRandomGeneratorSeed" );
    }
    MatrixBase::n_matrix++;
    ASSERT( NULL == m_data );
}   // end of Matrix


Matrix::Matrix( int n_row, int n_column, FunctionType type ) {
    if ( 0 == MatrixBase::n_matrix ) {
        long maxFreeByte = 0;
        int bestChoice = 0;

        int nDevice = 0;
        CUDA_CHECK( cudaGetDeviceCount( &nDevice ), "cudaGetDeviceCount" );
        ASSERT( nDevice > 0 );
        cout << CurrentTime() << ") There are " << nDevice << " GPU(s)" << endl; 

        for ( int i = 0; i < nDevice; i++ ) {
            size_t freeByte, total;
            CUDA_CHECK( cudaSetDevice(i), "cudaSetDevice" );
            CUDA_CHECK( cudaMemGetInfo( &freeByte, &total ), "cudaMemGetInfo" );
            if ( freeByte > maxFreeByte ) {
                bestChoice = i;
                maxFreeByte = freeByte;
            }
            CUDA_CHECK( cudaDeviceReset(), "cudaDeviceReset" );
        }

        CUDA_CHECK( cudaSetDevice( bestChoice ), "cudaSetDevice" );
        cout << CurrentTime() << ") device " << bestChoice << " is picked" << endl; 

        CUBLAS_CHECK( cublasCreate( &MatrixBase::handle ), "cublasCreate" );
        CURAND_CHECK( curandCreateGenerator( &MatrixBase::generator, 
                                             CURAND_RNG_PSEUDO_DEFAULT ), 
                      "curandCreateGenerator" );
        CURAND_CHECK( curandSetPseudoRandomGeneratorSeed( MatrixBase::generator, time(NULL) ), 
                      "curandSetPseudoRandomGeneratorSeed" );
    }
    Reshape( n_row, n_column, type );
    MatrixBase::n_matrix++;
}   // end of Matrix


Matrix::~Matrix() {
    if ( NULL != m_data ) {
        CUDA_CHECK( cudaFree( m_data ), "cudaFree" );
        m_data = NULL;
    }

    MatrixBase::n_matrix--;
    if ( 0 == MatrixBase::n_matrix ) {
        CUBLAS_CHECK( cublasDestroy( MatrixBase::handle ), "cublasDestroy" );
        MatrixBase::handle = NULL;

        CURAND_CHECK( curandDestroyGenerator( MatrixBase::generator ), 
                      "curandDestroyGenerator()" );
        MatrixBase::generator = NULL;

        CUDA_CHECK( cudaFree( MatrixBase::ones ), "cudaFree" );
        MatrixBase::ones = NULL;

        CUDA_CHECK( cudaFree( MatrixBase::zeros ), "cudaFree" );
        MatrixBase::zeros = NULL;

        CUDA_CHECK( cudaFree( MatrixBase::buffer ), "cudaFree" );
        MatrixBase::buffer = NULL;
        MatrixBase::buffer_size = 0;
    }
}   // end of ~Matrix


void Matrix::Reshape( int row, int column, FunctionType type ) {
    int newSize = row * column;

    if ( NULL != m_data && newSize > m_n_allocated ) {
        CUDA_CHECK( cudaFree( m_data ), "cudaFree" );
        m_data = NULL;
        m_n_allocated = 0;
    }

    m_n_row = row;
    m_n_column = column;
    m_n_stride = m_n_column;

    if ( NULL != m_data && kSetZero == type ) {
         thrust::device_ptr<float> thrustP = thrust::device_pointer_cast( m_data );
         thrust::fill( thrustP, thrustP + newSize, 0.0f );
    }

    if ( NULL == m_data ) {/*
        CUDA_CHECK( cudaMallocPitch( (void**)&m_data, 
                                     &m_n_stride, 
                                     m_n_column * sizeof(float), 
                                     m_n_row ), 
                    "cudaMallocPitch" );
        m_n_stride /= sizeof(float);*/
        m_n_allocated = newSize;
        CUDA_CHECK( cudaMalloc( (void**)&m_data, m_n_allocated * sizeof(float) ),
                    "cudaMalloc" );
        thrust::device_ptr<float> thrustP = thrust::device_pointer_cast( m_data );
        thrust::fill( thrustP, thrustP + newSize, 0.0f );
    }



    if ( MatrixBase::buffer_size < m_n_row || MatrixBase::buffer_size < m_n_column) {
        MatrixBase::buffer_size = max( m_n_row, m_n_column ) ;
        if ( NULL != MatrixBase::ones ) {
            CUDA_CHECK( cudaFree( MatrixBase::ones ), "cudaFree" );
        }
        if ( NULL != MatrixBase::zeros ) {
            CUDA_CHECK( cudaFree( MatrixBase::zeros ), "cudaFree" );
        }
        if ( NULL != MatrixBase::buffer ) {
            CUDA_CHECK( cudaFree( MatrixBase::buffer ), "cudaFree" );
        }

        CUDA_CHECK( cudaMalloc( (void**)&MatrixBase::ones, 
                                sizeof(float) * MatrixBase::buffer_size ), 
                    "cudaMalloc" );
        thrust::device_ptr<float> __ones = thrust::device_pointer_cast( MatrixBase::ones );
        thrust::fill( __ones, __ones + MatrixBase::buffer_size, 1.0f );

        CUDA_CHECK( cudaMalloc( (void**)&MatrixBase::zeros, 
                                sizeof(float) * MatrixBase::buffer_size ), 
                    "cudaMalloc" );
        thrust::device_ptr<float> __zeros = thrust::device_pointer_cast( MatrixBase::zeros );
        thrust::fill( __zeros, __zeros + MatrixBase::buffer_size, 0.0f );

        CUDA_CHECK( cudaMalloc( (void**)&MatrixBase::buffer, 
                                sizeof(float) * MatrixBase::buffer_size ), 
                    "cudaMalloc" );
    }
}   // end of Resahpe

// ==========================================================================================


SubMatrix::SubMatrix( const MatrixBase& matrix, MatrixRange range ) {
    ASSERT( matrix.ValidateRange( range ) );
    m_data = matrix.m_data + range.startRow * matrix.m_n_stride + range.startColumn;
    m_n_row = range.endRow - range.startRow;
    m_n_column = range.endColumn - range.startColumn;
    m_n_stride = matrix.m_n_stride;
    m_n_allocated = 0;
}   // end of SubMatrix


SubMatrix::SubMatrix( const SubMatrix& other ) {
    m_data = other.m_data;
    m_n_row = other.m_n_row;
    m_n_column = other.m_n_column;
    m_n_stride = other.m_n_stride;
    m_n_allocated = other.m_n_allocated;
}   // end of SubMatrix


SubMatrix& SubMatrix::operator = ( const SubMatrix& other ) {
    if ( this != &other ) {
        m_data = other.m_data;
        m_n_row = other.m_n_row;
        m_n_column = other.m_n_column;
        m_n_stride = other.m_n_stride;
        m_n_allocated = other.m_n_allocated;
    }
    return *this;
}   // end of operator =

// ==========================================================================================

ostream& operator << ( ostream& out, const MatrixBase& matrix ) {
    vector<float> buffer;
    matrix.GetData( &buffer );
    for ( int r = 0; r < matrix.Rows(); r++ ) {
        for ( int c = 0; c < matrix.Columns(); c++ ) {
            out << setw(8) << setprecision(4) << buffer[r * matrix.Columns() + c] << ' ';
        }
        out << endl;
    }
    return out;
}

// ==========================================================================================


#ifdef MATRIX_UNIT_TEST
int main( int argc, char** argv ) {
    Matrix gpu( 32, 32 );
    gpu.Fill( -1.0 );
    cout << gpu << endl;

    vector<float> cpu(8 * 16);
    for ( int i = 0; i < cpu.size(); i++ ) {
        cpu[i] = i / 100.0f;
        cpu[i] *= cpu[i];
    }
    SubMatrix sm1( gpu, MatrixRange(0, 8, 0, 16) );
    sm1.SetData( cpu );
    cout << gpu << endl;

    SubMatrix sm2( gpu, MatrixRange(0, 8, 16, 32) );
    sm2.Exp( sm1 );
    cout << gpu << endl;

    SubMatrix sm3( gpu, MatrixRange(8, 16, 0, 16) );
    sm3.Softmax( sm1 );
    cout << gpu << endl;

    Matrix index( 9, 1 );
    float idx[] = { 3, 7, 9, 7, 3, 4, 0, 7, 5 };
    index.SetData( vector<float>( idx, idx + 9 ) );

    SubMatrix sm4( gpu, MatrixRange(8, 17, 16, 32) );
    sm4.GetRows( SubMatrix( gpu, MatrixRange(0, 32, 0, 16) ), index );
    cout << gpu << endl;

    SubMatrix sm5( gpu, MatrixRange( 24, 32, 0, 32 ) );
    sm5.Random( -1.0f, 1.0f );
    cout << gpu << endl;

    // ======================================================================

    vector<float> buffer( 49 * 321 );
    for ( int i = 0; i < buffer.size(); i++ ) {
        int r = i / 321;
        int c = i % 321;
        buffer[i] = (r + c) / 100.0f;
    }

    Matrix easy( 49, 321 );
    easy.SetData( buffer );
    cout << easy << endl;

    Matrix large( 128, 512 );
    SubMatrix row( large, MatrixRange( 1, 2, 0, 321 ) );
    SubMatrix column( large, MatrixRange( 0, 49, 1, 2 ) );

    row.SumRowsOf( easy );
    cout << row << endl;

    column.SumColumnsOf( easy );
    cout << column << endl;

    easy.Add( 0.0f, 1.0f, row );
    cout << easy << endl;

    easy.Add( 0.0f, 1.0f, column );
    cout << easy << endl;

    easy.Add( 0.0f, 1.0f, SubMatrix( row, MatrixRange( 0, 1, 0, 49) ), CUBLAS_OP_T );
    cout << easy << endl;

    // ======================================================================

    Matrix nimeiG( 36, 128 );
    vector<float> nimeiC( 36 * 128 );
    for ( int i = 0; i < nimeiG.Rows(); i++ ) {
        for ( int j = 0; j < nimeiG.Columns(); j++ ) {
            nimeiC[i * nimeiG.Columns() + j] = j;
        }
    }
    nimeiG.SetData( nimeiC );
    cout << nimeiG << endl;

    index.Reshape( 9, 1 );
    float qq[] = { 30, 70, 90, 70, 30, 40, 0, 70, 50};
    index.SetData( vector<float>( qq, qq + sizeof(qq) / sizeof(float) ) );

    Matrix phuck( 36, 9 );
    phuck.GetColumns( nimeiG, index );
    cout << phuck << endl;


    return 0;
}
#endif
