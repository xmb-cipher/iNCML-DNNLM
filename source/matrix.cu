/*
Author      : Mingbin Xu (mingbin.xu@gmail.com)
Filename    : matrix.cu
Last Update : Feb 2, 2016
Description : All low-level implementation in CUDA is hidden in this file
Website     : https://wiki.eecs.yorku.ca/lab/MLL/

Copyright (c) 2016 iNCML (author: Mingbin Xu)
License: MIT License (see ../LICENSE)
 */

// nvcc -Xcompiler -rdynamic -lcurand -lcublas -lcusparse -o ../main matrix.cu -DMATRIX_UNIT_TEST


#include "matrix.h"

struct TestNanOrInf {
    __host__ __device__ bool operator() ( const float x ) const {
        return isnan(x) || isinf(x);
    }
};


int closest2Power(int x) {
    assert( x > 0 );
    int result = 1;
    while ( result <= x ) {
        result *= 2;
    }
    return result / 2;
}



__global__ static  void Clip( float* value, MatrixShape shape, float lower, float upper ) {
    int r = threadIdx.x + blockDim.x * blockIdx.x;
    int c = threadIdx.y + blockDim.y * blockIdx.y;
    if ( r < shape.row && c < shape.column ) {
        float __result = value[r * shape.stride + c];
        if ( isinf(__result) || isnan(__result) ) {
            __result = 0.0f;
        }
        else if ( __result < lower ) {
            __result = lower;
        }
        else if ( __result > upper ) {
            __result = upper;
        }
        value[r * shape.stride + c] = __result;
    }
}   // end of Clip



__global__ static void L2Norm( const float* __restrict__ value, MatrixShape shape,
                               float* result, int stride ) {
    extern __shared__ float buffer[];   // length is blockDim.x

    buffer[threadIdx.x] = 0.0f;
    const float* bor = value + blockIdx.x * shape.stride;

    for ( int i = threadIdx.x; i < shape.column; i += blockDim.x ) {
        buffer[threadIdx.x] += bor[i] * bor[i];
    }
    __syncthreads();

    if ( blockDim.x >= 512 ) {
        if ( threadIdx.x < 256 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 256];
        }
        __syncthreads();
    }

    if ( blockDim.x >= 256 ) {
        if ( threadIdx.x < 128 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if ( blockDim.x >= 128 ) {
        if ( threadIdx.x < 64 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 64];
        }
        __syncthreads();
    } 

    if ( threadIdx.x < 32 ) {
        if ( blockDim.x >= 64 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 32];
        }
        if ( blockDim.x >= 32 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 16];
        }
        if ( blockDim.x >= 16 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 8];
        }
        if ( blockDim.x >= 8 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 4];
        }
        if ( blockDim.x >= 4 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 2];
        }
        if ( blockDim.x >= 2 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 1];
        }
    }

    if ( threadIdx.x == 0 ) {
        result[blockIdx.x * stride] = buffer[0];
    }
}   // end of L2Norm





__global__ static void SumReduce( const float* __restrict__ value, MatrixShape shape,
                                  float* result, int stride ) {
    extern __shared__ float buffer[];   // length is blockDim.x

    buffer[threadIdx.x] = 0.0f;
    const float* bor = value + blockIdx.x * shape.stride;

    for ( int i = threadIdx.x; i < shape.column; i += blockDim.x ) {
        buffer[threadIdx.x] += bor[i];
    }
    __syncthreads();

    if ( blockDim.x >= 512 ) {
        if ( threadIdx.x < 256 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 256];
        }
        __syncthreads();
    }

    if ( blockDim.x >= 256 ) {
        if ( threadIdx.x < 128 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if ( blockDim.x >= 128 ) {
        if ( threadIdx.x < 64 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 64];
        }
        __syncthreads();
    } 

    if ( threadIdx.x < 32 ) {
        if ( blockDim.x >= 64 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 32];
        }
        if ( blockDim.x >= 32 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 16];
        }
        if ( blockDim.x >= 16 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 8];
        }
        if ( blockDim.x >= 8 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 4];
        }
        if ( blockDim.x >= 4 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 2];
        }
        if ( blockDim.x >= 2 ) {
            buffer[threadIdx.x] += buffer[threadIdx.x + 1];
        }
    }

    if ( threadIdx.x == 0 ) {
        result[blockIdx.x * stride] = buffer[0];
    }
}   // end of SumReduce


__global__ static void MaxReduce( const float* __restrict__ value, MatrixShape shape,
                                  float* result, int stride ) {
    extern __shared__ float buffer[];   // length is blockDim.x

    buffer[threadIdx.x] = FLT_MIN;
    const float* bor = value + blockIdx.x * shape.stride;

    for ( int i = threadIdx.x; i < shape.column; i += blockDim.x ) {
        buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], bor[i]);
    }
    __syncthreads();

    if ( blockDim.x >= 512 ) {
        if ( threadIdx.x < 256 ) {
            buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 256]);
        }
        __syncthreads();
    }

    if ( blockDim.x >= 256 ) {
        if ( threadIdx.x < 128 ) {
            buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 128]);
        }
        __syncthreads();
    } 

    if ( blockDim.x >= 128 ) {
        if ( threadIdx.x < 64 ) {
            buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 64]);
        }
        __syncthreads();
    } 

    if ( blockDim.x >= 64 && threadIdx.x < 32 ) {
        buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 32]);
    }
    if ( blockDim.x >= 32 && threadIdx.x < 16 ) {
        buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 16]);
    }
    if ( blockDim.x >= 16 && threadIdx.x < 8 ) {
        buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 8]);
    }
    if ( blockDim.x >= 8 && threadIdx.x < 4 ) {
        buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 4]);
    }
    if ( blockDim.x >= 4 && threadIdx.x < 2 ) {
        buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 2]);
    }
    if ( blockDim.x >= 2 && threadIdx.x < 1 ) {
        buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], buffer[threadIdx.x + 1]);
    }

    if ( threadIdx.x == 0 ) {
        result[blockIdx.x * stride] = buffer[0];
    }
}   // end of MaxReduce



__global__ static void MaxReduce( const float* __restrict__ value, MatrixShape shape,
                                  float* result, int r_stride,
                                  float* index, int i_stride ) {
    extern __shared__ float buffer[];   // length is blockDim.x
    int* ibuffer = (int*)(buffer + blockDim.x);

    buffer[threadIdx.x] = FLT_MIN;
    ibuffer[threadIdx.x] = threadIdx.x;
    const float* bor = value + blockIdx.x * shape.stride;

    for ( int i = threadIdx.x; i < shape.column; i += blockDim.x ) {
        if ( bor[i] > buffer[threadIdx.x] ) {
            buffer[threadIdx.x] = bor[i];
            ibuffer[threadIdx.x] = i;
        }
    }
    __syncthreads();

    if ( blockDim.x >= 512 ) {
        if ( threadIdx.x < 256 ) {
            if ( buffer[threadIdx.x + 256] > buffer[threadIdx.x] ) {
                buffer[threadIdx.x] = buffer[threadIdx.x + 256];
                ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 256];
            }
        }
        __syncthreads();
    }

    if ( blockDim.x >= 256 ) {
        if ( threadIdx.x < 128 ) {
            if ( buffer[threadIdx.x + 128] > buffer[threadIdx.x] ) {
                buffer[threadIdx.x] = buffer[threadIdx.x + 128];
                ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 128];
            }
        }
        __syncthreads();
    } 

    if ( blockDim.x >= 128 ) {
        if ( threadIdx.x < 64 ) {
            if ( buffer[threadIdx.x + 64] > buffer[threadIdx.x] ) {
                buffer[threadIdx.x] = buffer[threadIdx.x + 64];
                ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 64];
            }
        }
        __syncthreads();
    } 

    if ( blockDim.x >= 64 && threadIdx.x < 32 ) {
        if ( buffer[threadIdx.x + 32] > buffer[threadIdx.x] ) {
            buffer[threadIdx.x] = buffer[threadIdx.x + 32];
            ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 32];
        }
    }
    if ( blockDim.x >= 32 && threadIdx.x < 16 ) {
        if ( buffer[threadIdx.x + 16] > buffer[threadIdx.x] ) {
            buffer[threadIdx.x] = buffer[threadIdx.x + 16];
            ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 16];
        }
    }
    if ( blockDim.x >= 16 && threadIdx.x < 8 ) {
        if ( buffer[threadIdx.x + 8] > buffer[threadIdx.x] ) {
            buffer[threadIdx.x] = buffer[threadIdx.x + 8];
            ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 8];
        }
    }
    if ( blockDim.x >= 8 && threadIdx.x < 4 ) {
        if ( buffer[threadIdx.x + 4] > buffer[threadIdx.x] ) {
            buffer[threadIdx.x] = buffer[threadIdx.x + 4];
            ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 4];
        }
    }
    if ( blockDim.x >= 4 && threadIdx.x < 2 ) {
        if ( buffer[threadIdx.x + 2] > buffer[threadIdx.x] ) {
            buffer[threadIdx.x] = buffer[threadIdx.x + 2];
            ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 2];
        }
    }
    if ( blockDim.x >= 2 && threadIdx.x < 1 ) {
        if ( buffer[threadIdx.x + 1] > buffer[threadIdx.x] ) {
            buffer[threadIdx.x] = buffer[threadIdx.x + 1];
            ibuffer[threadIdx.x] = ibuffer[threadIdx.x + 1];
        }
    }

    if ( threadIdx.x == 0 ) {
        result[blockIdx.x * r_stride] = buffer[0];
        index[blockIdx.x * i_stride] = (float)ibuffer[0];
    }
}   // end of MaxReduce



__global__ static void MinReduce( const float* __restrict__ value, MatrixShape shape,
                                  float* result, int stride ) {
    extern __shared__ float buffer[];   // length is blockDim.x

    buffer[threadIdx.x] = FLT_MAX;
    const float* bor = value + blockIdx.x * shape.stride;

    for ( int i = threadIdx.x; i < shape.column; i += blockDim.x ) {
        buffer[threadIdx.x] = fminf(buffer[threadIdx.x], bor[i]);
    }
    __syncthreads();

    if ( blockDim.x >= 512 ) {
        if ( threadIdx.x < 256 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 256]);
        }
        __syncthreads();
    }

    if ( blockDim.x >= 256 ) {
        if ( threadIdx.x < 128 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 128]);
        }
        __syncthreads();
    } 

    if ( blockDim.x >= 128 ) {
        if ( threadIdx.x < 64 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 64]);
        }
        __syncthreads();
    } 

    if ( threadIdx.x < 32 ) {
        if ( blockDim.x >= 64 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 32]);
        }
        if ( blockDim.x >= 32 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 16]);
        }
        if ( blockDim.x >= 16 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 8]);
        }
        if ( blockDim.x >= 8 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 4]);
        }
        if ( blockDim.x >= 4 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 2]);
        }
        if ( blockDim.x >= 2 ) {
            buffer[threadIdx.x] = fminf(buffer[threadIdx.x], buffer[threadIdx.x + 1]);
        }
    }

    if ( threadIdx.x == 0 ) {
        result[blockIdx.x * stride] = buffer[0];
    }
}   // end of MinReduce



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


// template <FunctionType type>
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

/*
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
}   // end of ArgMax    */


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
        float __result = 0.0f;
        switch ( type ) {
            case kRelu:
                outP[dstIdx] = fmaxf( 0.0f, inP[srcIdx] );
                break;
                
            case kExp: // clip it/ceiling
                __result = inP[srcIdx] ;
                outP[dstIdx] = expf(__result > 81.0f ? 81.0f: __result)  ;
                break;
                
            case kReciprocal:
                __result = inP[srcIdx];
                outP[dstIdx]  = 1.0f /  (__result < 1e-32 ? 1e-32: __result);
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
                __result = inP[srcIdx]  ;
                outP[dstIdx] = logf(__result < 1e-32 ? 1e-32 : __result)  ;
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

// TODO: make sure SparseMatrix and Matrix are on the same GPU

cusparseHandle_t SparseMatrix::handle = NULL;
cusparseMatDescr_t SparseMatrix::descriptor = NULL;
int SparseMatrix::n_matrix = 0;
Matrix* SparseMatrix::buffer = NULL;


SparseMatrix::SparseMatrix() :
        m_nnz( 0 ), 
        m_n_row( 0 ), 
        m_n_column( 0 ),
        m_data( NULL ),
        m_indices( NULL ),
        m_indptr( NULL ),
        m_n_byte( 0 )    {
    if ( 0 == SparseMatrix::n_matrix ) {
        SparseMatrix::buffer = new Matrix( 16, 16 );
        CUSPARSE_CHECK( cusparseCreate( &SparseMatrix::handle ), 
                        "cusparseCreate" );
        CUSPARSE_CHECK( cusparseCreateMatDescr( &SparseMatrix::descriptor ),
                        "cusparseCreateMatDescr" ) ; 
        CUSPARSE_CHECK( cusparseSetMatType( SparseMatrix::descriptor, 
                                            CUSPARSE_MATRIX_TYPE_GENERAL ),
                        "cusparseSetMatType" );
        CUSPARSE_CHECK( cusparseSetMatIndexBase( SparseMatrix::descriptor, 
                                                 CUSPARSE_INDEX_BASE_ZERO ),
                        "cusparseSetMatIndexBase" );
    }
    SparseMatrix::n_matrix++;
    ASSERT( NULL == m_data );
    ASSERT( NULL == m_indices );
    ASSERT( NULL == m_indptr );
}    // end of SparseMatrix


SparseMatrix::SparseMatrix( const vector<float>& data, 
                            const vector<int>& indices, 
                            const vector<int>& indptr,
                            int n_column ) : 
        m_data( NULL ),
        m_indices( NULL ),
        m_indptr( NULL),
        m_n_byte( 0 )    { 
    if ( 0 == SparseMatrix::n_matrix ) {
        SparseMatrix::buffer = new Matrix( 16, 16 );
        CUSPARSE_CHECK( cusparseCreate( &SparseMatrix::handle ), 
                        "cusparseCreate" );
        CUSPARSE_CHECK( cusparseCreateMatDescr( &SparseMatrix::descriptor ),
                        "cusparseCreateMatDescr" ) ; 
        CUSPARSE_CHECK( cusparseSetMatType( SparseMatrix::descriptor, 
                                            CUSPARSE_MATRIX_TYPE_GENERAL ),
                        "cusparseSetMatType" );
        CUSPARSE_CHECK( cusparseSetMatIndexBase( SparseMatrix::descriptor, 
                                                 CUSPARSE_INDEX_BASE_ZERO ),
                        "cusparseSetMatIndexBase" );
    }
    SetData( data, indices, indptr );
    SparseMatrix::n_matrix++;
}    // end of SparseMatrix


SparseMatrix::~SparseMatrix() {
    if ( NULL != m_data ) {
        CUDA_CHECK( cudaFree( m_data ), "cudaFree" );
        m_data = NULL;
        m_indices = NULL;
        m_indptr = NULL;
    }
    
    SparseMatrix::n_matrix--;
    if ( 0 == SparseMatrix::n_matrix ) {
        delete SparseMatrix::buffer;
        SparseMatrix::buffer = NULL;
        CUSPARSE_CHECK( cusparseDestroyMatDescr( SparseMatrix::descriptor ),
                        "cusparseDestroyMatDescr" ); 
        SparseMatrix::descriptor = NULL;
        CUSPARSE_CHECK( cusparseDestroy( SparseMatrix::handle ),
                        "cusparseDestroy" );
        SparseMatrix::handle = NULL;
    }
}    // end of ~SparseMatrix        


void SparseMatrix::SetData( const vector<float>& data, 
                            const vector<int>& indices, 
                            const vector<int>& indptr,
                            int n_column ) {
    m_nnz = data.size();
    m_n_row = indptr.size() - 1;
    ASSERT( m_nnz == indices.size() );
    ASSERT( m_nnz == indptr.back() );
    ASSERT( indptr.size() > 0 && indptr[0] == 0 );
    
    m_n_column = max( *max_element( indices.begin(), indices.end() ) + 1, n_column );

    int total_size = sizeof(float) * m_nnz
                    + sizeof(float) * m_nnz
                    + sizeof(int) * indptr.size();
    void* buffer = operator new( total_size );
    if ( m_nnz > 0 ) {
        memcpy( buffer, &data[0], sizeof(float) * m_nnz );
        memcpy( (float*)buffer + m_nnz, &indices[0], sizeof(int) * m_nnz );
    }
    memcpy( (int*)((float*)buffer + m_nnz) + m_nnz, 
            &indptr[0], 
            sizeof(int) * indptr.size() );
    
    if ( NULL != m_data && total_size > m_n_byte ) {
        CUDA_CHECK( cudaFree( m_data ), "cudaFree" );
        m_data = NULL;
    }

    if ( NULL == m_data ) {
        CUDA_CHECK( cudaMalloc( (void**)&m_data, total_size ), "cudaMalloc" );
        m_n_byte = total_size;
    }
    
    CUDA_CHECK( cudaMemcpy( m_data, 
                            buffer, 
                            total_size, 
                            cudaMemcpyHostToDevice ), 
                "cudaMemcpy" );

    m_indices = (int*)(m_data + m_nnz);
    m_indptr = m_indices + m_nnz;

    operator delete( buffer );
}    // end of SetData

                

// ==========================================================================================

float* MatrixBase::ones = NULL;
float* MatrixBase::zeros = NULL;
float* MatrixBase::buffer = NULL;
int MatrixBase::buffer_size = 0;
int MatrixBase::n_matrix = 0;
cublasHandle_t MatrixBase::handle = NULL;
curandGenerator_t MatrixBase::generator = NULL;
cudaStream_t MatrixBase::stream = 0;

#ifdef XT
    cublasXtHandle_t MatrixBase::xt_handle = NULL;
#endif


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


void MatrixBase::SetDataAsync( const vector<float>& data ) {
    ASSERT( data.size() == m_n_row * m_n_column );
    CUDA_CHECK( cudaMemcpy2DAsync( m_data,
                                   m_n_stride * sizeof(float),
                                   &data[0],
                                   m_n_column * sizeof(float),
                                   m_n_column * sizeof(float),
                                   m_n_row,
                                   cudaMemcpyHostToDevice,
                                   MatrixBase::stream ),
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
#ifdef XT
    // TODO: during backpropagation, this function gives segmentation fault
    //       However, all data transfer from these three matrices to host memory 
    //       don't hit any error:(
    CUBLAS_CHECK( cublasXtSgemm( MatrixBase::xt_handle,
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
                  "cublasXtSgemm" );
#else
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
#endif
}   // end of Sgemm


void MatrixBase::Sgemm( float beta, float alpha,
                        const MatrixBase& A, cusparseOperation_t transa,
                        const SparseMatrix& B, cusparseOperation_t transb ) {
    CUSPARSE_CHECK( cusparseScsrmm2( SparseMatrix::handle,
                                     (transb == CUSPARSE_OPERATION_NON_TRANSPOSE ?
                                        CUSPARSE_OPERATION_TRANSPOSE : 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE),
                                     transa,
                                     m_n_column,
                                     m_n_row,
                                     (transb == CUSPARSE_OPERATION_NON_TRANSPOSE ? 
                                                    B.m_n_column : B.m_n_row),
                                     B.m_nnz,
                                     &alpha,
                                     SparseMatrix::descriptor,
                                     B.m_data,
                                     B.m_indptr,
                                     B.m_indices,
                                     A.m_data,
                                     A.m_n_stride,
                                     &beta,
                                     m_data,
                                     m_n_stride ), 
                    "cusparseScsrmm2" );
    CUDA_CHECK( cudaDeviceSynchronize(), "cudaDeviceSynchronize" );
}    // end of Sgemm


void MatrixBase::Sgemm( float beta, float alpha,
                        const SparseMatrix& A, cusparseOperation_t transa,
                        const MatrixBase& B, cusparseOperation_t transb ) {
    SparseMatrix::buffer->Reshape( m_n_column, m_n_row );
    SparseMatrix::buffer->Sgemm( 0.0f, 1.0f,
                                 B, ( transa == CUSPARSE_OPERATION_TRANSPOSE ? 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE : 
                                        CUSPARSE_OPERATION_TRANSPOSE ),
                                 A, ( transa == CUSPARSE_OPERATION_TRANSPOSE ? 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE : 
                                        CUSPARSE_OPERATION_TRANSPOSE ) );
    this->Add( beta, alpha, *SparseMatrix::buffer, CUBLAS_OP_T );
}   // end of Sgemm


void MatrixBase::Strmm( float alpha,
                        const MatrixBase& A, 
                        cublasSideMode_t side, 
                        cublasFillMode_t uplo,
                        const MatrixBase& B, 
                        cublasOperation_t trans ) {
    if ( side == CUBLAS_SIDE_LEFT ) {
        side = CUBLAS_SIDE_RIGHT;
    }
    else if ( side == CUBLAS_SIDE_RIGHT ) {
        side = CUBLAS_SIDE_LEFT;
    }
    else {
        ASSERT( false );
    }

    if ( uplo == CUBLAS_FILL_MODE_LOWER ) {
        uplo = CUBLAS_FILL_MODE_UPPER;
    }
    else if ( uplo == CUBLAS_FILL_MODE_UPPER ) {
        uplo = CUBLAS_FILL_MODE_LOWER;
    }
    else {
        ASSERT( false );
    }

    CUBLAS_CHECK( cublasStrmm( MatrixBase::handle,
                               side,
                               uplo,
                               trans,
                               CUBLAS_DIAG_NON_UNIT,
                               m_n_column,
                               m_n_row,
                               &alpha,
                               A.m_data,
                               A.m_n_stride,
                               B.m_data,
                               B.m_n_stride,
                               m_data,
                               m_n_stride ), 
                  "cublasStrmm()" );
}



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
    const float MINUS_ONE = -1.0f;
    const float ZERO = 0.0f;

    int __grid = value.m_n_row;
    int __block = closest2Power( value.m_n_column );
    if ( __block > 512 ) {
        __block = 512;
    }
    int __nByte = sizeof(float) * __block;

    ::MaxReduce<<<__grid, __block, __nByte>>>( value.m_data, 
                                               value.Shape(), 
                                               MatrixBase::buffer, 
                                               1 );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    Copy( value );

#ifdef XT
    CUBLAS_CHECK( cublasXtSgemm( MatrixBase::xt_handle,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   m_n_column,
                                   m_n_row,
                                   1,
                                   &MINUS_ONE,
                                   MatrixBase::ones,
                                   m_n_column,
                                   MatrixBase::buffer,
                                   1,
                                   &ONE,
                                   m_data,
                                   m_n_stride ), 
                  "cublasXtSgemm" );
    CUDA_CHECK( cudaDeviceSynchronize(), "cudaDeviceSynchronize" );
#else
    CUBLAS_CHECK( cublasSgemm( MatrixBase::handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               m_n_column,
                               m_n_row,
                               1,
                               &MINUS_ONE,
                               MatrixBase::ones,
                               m_n_column,
                               MatrixBase::buffer,
                               1,
                               &ONE,
                               m_data,
                               m_n_stride ), 
                  "cublasSgemm" );
#endif

    Exp( *this );

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



void MatrixBase::Add( float alpha, const MatrixBase& other ) {
    ASSERT( other.m_n_row == 1 && other.m_n_column == m_n_column || 
            other.m_n_column == 1 && other.m_n_row == m_n_row );
    if ( other.m_n_row == 1 ) {
        CUBLAS_CHECK( cublasSger( MatrixBase::handle,
                                  m_n_column,
                                  m_n_row,
                                  &alpha,
                                  other.m_data,
                                  1,
                                  MatrixBase::ones,
                                  1,
                                  m_data,
                                  m_n_stride ), 
                      "cublasSger" );
    }
    else {
        CUBLAS_CHECK( cublasSger( MatrixBase::handle,
                                  m_n_column,
                                  m_n_row,
                                  &alpha,
                                  MatrixBase::ones,
                                  1,
                                  other.m_data,
                                  other.m_n_stride,
                                  m_data,
                                  m_n_stride ), 
                      "cublasSger" );
    }
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


void MatrixBase::ArgMax( const MatrixBase& value ) {/*
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
    KERNEL_CHECK( __PRETTY_FUNCTION__ );*/

    ASSERT( m_n_column == 2 || m_n_column == 1 );
    int grid = m_n_row;
    int block = closest2Power( m_n_column );
    if ( block > 512 ) {
        block = 512;
    }
    int nByte = sizeof(float) * block + sizeof(int) * block;

    ::MaxReduce<<<grid, block, nByte>>>( value.m_data, value.Shape(), 
                                         (m_n_column == 2 ? m_data + 1 : MatrixBase::buffer), 
                                         (m_n_column == 2 ? m_n_stride : 1),
                                         m_data, m_n_stride );
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
    float result = thrust::reduce( buffer, buffer + m_n_row, 0.0f, thrust::plus<float>() );

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
    int grid = m_n_row;
    int block = closest2Power( m_n_column );
    if ( block > 512 ) {
        block = 512;
    }
    int nByte = sizeof(float) * block;

    ::MinReduce<<<grid, block, nByte>>>( m_data, Shape(), MatrixBase::buffer, 1 );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( MatrixBase::buffer );

    float result = thrust::reduce( deviceP, 
                                   deviceP + m_n_row, 
                                   numeric_limits<float>::max(),
                                   thrust::minimum<float>() );
    return result;
}


float MatrixBase::Max() {
    int grid = m_n_row;
    int block = closest2Power( m_n_column );
    if ( block > 512 ) {
        block = 512;
    }
    int nByte = sizeof(float) * block;

    ::MaxReduce<<<grid, block, nByte>>>( m_data, Shape(), MatrixBase::buffer, 1 );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( MatrixBase::buffer );

    float result = thrust::reduce( deviceP, 
                                   deviceP + m_n_row, 
                                   numeric_limits<float>::min(),
                                   thrust::maximum<float>() );
    return result;
}


float MatrixBase::Sum() {
    int grid = m_n_row;
    int block = closest2Power( m_n_column );
    if ( block > 512 ) {
        block = 512;
    }
    int nByte = sizeof(float) * block;

    ::SumReduce<<<grid, block, nByte>>>( m_data, Shape(), MatrixBase::buffer, 1 );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( MatrixBase::buffer );

    float result = thrust::reduce( deviceP, 
                                   deviceP + m_n_row, 
                                   0.0f,
                                   thrust::plus<float>() );
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


void MatrixBase::m_clear_buffer() {
    if ( MatrixBase::buffer != NULL && MatrixBase::buffer_size > 0 ) {
        thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( MatrixBase::buffer );
        thrust::fill( deviceP, deviceP + MatrixBase::buffer_size, 0.0f );
    }
}    // end of m_clear_buffer


float MatrixBase::L2Norm() {
    int grid = m_n_row;
    int block = closest2Power( m_n_column );
    if ( block > 512 ) {
        block = 512;
    }
    int nByte = sizeof(float) * block;

    ::L2Norm<<<grid, block, nByte>>>( m_data, Shape(), MatrixBase::buffer, 1 );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );

    thrust::device_ptr<float> deviceP = thrust::device_pointer_cast( MatrixBase::buffer );

    float result = thrust::reduce( deviceP, 
                                   deviceP + m_n_row, 
                                   0.0f,
                                   thrust::plus<float>() );
    return sqrt(result);
}    // end of L2Norm


void MatrixBase::Clip( float lower, float upper ) {
    ASSERT( lower < upper );
    dim3 block( CU2DBLOCK, CU2DBLOCK );
    dim3 grid( GridDim(m_n_row, CU2DBLOCK), GridDim(m_n_column, CU2DBLOCK) );
    ::Clip<<<grid, block>>>( m_data, Shape(), lower, upper );
    KERNEL_CHECK( __PRETTY_FUNCTION__ );
}   // end of Clip


void MatrixBase::Sparse2Dense( const SparseMatrix& other ) {
    ASSERT( m_n_row == other.m_n_row && m_n_column == other.m_n_column );
    
    // SparseMatrix is in CSR, but MatrixBase is row-major. 
    // cusparseScsr2dense is used instead.
    CUSPARSE_CHECK( cusparseScsc2dense( SparseMatrix::handle,
                                        m_n_column,
                                        m_n_row,
                                        SparseMatrix::descriptor,
                                        other.m_data,
                                        other.m_indices,
                                        other.m_indptr,
                                        m_data,
                                        m_n_stride ), 
                    "cusparseScsc2dense" );
    CUDA_CHECK( cudaDeviceSynchronize(), "cudaDeviceSynchronize" );
}    // end of SparseToDense


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
        CUDA_CHECK( cudaStreamCreate( &MatrixBase::stream), "cudaStreamCreate" );

    #ifdef XT
        int device[] = {0, 1};
        CUBLAS_CHECK( cublasXtCreate( &MatrixBase::xt_handle ), "cublasXtCreate" );
        CUBLAS_CHECK( cublasXtDeviceSelect( MatrixBase::xt_handle, 2, device ),
                      "cublasXtDeviceSelect" );
        CUDA_CHECK( cudaGetDevice( &bestChoice ), "cudaGetDevice" );
        cout << CurrentTime() << ") cublasXT is enabled. " 
             << bestChoice << " is the primary device." << endl;
    #endif

    #ifdef MULTIPLE
        int canAccess = 0;
        CUDA_CHECK( cudaDeviceCanAccessPeer( &canAccess, 0, 1 ), "cudaDeviceCanAccessPeer");
        cout << CurrentTime() << ") Device 0 " << (canAccess ? "can" : "cannot")
             << " access Device 1" << endl;
        if ( canAccess ) {
            CUDA_CHECK( cudaSetDevice(0), "cudaSetDevice" );
            CUDA_CHECK( cudaDeviceEnablePeerAccess(1, 0), "cudaDeviceEnablePeerAccess" );
        }
        CUDA_CHECK( cudaDeviceCanAccessPeer( &canAccess, 1, 0 ), "cudaDeviceCanAccessPeer");
        cout << CurrentTime() << ") Device 1 " << (canAccess ? "can" : "cannot")
             << " access Device 0" << endl;
        if ( canAccess ) {
            CUDA_CHECK( cudaSetDevice(1), "cudaSetDevice" );
            CUDA_CHECK( cudaDeviceEnablePeerAccess(0, 0), "cudaDeviceEnablePeerAccess" );
        }

        CUDA_CHECK( cudaSetDevice( bestChoice ), "cudaSetDevice" );
    #endif 
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

    #ifdef XT
        CUBLAS_CHECK( cublasXtDestroy( MatrixBase::xt_handle ), "cublasXtDestroy" );
    #endif 
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
        thrust::device_ptr<float> __buffer = thrust::device_pointer_cast( MatrixBase::buffer );
        thrust::fill( __buffer, __buffer + MatrixBase::buffer_size, 0.0f );
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
    const int NUM_ROWS = 16;
    const int NUM_COLUMNS = 512;

    Matrix projection( NUM_COLUMNS, 128 );
    projection.Random( -1.0f, 1.0f );
    
    vector<float> dense;
    vector<float> data;
    vector<int> indices;
    vector<int> indptr( 1, 0 );
    
    for ( int r = 0; r < NUM_ROWS; r++ ) {
        for ( int c = 0; c < NUM_COLUMNS; c++ ) {
            if ( c % (r + 1) == 0 ) {
                float candidate = (r + c / NUM_COLUMNS) / NUM_ROWS;
                dense.push_back( candidate );
                data.push_back( candidate );
                indices.push_back( c );
            }
            else {
                dense.push_back( 0.0f );
            }
        }
        indptr.push_back( data.size() );
    }
    
    Matrix dense_matrix( NUM_ROWS, NUM_COLUMNS );
    dense_matrix.SetData( dense );
    SparseMatrix sparse_matrix( data, indices, indptr, NUM_COLUMNS );
    cout << CurrentTime() << ") constructors and SetData seem fine" << endl; 
    
    Matrix gpu_input( NUM_ROWS, NUM_COLUMNS );
    gpu_input.Sparse2Dense( sparse_matrix );
    cout << CurrentTime() << ") Sparse2Dense doesn't fail" << endl;
    
    vector<float> cpu_input;
    gpu_input.GetData( &cpu_input );
    int n_diff_byte = memcmp( &cpu_input[0], &dense[0], sizeof(float) * NUM_ROWS * NUM_COLUMNS );
    cout << CurrentTime() << ") number of different bytes: " << n_diff_byte << endl;
    if ( 0 == n_diff_byte ) {
        cout << CurrentTime() << ") Sparse2Dense seems correct" << endl;
    }
    
    Matrix gpu_out_from_sparse( NUM_ROWS, 128 );
    Matrix gpu_out_from_dense( NUM_ROWS, 128 );
    Matrix gpu_buffer( 128, NUM_ROWS );
    
    gpu_out_from_dense.Sgemm( 0.0f, 1.0f, 
                              dense_matrix, CUBLAS_OP_N, 
                              projection, CUBLAS_OP_N );
    gpu_buffer.Sgemm( 0.0f, 1.0f,
                      projection, CUSPARSE_OPERATION_TRANSPOSE,
                      sparse_matrix, CUSPARSE_OPERATION_TRANSPOSE );
    gpu_out_from_sparse.Add( 0.0f, 1.0f, gpu_buffer, CUBLAS_OP_T );
    cout << CurrentTime() << ") SparseMatrix doesn't throw exception" << endl;
    
    vector<float> cpu_out_from_dense;
    vector<float> cpu_out_from_sparse;
    gpu_out_from_dense.GetData( &cpu_out_from_dense );
    gpu_out_from_sparse.GetData( &cpu_out_from_sparse );
    n_diff_byte = memcmp( &cpu_out_from_dense[0], &cpu_out_from_sparse[0],
                          sizeof(float) * NUM_ROWS * 128 );
    cout << CurrentTime() << ") number of different bytes " << n_diff_byte << endl;
    if ( 0 == n_diff_byte ) {
        cout << CurrentTime() << ") SparseMatrix::Sgemm seems correct" << endl;
    }

    gpu_out_from_dense.Sgemm( 0.0f, 1.0f, 
                              sparse_matrix, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              projection, CUSPARSE_OPERATION_NON_TRANSPOSE );
    gpu_out_from_sparse.GetData( &cpu_out_from_sparse );
    n_diff_byte = memcmp( &cpu_out_from_dense[0], &cpu_out_from_sparse[0],
                          sizeof(float) * NUM_ROWS * 128 );
    cout << CurrentTime() << ") number of different bytes " << n_diff_byte << endl;
    if ( 0 == n_diff_byte ) {
        cout << CurrentTime() << ") SparseMatrix::Sgemm seems correct" << endl;
    }
    return 0;
}
#endif
