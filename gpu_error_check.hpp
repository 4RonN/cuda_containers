#ifndef __GPU_ERROR_CHECK_HPP
#define __GPU_ERROR_CHECK_HPP

#include <cstdio>

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

inline __host__ __device__ void error_check( cudaError_t error, const char* loc ="unknown location" )
{ 
    if ( error != 0 )
    {
        #ifndef __CUDA_ARCH__
            printf( "error: %s, ( %s )", cudaGetErrorString( error ), loc );
            cudaDeviceSynchronize();
            exit( -1 );
        #else
            printf( "error: %s", loc );
        #endif
    }
}

inline __host__ __device__ void error_check( const char* loc ="unknown location" )
{ 
    cudaError_t error = cudaGetLastError();
    error_check( error, loc );
}

#endif // __GPU_ERROR_CHECK_HPP
