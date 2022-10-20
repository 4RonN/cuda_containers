
#include "gpu_arrays.hpp"

/**
 *   @brief helper function to map device function to each element of a gpu_vector
 */ 
template< typename F, typename T >
__global__ void map_gpu( F f, gpu_vector< T > x )
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
           stride = blockDim.x * gridDim.x; 
     
    for ( ; idx < x.size(); idx += stride )
        x[ idx ] = f( x[ idx ] ); 
}    

/**
 *   @brief map device function to each element of a gpu_vector
 */ 
template< typename F, typename T >
void map( F const& f, gpu_vector< T > const& x )
{
    // Start with one block of 64 threads. This is very specific, 
    // and should usually be tailored to the problem and hardware.
    map_gpu <<< 1, 64 >>> ( f, x );  
}

int main()
{
    std::vector< int > x_host = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    auto x_device = push( x_host );
    auto f = [] __device__ ( float tmp ) { return tmp * tmp; }; // device lambda function 
    
    map( f, x_device  
    
    auto y = pull( x_device ); // pull results to new host memory array
    
    for ( auto y_i : y  )
        std::cout << y_i << std::endl; // print results
    
    async_vector< int > a = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    return 0;
}

