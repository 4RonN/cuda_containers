#ifndef __GPU_ARRAYS_HPP 
#define __GPU_ARRAYS_HPP 

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "gpu_error_check.hpp"

template< typename T >
struct gpu_vector
{
    // type definition:
    
    using value_type             = T;
    using reference              = T&;
    using const_reference        = const T&;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = std::reverse_iterator< iterator >;
    using const_reverse_iterator = std::reverse_iterator< const_iterator >;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;

    protected:

    // data storages:
    T*        store;
    size_type count;
    unsigned  use_count;

    public:

    // default empty constructor and move contructor:
    gpu_vector() = default;
    gpu_vector( gpu_vector && ) = default;

    /**
     *  @brief Copy constructor: acts like a std::shared_pointer
     */ 
    __host__ __device__ gpu_vector( gpu_vector const& rhs ) : store( rhs.store ), count( rhs.count ), use_count( rhs.use_count + 1 ) { }  

    /**
     *  @brief Construct gpu_vector with a centrain size
     */ 
    __host__ gpu_vector( size_type c ) : count( c ) 
    { 
        cudaMalloc( (void**) &store, count * sizeof( T ) ); error_check( "alloc gpu_vector" ); 
    }

    /**
     *  @brief Construct gpu_vector with a centrain size and initialize with c
     */ 
    __host__ gpu_vector( size_type c, char init ) : gpu_vector( c ) 
    { 
        memset( init );  
    }
    
    __host__ __device__ ~gpu_vector()
    { 
        if ( use_count == 0 and count != 0 ) // if this is the last reference, free the used memory 
        {
            cudaFree( store );
            error_check( "delete/free gpu_vector type" ); 
        }
    }

    /**
     *  @brief Resize gpu_vector to new size
     *  @param new_size the new size 
     */ 
    __host__ void resize( size_type new_size ) 
    {
        if ( count != 0 )
            cudaFree( store ); 

        error_check( cudaMalloc( (void**) &store, new_size * sizeof( T ) ), "alloc gpu_vector" ); 
        count = new_size; 
    }

    /**
     *  @brief Query the size of the array
     */ 
    __device__ __host__ size_type size() const { return count; }

    /**
     *  @brief array style acces to the element n 
     */ 
    __device__ reference  operator[]( size_type n ) const { return store[n]; }
               
    /**
     *  @brief Iterator at the begin of the memory
     */ 
    __device__ iterator       begin()        { return store; }
    __device__ const_iterator begin()  const { return store; }
    __device__ const_iterator cbegin() const { return store; }

    /**
     *  @brief Iterators at the end of the memory
     */ 
    __device__ iterator       end()          { return store + count; }
    __device__ const_iterator end()    const { return store + count; }
    __device__ const_iterator cend()   const { return store + count; }
    
    /**
     *  @brief Access the first element 
     */ 
    __device__ reference       front()        
    {
            return store[ 0 ];
    }
    __device__ const_reference front() const { return front(); } 

    /**
     *  @brief Access the last element 
     */ 
    __device__ reference       back()        
    {
            return store[ count-1 ];
    }
    __device__ const_reference back()  const { return back(); } 

    /**
     *  @brief Direct access to the memory via the returned pointer
     */ 
    __device__ __host__ value_type*  data() const { return store; }

    /**
     *  @brief memset the memory from the host  
     */ 
    __host__ void memset( char c ) 
    { 
        error_check( cudaMemset( store, c, sizeof( T ) * count ), "gpu_vector set" ); 
    }
};

template< class T >
struct async_vector 
{
    // type definition: 

    using value_type             = T;
    using reference              = T&;
    using const_reference        = const T&;
    using iterator               = T*;
    using const_iterator         = const T*;
    using reverse_iterator       = std::reverse_iterator< iterator >;
    using const_reverse_iterator = std::reverse_iterator< const_iterator >;
    using size_type              = size_t;
    using difference_type        = ptrdiff_t;

    private:

    T* host_store;     // CPU memory 
    T* device_store;   // GPU memory
 
    size_type count;      // number of items T 
    unsigned  use_count;  // number of copies using the same memory 
                          // the conting starts at 0 to be enable to make the class a POD type.
    
    public:

    // default empty constructor and move contructor:
    async_vector() = default;
    async_vector( async_vector && ) = default;

    /**
     *  @brief Copy constructor: acts like a std::shared_pointer
     */ 
    __host__ __device__ async_vector( async_vector const& rhs ) : host_store( rhs.host_store ), device_store( rhs.device_store ), 
                                                                   count( rhs.count ), use_count( rhs.use_count + 1 ) {}  
    /**
     *  @brief initialize the async_array with size c 
     */ 
    __host__ async_vector( size_type c ) : count( c ), use_count( 0 ) 
    { 
        cudaMallocHost( (void**) &host_store,   count * sizeof( T ) ); error_check( "alloc async_vector host" );  
        cudaMalloc    ( (void**) &device_store, count * sizeof( T ) ); error_check( "alloc async_vector device" ); 
    }
    
    __host__ async_vector( std::initializer_list< T > list ) : async_vector( list.size() )
    {
        //resize( list.size() );
        std::memcpy( begin(), list.begin(), list.size() * sizeof( T ) );   
    }

    __host__ async_vector& operator = ( std::initializer_list< T > list )
    {
        resize( list.size() );
        std::memcpy( begin(), list.begin(), list.size() * sizeof( T ) );   
    }
    
    __host__ __device__ ~async_vector()  
    { 
        if ( use_count == 0 and count != 0 ) // if this is the last reference, free the used memory
        {
            cudaFree( device_store ); 
            error_check( "delete/free async_vector device type" ); 
            cudaFreeHost( host_store ); 
            error_check( "delete/free async_vector host type" ); 
        }
        else 
            use_count -= 1;
    }
    
    /**
     *  @brief Resize gpu_vector to new size
     *  @param new_size the new size 
     */ 
    __host__ void resize( size_type new_size ) 
    {
        if ( count != 0 )
        {
            cudaFree( device_store ); 
            cudaFreeHost( host_store );
        }

        cudaMallocHost( (void**) &host_store,   new_size * sizeof( T ) ); error_check( "alloc async_vector host" );  
        cudaMalloc    ( (void**) &device_store, new_size * sizeof( T ) ); error_check( "alloc async_vector device" ); 
        count = new_size; 
    }

    __host__ __device__ size_type size() const { return count; }
    
    /**
     *  @brief Copy host to device memory
     */
    void push() 
    {
        cudaMemcpy( device_store, host_store, count * sizeof( T ), cudaMemcpyHostToDevice );
        error_check( "async_vector::push" );
    }

    /**
     *  @brief Copy device to host memory
     */
    void pull()
    {
        cudaMemcpy( host_store, device_store, count * sizeof( T ), cudaMemcpyDeviceToHost );
        error_check( "async_vector::pull" );
    }

    /**
     *  @brief Iterator at begin, depending if executed on the host or device, the respective pointer is returned.  
     */
    __host__ __device__ iterator       begin()        
    {
        #ifdef __CUDA_ARCH__ 
            return device_store;
        #else 
            return host_store;
        #endif 
    }
    __host__ __device__ const_iterator begin()  const { return begin(); } // inherit the feature to const criteria 
    __host__ __device__ const_iterator cbegin() const { return begin(); } 
    
    /**
     *  @brief End iterator, depending if executed on the host or device, the respective pointer is returned.  
     */
    __host__ __device__ iterator       end()          
    {
        #ifdef __CUDA_ARCH__ 
            return device_store + count;
        #else 
            return host_store   + count;
        #endif 
    }
    __host__ __device__ const_iterator end()  const { return end(); } // inherit the feature to const criteria 
    __host__ __device__ const_iterator cend() const { return end(); } 

    /**
     *  @brief Reverse iterator   
     */
    __host__ __device__ reverse_iterator       rbegin()       { return reverse_iterator( end() ); }
    __host__ __device__ const_reverse_iterator rbegin() const { return rbegin(); }

    __host__ __device__ reverse_iterator       rend()        { return reverse_iterator( begin() ); }
    __host__ __device__ const_reverse_iterator rend()  const { return rend(); }

    /**
     *   @brief Test if empty
     */
    bool empty() const { return count == 0; }

    __host__ __device__ reference       operator[] ( size_type n )
    {
        #ifdef __CUDA_ARCH__ 
            return device_store[ n ];
        #else 
            return host_store[ n ];
        #endif 
    }
    __host__ __device__ const_reference operator[] ( size_type n ) const { return operator[] ( n ); }

    __host__ __device__ reference       front()      
    {
        #ifdef __CUDA_ARCH__ 
            return device_store[ 0 ];
        #else 
            return host_store[ 0 ];
        #endif 
    }
    __host__ __device__ const_reference front() const { return front(); } 

    __host__ __device__ reference       back()        
    {
        #ifdef __CUDA_ARCH__ 
            return device_store[ count-1 ];
        #else 
            return host_store[ count-1 ];
        #endif 
    }
    __host__ __device__ const_reference back()  const { return back(); } 

    /**
     *  @brief Direct access to the pointer, analog to std::vector. 
     */ 
    __host__ __device__ T*       data()
    { 
        #ifdef __CUDA_ARCH__ 
            return device_store;
        #else 
            return host_store;
        #endif 
    }
    __host__ __device__ const T* data()  const { return data(); }

    /**
     *  @brief Fill the host and device memory with a certain value. 
     */ 
    void fill( T const value )
    {   
        for( size_t i = 0; i < count; ++i ) host_store[i] = value;
        push(); 
    }
    
    /**
     *  @brief memset the host and device memory with a certain value. 
     */ 
    void memset( char c ) 
    {
        std::memset( host_store, c, sizeof( T ) * count );  
        error_check( cudaMemset( device_store, c, sizeof( T ) * count ), "gpu_vector set" ); 
    }
};

/**
 *  @brief push a std::vector into a gpu_vector
 */ 
template< typename T >
void push( gpu_vector< T >& destination, std::vector< T > const& source )
{
    if ( source.size() != destination.size() )
        destination.resize( source.size() );

    cudaMemcpy( destination.data(), source.data(), sizeof( T ) * source.size(), cudaMemcpyHostToDevice );
    error_check( "push" ); 
}

/**
 *  @brief push a std::vector to the gpu by creating a gpu_vector
 */ 
template< typename T >
gpu_vector< T > push( std::vector< T > const& source )
{
    gpu_vector< T > destination( source.size() );
    push( destination, source );
    return destination;
}

/**
 *  @brief pull a gpu_vector to the gpu by creating a std::vector
 */ 
template< typename T >
void pull( std::vector< T >& destination, gpu_vector< T > const& source )
{
    if ( source.size() != destination.size() )
        destination.resize( source.size() );
   
    cudaMemcpy( destination.data(), source.data(), sizeof( T ) * source.size(), cudaMemcpyDeviceToHost );
}

/**
 *  @brief pull a gpu_vector to the gpu by creating a std::vector
 */ 
template< typename T >
std::vector< T > pull( gpu_vector< T > const& source )
{
    std::vector< T > destination( source.size() );
    pull( destination, source );
    return destination;
}

#endif // __GPU_ARRAYS_HPP 
