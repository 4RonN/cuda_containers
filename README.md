# cuda_containers
Template container classes for GPU memory in c++/CUDA.

The class gpu_vector holds a dynamically allocated block of memory, 
which can be trasferred from a std::vector via convenient push() and pull() functions.
```
    std::vector< int > x_host   = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    gpu_vector< int >  x_device = push( x_host );
    
    // do stuff on the GPU... 

    pull( x_host, x_device );
    
    // do stuff on the CPU...    
```
The second class async_vector combines two equally sized memory blocks in host and device memory, and integrate the push/pull behavior. 
In this case page locked/pinned memory is used to optimize the transfer between host and device.
```
    async_array< int > x = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    x.push();
  
    // do stuff on the GPU... 

    x.pull();
    
    // do stuff on the CPU...    
```
As argument to a __global__ kernel function, 
the container's copy constructers do no allocate new memory, but instead increment a usage counter.
Only when the last copy is destructed, the memory is freed. 
  
See main.cu for a miminal example. The makefile contians compilation instructions. 
