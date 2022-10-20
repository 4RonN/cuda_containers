cuda_flags = -std=c++17 -ccbin g++ --compiler-options -Wall,-Wextra -arch=sm_61 --expt-extended-lambda -rdc=true

objects= main.o 
all: cuda_flags += -O3 -DNDEBUG
all: $(objects)	
	nvcc $(cuda_flags) $(objects) -o main -lcudart -L$(CUDA)/lib64 

main.o: main.cu gpu_arrays.hpp 
	nvcc $(cuda_flags) main.cu -c -I$(CUDA)/include 

clean: 
	rm -f *.o

