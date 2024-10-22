INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart -lcfitsio -lcufft
GCC := g++
NVCC := ${CUDA_HOME}/bin/nvcc

GCC_OPTS :=-O3 -fPIC -Wall -Wextra $(INC) -std=c++11
NVCCFLAGS :=-O3 -gencode arch=compute_90,code=sm_90 --ptxas-options=-v -Xcompiler -fPIC -Xcompiler -Wextra -lineinfo $(INC) --use_fast_math

all: clean sharedlibrary_gpu

sharedlibrary_gpu: pipe_interface.o FIPkernels.o
	$(NVCC) -o sharedlibrary_gpu $(NVCCFLAGS) pipe_interface.o FIPkernels.o $(LIB)

pipe_interface.o: pipe_interface.cpp
	$(GCC) -c pipe_interface.cpp $(GCC_OPTS) -o pipe_interface.o

FIPkernels.o: FIPkernels.cu
	$(NVCC) -c FIPkernels.cu  $(NVCCFLAGS) -o FIPkernels.o

clean:	
	rm -f *.o *.so
