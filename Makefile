CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 90

CFITSIO_HOME := $(HOME)/Libraries/cfitsio
CASACORE_HOME ?= /usr

NVCC := $(CUDA_HOME)/bin/nvcc

INC := -I$(CUDA_HOME)/include -I$(CFITSIO_HOME)/include -I$(CASACORE_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -L$(CFITSIO_HOME)/lib -lcudart -lcfitsio -lcufft -lcublas -lcusolver \
       -L$(CASACORE_HOME)/lib/x86_64-linux-gnu -lcasa_tables -lcasa_casa

NVCCFLAGS := -O3 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
             --ptxas-options=-v -Xcompiler -fPIC -Xcompiler -Wextra -lineinfo \
             $(INC) --use_fast_math -Xcompiler -pthread

all: svd_integrated_imager_gpu

svd_integrated_imager_gpu: SVDIntegratedImager.cu
	$(NVCC) -o svd_integrated_imager_gpu $(NVCCFLAGS) SVDIntegratedImager.cu $(LIB)

clean:
	rm -f *.o *.so svd_integrated_imager_gpu
