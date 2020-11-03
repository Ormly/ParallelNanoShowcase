# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NP ?= 1
NVCC=nvcc
MPICXX=mpicxx
MPIRUN ?= mpirun
GPP=mpic++
CUDA_HOME=/usr/local/cuda
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 
GENCODE_FLAGS	:= $(GENCODE_SM60) $(GENCODE_SM70)
ifdef CUB_HOME
        NVCC_FLAGS = -DHAVE_CUB -I$(CUB_HOME)
else
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
endif
NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++11
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -std=c++11
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt
main:   Makefile main.cpp kernel.o
	$(GPP) $(MPICXX_FLAGS) main.cpp kernel.o $(LD_FLAGS) -o output

kernel.o: Makefile kernel.cu
	$(NVCC) kernel.cu -c

