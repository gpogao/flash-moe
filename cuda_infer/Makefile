NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90 -std=c++17 -Xcompiler -Wall -Xcompiler -fPIC
CUDA_LIBS = -lcublas -lcudart

all: infer libkernels.so

tokenizer.o: tokenizer.c tokenizer.h
	gcc -c -O3 -std=c11 -fPIC -o $@ tokenizer.c

libkernels.so: kernels.cu kernels.h
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ kernels.cu $(CUDA_LIBS)

infer: infer.cu kernels.cu tokenizer.o
	$(NVCC) $(NVCC_FLAGS) -o $@ infer.cu kernels.cu tokenizer.o $(CUDA_LIBS)

clean:
	rm -f infer *.o *.so

.PHONY: all clean