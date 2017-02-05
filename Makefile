# on Edision we will benchmark you against the default vendor-tuned BLAS. The compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. To change compilers, you need to type something like: "module swap PrgEnv-pgi PrgEnv-gnu" See the NERSC documentation for available compilers.

# for running on NERSC
#CC = cc 
#OPT = -fast -no-ipo -qopt-report=3 -qopt-report-phase=vec -qopt-report-file=vectReport.txt
#CFLAGS = -Wall -std=gnu99 $(OPT)
#LDFLAGS = -Wall
# librt is needed for clock_gettime
#LDLIBS = -lrt   #for NERSC

# for running on MacOS
CC = gcc 
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
LDFLAGS = -Wall
LDLIBS = -framework Accelerate  #for local MacOS

targets = benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o  

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
test-block-sizes : test-block-sizes.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
