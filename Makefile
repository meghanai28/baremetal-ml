CC = gcc

COMMON_FLAGS = -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L
DEBUG_FLAGS = -O0
RELEASE_FLAGS = -O3 -march=native

NAIVE_SRC = linreg_naive.c
BLAS_SRC = linreg_blas.c

NAIVE_TARGET = linreg_naive
BLAS_TARGET = linreg_blas

BLAS_LIBS = -lopenblas

all: release

debug: debug-naive debug-blas

release: release-naive release-blas

debug-naive:
	$(CC) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(NAIVE_SRC) -o $(NAIVE_TARGET)

debug-blas:
	$(CC) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(BLAS_SRC) -o $(BLAS_TARGET) $(BLAS_LIBS)

release-naive:
	$(CC) $(COMMON_FLAGS) $(RELEASE_FLAGS) $(NAIVE_SRC) -o $(NAIVE_TARGET)

release-blas:
	$(CC) $(COMMON_FLAGS) $(RELEASE_FLAGS) $(BLAS_SRC) -o $(BLAS_TARGET) $(BLAS_LIBS)

run-naive: release-naive
	./$(NAIVE_TARGET)

run-blas: release-blas
	OPENBLAS_NUM_THREADS=1 ./$(BLAS_TARGET)

run: release
	@echo "===== Naive C ====="
	./$(NAIVE_TARGET)
	@echo ""
	@echo "===== C + BLAS ====="
	OPENBLAS_NUM_THREADS=1 ./$(BLAS_TARGET)

clean:
	rm -f $(NAIVE_TARGET) $(BLAS_TARGET)