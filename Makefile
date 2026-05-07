# ---------------- Compiler and flags ----------------
NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -Isha256 -Idata -Imerkle
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80
RDCFLAGS        = -rdc=true
OMPFLAGS        = -Xcompiler -fopenmp
#DEFINES_TEST    = -DMERKLE_TEST

# ---------------- Source files ----------------
SRC_MAIN_CU     = main.cu sha256/sha256.cu merkle/naive_solution.cu
SRC_MAIN_CPP    = data/data_generator.cpp sha256/sha256_CPU.cpp

SRC_SHA256_TEST_CU  = tests/sha256_tests.cu \
					  sha256/sha256_GPU.cu
SRC_SHA256_TEST_CPP = data/data_generator.cpp \
					  sha256/sha256_CPU.cpp

SRC_MERKLE_TEST_CU  = tests/merkle_tests.cu \
					  merkle/naive_solution_build.cu \
					  merkle/shared_mem_solution_build.cu \
					  merkle/merkle_proof.cu \
					  sha256/sha256_GPU.cu

SRC_MERKLE_TEST_CPP = data/data_generator.cpp \
					  sha256/sha256_CPU.cpp \
					  merkle/merkle_tree_cpu.cpp \
					  merkle/merkle_tree_cpu_openmp.cpp \
					  merkle/utils_cpu.cpp

SRC_SHA256_BENCH_CU  = benchmarks/sha256_benchmarks.cu \
					   sha256/sha256_GPU.cu

SRC_SHA256_BENCH_CPP = data/data_generator.cpp \
					   sha256/sha256_CPU.cpp

SRC_MERKLE_BENCH_CU  = benchmarks/merkle_benchmarks.cu \
					   sha256/sha256_GPU.cu \
					   merkle/naive_solution_build.cu \
					   merkle/shared_mem_solution_build.cu \
					   merkle/merkle_proof.cu

SRC_MERKLE_BENCH_CPP = data/data_generator.cpp \
					   sha256/sha256_CPU.cpp \
					   merkle/merkle_tree_cpu.cpp \
					   merkle/merkle_tree_cpu_openmp.cpp \
					   merkle/utils_cpu.cpp

OBJ_MAIN_CU          = $(SRC_MAIN_CU:.cu=.o)
OBJ_MAIN_CPP         = $(SRC_MAIN_CPP:.cpp=.o)
OBJ_SHA256_TEST_CU   = $(SRC_SHA256_TEST_CU:.cu=.o)
OBJ_SHA256_TEST_CPP  = $(SRC_SHA256_TEST_CPP:.cpp=.o)
OBJ_MERKLE_TEST_CU   = $(SRC_MERKLE_TEST_CU:.cu=.o)
OBJ_MERKLE_TEST_CPP  = $(SRC_MERKLE_TEST_CPP:.cpp=.o)
OBJ_SHA256_BENCH_CU  = $(SRC_SHA256_BENCH_CU:.cu=.o)
OBJ_SHA256_BENCH_CPP = $(SRC_SHA256_BENCH_CPP:.cpp=.o)
OBJ_MERKLE_BENCH_CU  = $(SRC_MERKLE_BENCH_CU:.cu=.o)
OBJ_MERKLE_BENCH_CPP = $(SRC_MERKLE_BENCH_CPP:.cpp=.o)

# ---------------- Targets ----------------
TARGET_MAIN         = main
TARGET_SHA256_TEST  = sha256_tests
TARGET_MERKLE_TEST  = merkle_tests
TARGET_SHA256_BENCH = sha256_benchmarks
TARGET_MERKLE_BENCH = merkle_benchmarks

# ---------------- Default ----------------
.DEFAULT_GOAL := all
.PHONY: all clean test bench

all: $(TARGET_MAIN)

# ---------------- Build main executable ----------------

$(TARGET_MAIN): $(OBJ_MAIN_CU) $(OBJ_MAIN_CPP)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(OMPFLAGS) $^ -o $@

# ---------------- Build test executables ----------------

$(TARGET_SHA256_TEST): $(OBJ_SHA256_TEST_CU) $(OBJ_SHA256_TEST_CPP)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(OMPFLAGS) $(DEFINES_TEST) $^ -o $@

$(TARGET_MERKLE_TEST): $(OBJ_MERKLE_TEST_CU) $(OBJ_MERKLE_TEST_CPP)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(OMPFLAGS) $(DEFINES_TEST) $^ -o $@

# ---------------- Build benchmark executables ----------------

$(TARGET_SHA256_BENCH): $(OBJ_SHA256_BENCH_CU) $(OBJ_SHA256_BENCH_CPP)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(OMPFLAGS) $^ -o $@

$(TARGET_MERKLE_BENCH): $(OBJ_MERKLE_BENCH_CU) $(OBJ_MERKLE_BENCH_CPP)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(OMPFLAGS) $^ -o $@

# ---------------- Compile rules ----------------

%.o: %.cu
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(OMPFLAGS) $(DEFINES_TEST) -dc $< -o $@

%.o: %.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(OMPFLAGS) $(DEFINES_TEST) -c $< -o $@

# ---------------- Target test ----------------

test: $(TARGET_SHA256_TEST) $(TARGET_MERKLE_TEST)

# ---------------- Target bench ----------------

bench: $(TARGET_SHA256_BENCH) $(TARGET_MERKLE_BENCH)

# ---------------- Clean ----------------

clean:
	rm -f \
	$(OBJ_MAIN_CU) $(OBJ_MAIN_CPP) \
	$(OBJ_SHA256_TEST_CU) $(OBJ_SHA256_TEST_CPP) \
	$(OBJ_MERKLE_TEST_CU) $(OBJ_MERKLE_TEST_CPP) \
	$(OBJ_SHA256_BENCH_CU) $(OBJ_SHA256_BENCH_CPP) \
	$(OBJ_MERKLE_BENCH_CU) $(OBJ_MERKLE_BENCH_CPP) \
	$(TARGET_MAIN) \
	$(TARGET_SHA256_TEST) \
	$(TARGET_MERKLE_TEST) \
	$(TARGET_SHA256_BENCH) \
	$(TARGET_MERKLE_BENCH)