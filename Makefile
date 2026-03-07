# ---------------- Compiler and flags ----------------
NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -Isha256 -Idata -Imerkle
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80
RDCFLAGS        = -rdc=true
DEFINES_TEST    = -DMERKLE_TEST

# ---------------- Source files ----------------
SRC_MAIN_CU     = main.cu sha256/sha256.cu merkle/naive_solution.cu
SRC_MAIN_CPP    = data/data_generator.cpp sha256/sha256_CPU.cpp

SRC_TEST_CU     = tests/test_runner.cu sha256/sha256_GPU.cu merkle/naive_solution.cu
SRC_TEST_CPP    = data/data_generator.cpp sha256/sha256_CPU.cpp

OBJ_MAIN_CU     = $(SRC_MAIN_CU:.cu=.o)
OBJ_MAIN_CPP    = $(SRC_MAIN_CPP:.cpp=.o)
OBJ_TEST_CU     = $(SRC_TEST_CU:.cu=.o)
OBJ_TEST_CPP    = $(SRC_TEST_CPP:.cpp=.o)

# ---------------- Targets ----------------
TARGET_MAIN     = main
TARGET_TEST     = test_runner

# ---------------- Default ----------------
.DEFAULT_GOAL := all
.PHONY: all clean test

all: $(TARGET_MAIN) $(TARGET_TEST)

# ---------------- Build main executable ----------------
$(TARGET_MAIN): $(OBJ_MAIN_CU) $(OBJ_MAIN_CPP)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $^ -o $@

# ---------------- Build test executable ----------------
$(TARGET_TEST): $(OBJ_TEST_CU) $(OBJ_TEST_CPP)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(DEFINES_TEST) $^ -o $@

# ---------------- Compile rules ----------------
%.o: %.cu
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $(DEFINES_TEST) -dc $< -o $@

%.o: %.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(DEFINES_TEST) -c $< -o $@

# ---------------- Target test ----------------
test: $(TARGET_TEST)

# ---------------- Clean ----------------
clean:
	rm -f $(OBJ_MAIN_CU) $(OBJ_MAIN_CPP) $(OBJ_TEST_CU) $(OBJ_TEST_CPP) $(TARGET_MAIN) $(TARGET_TEST)