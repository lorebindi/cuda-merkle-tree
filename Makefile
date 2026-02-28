NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -Isha256
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80
RDCFLAGS        = -rdc=true

# ---------------- Targets ----------------
TARGETS         = main tests_sha256

# ---------------- Source files ----------------
SRC_MAIN        = main.cu sha256/sha256.cu
OBJ_MAIN        = main.o sha256/sha256.o

SRC_TEST        = tests/tests_sha256.cu sha256/sha256.cu
OBJ_TEST        = tests/tests_sha256.o sha256/sha256.o

# ---------------- Default ----------------
.DEFAULT_GOAL := all
.PHONY: all clean

all: $(TARGETS)

# ---------------- Build main executable ----------------
main: $(OBJ_MAIN)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $^ -o $@

# ---------------- Build test executable ----------------
tests_sha256: $(OBJ_TEST)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) $^ -o $@

# ---------------- Compile .cu -> .o ----------------
%.o: %.cu
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(RDCFLAGS) -dc $< -o $@

# ---------------- Clean ----------------
clean:
	rm -f main.o sha256/sha256.o test_sha256.o $(TARGETS)