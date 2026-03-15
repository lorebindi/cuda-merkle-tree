/* 
* This file contains all helper functions and kernels used by the solutions in this folder.
*/

#include <cstdint>
#include "../sha256/sha256_GPU.cuh"

/*
 * Error checking helper for CUDA calls.
 *
 * gpuAssert:
 *   - Checks a CUDA error code and prints an error message with file and line if it failed.
 *   - Terminates execution on error.
 *
 * gpuErrchk:
 *   - Macro wrapper that passes __FILE__ and __LINE__ automatically.
 *   - Example usage: gpuErrchk(cudaMalloc(&ptr, size));
 */
inline void gpuAssert(cudaError_t code, const char *file, int line){
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* 
* Returns the current time in nanoseconds since the Unix epoch.
*/
inline uint64_t current_time_nsecs(){
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

/*
* Computes and return the total number of nodes required to store a complete Merkle tree
* in a contiguous array representation (heap-like layout), given the number of leaf nodes.
*
* Parameters:
*  - n_leaf: number of leaf nodes in the Merkle tree.
*/
size_t compute_merkle_tree_size(size_t n_leaf);

/*
* Kernel that builds the leaf level of the merkle tree starting from the input data.
*
* Parameters:
*  - n: number of input blocks.
*  - size_t leaf_offset:
*  - data: pointer to a byte array. Each 64 contiguos positions (i.e bytes) 
*          correspond to a single input block.
*  - merkle_tree: pointer to a byte array. Each 32 contiguos positions (i.e bytes)
*                 correspond to a single node of the Merkle Tree (stored like an heap).
*/
__global__ void leaf_level_build(int n, size_t leaf_offset, uint8_t *data, uint8_t *merkle_tree, bool sha256_windowed);