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
* If a level contains an odd number of nodes, the last parent node will have
* only one child (i.e., the right child is missing). In typical Merkle tree
* constructions this node is later hashed with itself when computing the parent.
*
* Parameters:
*  - n_leaf: number of leaf nodes in the Merkle tree.
*/
__host__ __device__ __forceinline__ size_t compute_merkle_tree_size(size_t n_leaf){
    size_t size = 0;
    while (n_leaf > 1) {     
        size += n_leaf;
        n_leaf = (n_leaf + 1) / 2;
    }
    size += 1;
    return size;
}

/*
 * Computes the hash of a parent node in a Merkle tree.
 *
 * The function concatenates the left and right child hashes (each of size
 * SHA256_OUTPUT_BLOCK_SIZE bytes) into a temporary buffer and applies a
 * single-block SHA-256 hash. The resulting hash is written to the parent node.
 *
 * Parameters:
 * - parent: pointer to the output buffer where the computed hash is stored
 * - left: pointer to the left child hash
 * - right: pointer to the right child hash
 * - sha256_windowed: flag to select the SHA-256 implementation variant
 *
 * Note:
 * - The caller must ensure that both left and right point to valid memory.
 * - Handling of odd nodes (e.g., duplicating the left child) must be done
 *   outside this function.
 */
__device__ __forceinline__ void compute_parent_hash(uint8_t* parent, uint8_t* left, uint8_t* right, bool sha256_windowed){
    
    uint8_t concatenated[64];
    memcpy(concatenated, left, SHA256_OUTPUT_BLOCK_SIZE);
    memcpy(concatenated+SHA256_OUTPUT_BLOCK_SIZE, right, SHA256_OUTPUT_BLOCK_SIZE);

    sha256_single_block(concatenated, parent, sha256_windowed);
}

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