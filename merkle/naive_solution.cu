
#include<iostream>
#include<stdlib.h>
#include<cstdint>

#include "../sha256/sha256_GPU.cuh"
#include "../data/data_generator.hpp"
#include "naive_solution.cuh"

using namespace std;

#define THREADS_PER_BLOCK 256

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

/*
* Kernel that builds the leaf level of the merkle tree starting from the input data.
*
* Parameters:
*  - n: number of input blocks.
*  - data: pointer to a byte array. Each 64 contiguos positions (i.e bytes) 
*          correspond to a single input block.
*  - merkle_tree: pointer to a byte array. Each 32 contiguos positions (i.e bytes)
*                 correspond to a single node of the Merkle Tree (stored like an heap).
*/
__global__ void leaf_level_build(int n, uint8_t *data, uint8_t *merkle_tree) {
     
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (i >= n) return;

    // Each data block is 64 byte.
    uint8_t* input_data_block = data + (i*SHA256_INPUT_BLOCK_SIZE); 
    // Each node is 32 byte.
    uint8_t* output_hash = merkle_tree + ((n-1+i) * SHA256_OUTPUT_BLOCK_SIZE);
    // Hashing
    sha256_single_block(input_data_block, output_hash, false);
}

/*
* Kernel that builds the intenral level of the merkle tree.
*
* Parameters:
*  - level_size: number of nodes in the parent level.
*  - parents_level: pointer to the byte (sub-)array of the parents node. Each 32 
*                   contiguos positions (i.e bytes) correspond to a single node of the 
*                   Merkle Tree (stored like an heap).
*  - children_level: pointer to the byte (sub-)array of the children node. Each 32 
*                   contiguos positions (i.e bytes) correspond to a single node of the 
*                   Merkle Tree (stored like an heap).
*/
__global__ void internal_level_build(int level_size, uint8_t *parents_level, uint8_t *children_level) {
     
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (i >= level_size) return;

    uint8_t *left = children_level + (2*i)*SHA256_OUTPUT_BLOCK_SIZE;
    uint8_t *right = children_level + (2*i+1)*SHA256_OUTPUT_BLOCK_SIZE;
    uint8_t *parent = parents_level + i*SHA256_OUTPUT_BLOCK_SIZE;

    uint8_t concatenated[64];
    memcpy(concatenated, left, SHA256_OUTPUT_BLOCK_SIZE);
    memcpy(concatenated+SHA256_OUTPUT_BLOCK_SIZE, right, SHA256_OUTPUT_BLOCK_SIZE);

    sha256_single_block(concatenated, parent, false);
}

/* host_merkle_tree used only for testing */
void build_merkle_tree_naive(size_t n_blocks, uint8_t* host_merkle_tree){
    // allocation of the byte array of input data blocks.
    uint8_t *host_data_bytes = generate_random_blocks(n_blocks);
    
    // set the working device
    cudaSetDevice(0); 
    // allocation of GPU arrays
    uint8_t *dev_data_bytes;
    uint8_t *dev_merkle_tree;
    gpuErrchk(cudaMalloc((void**) &dev_merkle_tree, (2*n_blocks-1)*SHA256_OUTPUT_BLOCK_SIZE));
    gpuErrchk(cudaMalloc((void**) &dev_data_bytes, (n_blocks)*SHA256_INPUT_BLOCK_SIZE));

    uint64_t initial_time = current_time_nsecs();

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_data_bytes, host_data_bytes, (n_blocks)*SHA256_INPUT_BLOCK_SIZE, cudaMemcpyHostToDevice));

    // computing the leaf level on GPU
    int blocks_per_grid = (n_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // rouding up
    leaf_level_build<<<blocks_per_grid, THREADS_PER_BLOCK>>>(n_blocks, dev_data_bytes, dev_merkle_tree);

    // deallocate GPU data bytes
    gpuErrchk(cudaFree(dev_data_bytes));

#ifdef MERKLE_TEST
    
    if(host_merkle_tree != nullptr)
        gpuErrchk(cudaMemcpy(host_merkle_tree, dev_merkle_tree,(2*n_blocks-1)*SHA256_OUTPUT_BLOCK_SIZE,
            cudaMemcpyDeviceToHost));

#endif

    // loop to build internal level


    // deallocate GPU merkle_tree
    gpuErrchk(cudaFree(dev_merkle_tree));

}