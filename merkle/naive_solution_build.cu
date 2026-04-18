
#include<iostream>
#include<stdlib.h>
#include<cstdint>

#include "../sha256/sha256_GPU.cuh"
#include "../data/data_generator.hpp"
#include "utils.cuh"
#include "merkle_tree.cuh"
#include "naive_solution_build.cuh"

using namespace std;

#define THREADS_PER_BLOCK 256

/*
* Kernel that builds a single internal level of a merkle tree stored in GMEM in heap-style.
*
* The kernel build a full tree level by computing parent hashes in parallel,
* where each thread processes one pair of child nodes and writes a single parent node.
*
* Example: 8 leaves, 2 levels per block, 4 blocks (THREADS_PER_BLOCK = 2)
*
* Heap layout:
*  Level 0:                         [n0]                                
*  Level 1:               [n1]                 [n2]             
*  Level 2:         *[n3]       [n4]      [n5]       [n6]      <- parents_level points at * (beginning of this level)
*  Level 3:       *[n7][n8]  [n9][n10] [n11][n12] [n13][n14]    <- children_level points at * (beginning of this level)
*  
*
* Block 0 thread 0: reads [n7][n8] from GMEM, computes [n3] and store it in GMEM. 
* Block 0 thread 1: reads [n9][n10] from GMEM, computes [n4] and store it in GMEM.
* Block 0 thread 2: ....
*
* After the invocation of this kernel, a new invocation can treat parents_level as the new
* children_levels and so on up to the root.
*
*
* Parameters:
*  - parent_level_size: number of nodes in the parent level.
*  - parents_level: pointer to the byte (sub-)array of the parents node. Each 32 
*                   contiguos positions (i.e bytes) correspond to a single node of the 
*                   Merkle Tree (stored like an heap).
*  - children_level_size: number of nodes in the children level.
*  - children_level: pointer to the byte (sub-)array of the children node. Each 32 
*                   contiguos positions (i.e bytes) correspond to a single node of the 
*                   Merkle Tree (stored like an heap).
*
* Notes:
*  - When a level contains an odd number of nodes, the last node is hashed
*    with itself.
*/
__global__ void internal_level_build_naive(int parent_level_size, uint8_t *parents_level, int children_level_size, uint8_t *children_level, bool sha256_windowed) {
     
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (i >= parent_level_size) return;

    uint8_t *left = children_level + (2*i)*SHA256_OUTPUT_BLOCK_SIZE;
    uint8_t *right = children_level + (2*i+1)*SHA256_OUTPUT_BLOCK_SIZE;
    uint8_t *parent = parents_level + i*SHA256_OUTPUT_BLOCK_SIZE;

    // Odd node management: the last node is duplicated
    bool is_last_odd = (children_level_size % 2 == 1) && (i == parent_level_size - 1);
    if (is_last_odd) right = left;

    compute_parent_hash(parent, left, right, sha256_windowed);

}

/*
 * Naive GPU implementation of Merkle tree construction.
 *
 * This function builds a Merkle tree starting from a set of input data blocks.
 * Each block is first hashed (SHA256) to generate the leaf level of the tree, then the
 * internal levels are iteratively computed on the GPU until the root is produced.
 *
 * The tree is stored in a contiguous array using a heap-like layout, where
 * leaves are placed at the end of the array and internal nodes are computed
 * level-by-level moving upward toward the root.
 *
 * Parameters:
 *  - n_blocks: number of input data blocks (i.e., number of leaves).
 *  - host_data_bytes: pointer to the host array containing the input blocks.
 *  - sha256_windowed: selects the SHA256 implementation used during hashing.
 *                     If true, the windowed message schedule version is used;
 *                     otherwise the standard implementation is used.
 *
 * Returns:
 *  - A pointer to a MerkleTreeGPU structure representing the tree stored in GPU memory.
 *    The structure contains the device pointer to the tree and its metadata. The caller
 *    is responsible for freeing it.
 */
MerkleTreeGPU* build_merkle_tree_naive(size_t n_blocks, uint8_t* host_data_bytes, bool sha256_windowed){
 
    //computing the merkle_tree dimension
    const size_t merkle_tree_size = compute_merkle_tree_size(n_blocks);
    size_t leaf_offset = merkle_tree_size - n_blocks;
  
    // set the working device
    cudaSetDevice(0); 
    // allocation of GPU arrays
    uint8_t *dev_data_bytes;
    uint8_t *dev_merkle_tree;
    gpuErrchk(cudaMalloc((void**) &dev_merkle_tree, (merkle_tree_size)*SHA256_OUTPUT_BLOCK_SIZE));
    gpuErrchk(cudaMalloc((void**) &dev_data_bytes, (n_blocks)*SHA256_INPUT_BLOCK_SIZE));

    uint64_t initial_time = current_time_nsecs();

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_data_bytes, host_data_bytes, (n_blocks)*SHA256_INPUT_BLOCK_SIZE, cudaMemcpyHostToDevice));

    // computing the leaf level on GPU
    int blocks_per_grid = (n_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // rouding up
    leaf_level_build<<<blocks_per_grid, THREADS_PER_BLOCK>>>(n_blocks, leaf_offset, dev_data_bytes, dev_merkle_tree, sha256_windowed);
    cudaDeviceSynchronize();

    // deallocate GPU data bytes
    gpuErrchk(cudaFree(dev_data_bytes));

    int children_level_size = n_blocks;
    size_t children_offset = merkle_tree_size - n_blocks;

    // loop to build internal level
    while(children_level_size > 1){
        int parent_level_size = (children_level_size + 1) /2;
        size_t parent_offset = children_offset - parent_level_size;

        uint8_t *dev_curr_children_level = dev_merkle_tree + (children_offset * SHA256_OUTPUT_BLOCK_SIZE);
        uint8_t *dev_curr_parent_level = dev_merkle_tree + (parent_offset * SHA256_OUTPUT_BLOCK_SIZE);
        // computing internal level 
        int blocks_per_grid = (parent_level_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // rouding up
        internal_level_build_naive<<<blocks_per_grid, THREADS_PER_BLOCK>>>(parent_level_size, dev_curr_parent_level, children_level_size, dev_curr_children_level, sha256_windowed);
        cudaDeviceSynchronize();
        
        // parents becomes children in the next iteration
        children_level_size = parent_level_size;
        children_offset = parent_offset;
    }

    // return the result
    MerkleTreeGPU* result = merkle_tree_gpu_create(dev_merkle_tree, merkle_tree_size, n_blocks);
    return result;

}