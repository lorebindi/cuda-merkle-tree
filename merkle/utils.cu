#include<iostream>
#include<stdlib.h>
#include<cstdint>
#include "utils.cuh"

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
size_t compute_merkle_tree_size(size_t n_leaf){
    size_t size = 0;
    while (n_leaf > 1) {     
        size += n_leaf;
        n_leaf = (n_leaf + 1) / 2;
    }
    size += 1;
    return size;
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
__global__ void leaf_level_build(int n, size_t leaf_offset, uint8_t *data, uint8_t *merkle_tree, bool sha256_windowed) {
     
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (i >= n) return;

    // Each data block is 64 byte.
    uint8_t* input_data_block = data + (i*SHA256_INPUT_BLOCK_SIZE); 
    // Each node is 32 byte.
    uint8_t* output_hash = merkle_tree + ((leaf_offset + i) * SHA256_OUTPUT_BLOCK_SIZE);
    // Hashing
    sha256_single_block(input_data_block, output_hash, sha256_windowed);
}