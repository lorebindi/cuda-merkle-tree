
#include <cstdint>
#include <cstdlib> 
#include <cstring>   
#include <cstdio> 

#include "../sha256/sha256_CPU.hpp"
#include "merkle_tree.cuh"

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
void host_compute_parent_hash(uint8_t* parent, uint8_t* left, uint8_t* right, bool sha256_windowed){
    
    uint8_t concatenated[64];
    memcpy(concatenated, left, SHA256_OUTPUT_BLOCK_SIZE);
    memcpy(concatenated+SHA256_OUTPUT_BLOCK_SIZE, right, SHA256_OUTPUT_BLOCK_SIZE);

    sha256_single_block_CPU(concatenated, parent, sha256_windowed);
}

/*
* Builds a full Merkle tree on the CPU starting from a set of input data blocks.
*
* The function allocates a contiguous memory region representing the entire tree
* in a heap-like layout, where:
*  - leaves are placed at the bottom level,
*  - internal nodes are computed bottom-up until the root is reached,
*  - each parent node is the SHA-256 hash of the concatenation of its two children.
*
* The computation proceeds in two phases:
*  1. Leaf computation: each input block is hashed into a leaf node.
*  2. Bottom-up construction: each level is iteratively computed from the previous one,
*     handling odd nodes by duplicating the last child when necessary.
*
* Parameters:
*  - n_blocks: number of input data blocks (leaf nodes)
*  - host_data_bytes: pointer to input data in host memory
*  - sha256_windowed: selects between SHA-256 implementation variants
*
* Returns a pointer to a MerkleTreeCPU structure.
*/
MerkleTreeCPU* host_build_merkle_tree(size_t n_blocks, uint8_t* host_data_bytes, bool sha256_windowed) {

    const size_t tree_size = compute_merkle_tree_size(n_blocks);
    const size_t leaf_offset = tree_size - n_blocks;

    uint8_t* tree = (uint8_t*) malloc(tree_size * SHA256_OUTPUT_BLOCK_SIZE);
    if (!tree) {
        fprintf(stderr, "malloc failed in host_build_merkle_tree\n");
        return NULL;
    }

    // leaf computation
    for (size_t i = 0; i < n_blocks; i++) {
        uint8_t* input = host_data_bytes + i * SHA256_INPUT_BLOCK_SIZE;
        uint8_t* leaf  = tree + (leaf_offset + i) * SHA256_OUTPUT_BLOCK_SIZE;
        sha256_single_block_CPU(input, leaf, sha256_windowed);
    }

    // bottom-up internal node computation
    size_t curr_level_size = n_blocks;
    size_t curr_level_offset = leaf_offset;

    while (curr_level_size > 1) {
        size_t parent_level_size   = (curr_level_size + 1) / 2;
        size_t parent_level_offset = curr_level_offset - parent_level_size;

        for (size_t i = 0; i < parent_level_size; i++) {
            uint8_t* left   = tree + (curr_level_offset + 2*i) * SHA256_OUTPUT_BLOCK_SIZE;
            uint8_t* right  = (2*i+1 < curr_level_size)
                                ? tree + (curr_level_offset + 2*i+1) * SHA256_OUTPUT_BLOCK_SIZE
                                : left; // odd node: duplication of the left node
            uint8_t* parent = tree + (parent_level_offset + i) * SHA256_OUTPUT_BLOCK_SIZE;

            host_compute_parent_hash(parent, left, right, sha256_windowed);
        }

        curr_level_size   = parent_level_size;
        curr_level_offset = parent_level_offset;
    }

    return merkle_tree_cpu_create(tree, tree_size, n_blocks);
}