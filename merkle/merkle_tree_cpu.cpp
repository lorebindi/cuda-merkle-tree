
#include <cstdint>
#include <cstdlib> 
#include <cstring>   
#include <cstdio> 
#include <time.h>
#include <inttypes.h>
#include <stdio.h>

#include "../sha256/sha256_CPU.hpp"
#include "../data/data_generator.hpp"
#include "merkle_tree.cuh"
#include "utils_cpu.hpp"


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
*  - out_elapsed: elapsed merkle tree building time.
*
* Returns a pointer to a MerkleTreeCPU structure.
*/
MerkleTreeCPU* host_build_merkle_tree_serial(size_t n_blocks, uint8_t* host_data_bytes, bool sha256_windowed, uint64_t* out_elapsed = nullptr) {

    const size_t tree_size = host_compute_merkle_tree_size(n_blocks);
    const size_t leaf_offset = tree_size - n_blocks;

    uint8_t* tree = (uint8_t*) malloc(tree_size * SHA256_OUTPUT_BLOCK_SIZE);
    if (!tree) {
        fprintf(stderr, "malloc failed in host_build_merkle_tree\n");
        return NULL;
    }

    uint64_t initial_time = current_time_nsecs();

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

    uint64_t end_time = current_time_nsecs();
    if(out_elapsed)
        *out_elapsed = end_time - initial_time;

    return merkle_tree_cpu_create(tree, tree_size, n_blocks);
}

/*
 * Verifies a batch of Merkle proofs on the CPU against a precomputed Merkle tree.
 *
 * For each proof, the function:
 *  - hashes the raw leaf data into a 32-byte SHA-256 digest,
 *  - iteratively reconstructs parent hashes up to the root,
 *  - compares each computed parent node against the corresponding node in the tree.
 *
 * The tree is assumed to be stored in heap-style layout (array-based binary tree),
 * where each node is a 32-byte SHA-256 hash.
 *
 * Parameters:
 *  - proof_batch: batch of raw Merkle proofs and associated leaf indices
 *  - merkle_tree: CPU representation of the full Merkle tree
 *  - sha256_windowed: selects SHA-256 variant (windowed or standard)
 *  - out_elapsed: elapsed merkle proof verification time.
 *
 * Returns:
 *  - Array of boolean values indicating whether each proof is valid.
 *    The caller is responsible for freeing the returned memory.
 */
bool* host_compute_merkle_proofs_serial(ProofBatch* proof_batch, MerkleTreeCPU* merkle_tree, bool sha256_windowed, uint64_t* out_elapsed){
    
    size_t n_proofs = proof_batch->n_proofs;
    // merkle proof hashing
    uint8_t* hashed_proofs = get_host_hashed_proofs(proof_batch, n_proofs, sha256_windowed);
    // allocating results
    bool* results = (bool*) malloc(sizeof(bool) * n_proofs);

    uint64_t initial_time = current_time_nsecs();

    for (size_t i = 0; i < n_proofs; i++) {

        int curr_lev_size   = merkle_tree->base.n_leaves;
        // offset of the first node in the current level
        int curr_lev_offset = merkle_tree->base.size - merkle_tree->base.n_leaves;
        // computing the offset of the current node
        int curr_node_offset = curr_lev_offset + proof_batch->leaf_index[i];

        // computing the sibling offset of the current node
        int index_in_level = curr_node_offset - curr_lev_offset;
        int sibling_offset_step = (index_in_level & 1) ? -1 : +1;
        int sibling_node_offset = curr_node_offset + sibling_offset_step;

        bool has_sibling = (sibling_node_offset >= curr_lev_offset) &&
                           (sibling_node_offset < curr_lev_offset + merkle_tree->base.n_leaves);

        if (!has_sibling) {
            sibling_node_offset = curr_node_offset;
        }

        // pointers to the current node and it's sibling
        uint8_t* curr_node = hashed_proofs + i * SHA256_OUTPUT_BLOCK_SIZE;
        uint8_t* sibling_node = merkle_tree->host_tree + sibling_node_offset * SHA256_OUTPUT_BLOCK_SIZE;
        
        uint8_t temp_parent_node[SHA256_OUTPUT_BLOCK_SIZE];

        uint8_t *left, *right;
        int parent_lev_offset = 0, parent_offset = 0;

        results[i] = true;

        // check for a merkle tree with only the root
        if (merkle_tree->base.depth == 0) {
            results[i] = (memcmp(curr_node, merkle_tree->host_tree, SHA256_OUTPUT_BLOCK_SIZE) == 0);
            continue;
        }

        for (uint32_t lev = 0; lev < merkle_tree->base.depth; lev++) {

            // computing and storing the hash of the temporary parent node
            left  = (sibling_offset_step == +1) ? curr_node : sibling_node;
            right = (sibling_offset_step == +1) ? sibling_node : curr_node;
            host_compute_parent_hash(temp_parent_node, left, right, sha256_windowed);
            // computing parent offset 
            parent_lev_offset = curr_lev_offset - (curr_lev_size + 1) / 2;
            parent_offset = parent_lev_offset +
                           ((curr_node_offset - curr_lev_offset) >> 1);

            // comparing the just-computed parent hash with the one in the merkle tree               
            if (results[i] &&
                memcmp(temp_parent_node,
                       merkle_tree->host_tree + parent_offset * SHA256_OUTPUT_BLOCK_SIZE,
                       SHA256_OUTPUT_BLOCK_SIZE) != 0)
            {
                results[i] = false;
                break;
            }

            // updating offsets for the upper level
            curr_lev_size   = (curr_lev_size + 1) / 2; // computing the upper level size.
            curr_lev_offset = parent_lev_offset; // parent lev offset became curr lev offset.
            curr_node       = temp_parent_node; // the temporary parent node became the curr node.
            curr_node_offset = parent_offset;

            // computing sibling for the upper level
            index_in_level = curr_node_offset - curr_lev_offset;
            sibling_offset_step = (index_in_level & 1) ? -1 : +1;
            sibling_node_offset = curr_node_offset + sibling_offset_step;
            // odd management. 
            has_sibling = (sibling_node_offset >= curr_lev_offset) &&
                          (sibling_node_offset < curr_lev_offset + curr_lev_size);

            if (!has_sibling) {
                sibling_node_offset = curr_node_offset;
            }
            sibling_node = merkle_tree->host_tree + sibling_node_offset * SHA256_OUTPUT_BLOCK_SIZE;
        }
    }

    uint64_t end_time = current_time_nsecs();
    if(out_elapsed)
        *out_elapsed = end_time - initial_time;

    free(hashed_proofs);
    return results;
}