#include <cstdint>
#include "merkle_tree.cuh"

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
MerkleTreeGPU* build_merkle_tree_naive(size_t n_blocks, uint8_t* host_data_bytes, bool sha256_windowed=true);