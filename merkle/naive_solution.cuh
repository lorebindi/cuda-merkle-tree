#include <cstdint>

/*
* Computes and return the total number of nodes required to store a complete Merkle tree
* in a contiguous array representation (heap-like layout), given the number of leaf nodes.
*
* Parameters:
*  - n_leaf: number of leaf nodes in the Merkle tree.
*/
size_t compute_merkle_tree_size(size_t n_leaf);

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
 *  - host_merkle_tree: optional host buffer where the computed tree is copied 
 *                      at the end (used only when MERKLE_TEST is enabled).
 *  - sha256_windowed: selects the SHA256 implementation used during hashing.
 *                     If true, the windowed message schedule version is used;
 *                     otherwise the standard implementation is used.
 */
void build_merkle_tree_naive(size_t n_blocks, uint8_t* host_data_bytes, uint8_t* host_merkle_tree, bool sha256_windowed=true);