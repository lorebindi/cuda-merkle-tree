#include <cstdint>
#include "merkle_tree.cuh"
#include "../data/data_generator.hpp"

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
MerkleTreeCPU* host_build_merkle_tree(size_t n_blocks, uint8_t* host_data_bytes, bool sha256_windowed);

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
 *
 * Returns:
 *  - Array of boolean values indicating whether each proof is valid.
 *    The caller is responsible for freeing the returned memory.
 */
bool* host_compute_merkle_proofs(ProofBatch* proof_batch, MerkleTreeCPU* merkle_tree, bool sha256_windowed);