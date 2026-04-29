#include "data_generator.hpp"
#include "merkle_tree.cuh"

/*
* Computes SHA-256 hashes for all proof inputs in a ProofBatch using the GPU.
*
* This function takes a batch of raw proof data stored on the host, transfers
* it to the GPU, computes the SHA-256 hash of each proof in parallel using
* a CUDA kernel, and returns the resulting hashes to the host.
*
* Parameters:
*  - proof_batch: batch containing raw proof inputs to be hashed
*  - n_proofs: number of proofs in the batch
*  - sha256_windowed: selects between standard and windowed SHA-256 variant
*
* Returns:
*  - Pointer to a host-allocated array of size:
*      n_proofs * SHA256_OUTPUT_BLOCK_SIZE
*    containing the computed SHA-256 hashes.
*/
uint8_t* get_hashed_proofs(ProofBatch* proof_batch, size_t n_proofs, bool sha256_windowed);

/*
 * Launches a GPU kernel to verify a batch of Merkle proofs in parallel.
 *
 * Each proof in the batch is checked independently by a separate GPU thread,
 * which reconstructs the Merkle root from a leaf and its authentication path
 * and compares it against the known root stored in the Merkle tree.
 *
 * Parameters:
 *  - proof_batch: batch of Merkle proofs to verify
 *  - merkle_tree_gpu: GPU-resident Merkle tree used as reference
 *  - sha256_windowed: selects SHA-256 implementation variant
 *  - out_elapsed: elapsed merkle proofs verification time.
 *
 * Returns:
 *  - Pointer to an array of boolean values of size n_proofs:
 *    true  -> proof is valid
 *    false -> proof is invalid
 */
bool* compute_merkle_proofs(ProofBatch* proof_batch, MerkleTreeGPU* merkle_tree_gpu, uint64_t* out_elapsed=nullptr);

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
//bool* host_compute_merkle_proofs(ProofBatch* proof_batch, MerkleTreeCPU* merkle_tree);