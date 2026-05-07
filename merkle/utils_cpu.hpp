#include <cstdint>
#include <time.h>
#include <cstring>
#include "../sha256/sha256_CPU.hpp"
#include "../data/data_generator.hpp"

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
*  - n_leaf: number of leaf nodes in the Merkle tree.*/

inline size_t host_compute_merkle_tree_size(size_t n_leaf){
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
inline void host_compute_parent_hash(uint8_t* parent, uint8_t* left, uint8_t* right, bool sha256_windowed){
    
    uint8_t concatenated[64];
    memcpy(concatenated, left, SHA256_OUTPUT_BLOCK_SIZE);
    memcpy(concatenated+SHA256_OUTPUT_BLOCK_SIZE, right, SHA256_OUTPUT_BLOCK_SIZE);

    sha256_single_block_CPU(concatenated, parent, sha256_windowed);
}

/*
 * Computes the SHA-256 hash for a batch of Merkle proof inputs on the CPU.
 *
 * Each proof is assumed to be a single 64-byte block (SHA256_INPUT_BLOCK_SIZE).
 * The function applies the SHA-256 compression (windowed or standard) to each
 * block and stores the resulting 32-byte hash (SHA256_OUTPUT_BLOCK_SIZE)
 * in a contiguous output array.
 *
 * Parameters:
 *  - proof_batch: pointer to the structure containing raw proof data.
 *  - n_proofs: number of proofs to hash.
 *  - sha256_windowed: selects the SHA-256 variant (true = windowed, false = standard).
 *
 * Returns:
 *  - Pointer to a newly allocated array containing the hashed proofs.
 *    The caller is responsible for freeing this memory.
 */
uint8_t* get_host_hashed_proofs(ProofBatch* proof_batch, size_t n_proofs, bool sha256_windowed);