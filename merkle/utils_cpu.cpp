#include <cstdlib> 
#include <cstdint>
#include "utils_cpu.hpp"
#include "../data/data_generator.hpp"

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
uint8_t* get_host_hashed_proofs(ProofBatch* proof_batch, size_t n_proofs, bool sha256_windowed){
    // allocate the return value
    uint8_t* hashed_proofs = (uint8_t*) malloc (sizeof(uint8_t)* SHA256_OUTPUT_BLOCK_SIZE * n_proofs);

    for(size_t i=0; i<n_proofs; i++){
        sha256_single_block_CPU(proof_batch->proofs + i * SHA256_INPUT_BLOCK_SIZE, hashed_proofs + i*SHA256_OUTPUT_BLOCK_SIZE, sha256_windowed);
    }

    return hashed_proofs;
}