#include <cstdlib>
#include <cstdint>
#include <random>
#include <cstring>
#include <iostream>
#include "data_generator.hpp"

using namespace std;

#define BLOCK_SIZE 64  // each data block is 64 bytes

/*
 * Allocates a contiguous buffer of size n_blocks * BLOCK_SIZE bytes.
 * Each block is filled with pseudo-random data.
 */
uint8_t* generate_random_blocks(size_t n_blocks) {
    size_t total_size = n_blocks * BLOCK_SIZE;

    uint8_t* buffer = (uint8_t*) malloc(total_size);
    if (!buffer) {
        cerr << "Error allocating memory for data blocks\n";
        exit(EXIT_FAILURE);
    }

    mt19937_64 rng(12345);  // fixed seed for reproducibility
    uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    // Fill 8 bytes at a time
    for (size_t i = 0; i < total_size; i += 8) {
        uint64_t value = dist(rng);
        memcpy(buffer + i, &value, 8);
    }

    return buffer;
}

/*
 * Frees the buffer of blocks
 */
void free_blocks(uint8_t* ptr) {
    if (ptr) free(ptr);
}

/*
 * Generates a batch of n_proofs Merkle proof requests from a pool of 'n_blocks' original data 'blocks'.
 * 
 * Proof requests are generated with monotonically increasing leaf indices to ensure
 * coalesced memory access patterns on the GPU. Each leaf is assigned at least
 * (n_proofs / n_blocks) proof requests; the first (n_proofs % n_blocks) leaves
 * receive one additional request, yielding the index distribution:
 *
 *   [0, 0, 1, 1, ..., extra-1, extra-1, extra, extra+1, ..., n_blocks-1]
 *
 * Each proof request is either:
 *   - VALID:         original block data + correct leaf index
 *   - TAMPERED_DATA: random block data   + correct leaf index (invalid proof)
 *
 * The fraction of tampered requests is controlled by 'tamper_rate' in [0.0, 1.0].
 * The expected[] bitmap records ground truth for result verification.
 */
ProofBatch* generate_proof_requests(const uint8_t* blocks, size_t n_blocks, size_t n_proofs, float tamper_rate){

    ProofBatch* batch = (ProofBatch*) malloc(sizeof(ProofBatch));
    batch->proofs = (uint8_t*) malloc(n_proofs * BLOCK_SIZE * sizeof(uint8_t));
    batch->n_proofs = n_proofs;
    batch->leaf_index = (uint32_t*) malloc(n_proofs * sizeof(uint32_t));
    batch->expected = (bool *) malloc (n_proofs * sizeof(bool));

    if (!batch->proofs || !batch->leaf_index || !batch->expected) {
        cerr << "Error allocating ProofBatch\n";
        exit(EXIT_FAILURE);
    }

    mt19937_64 rng(42);
    uniform_real_distribution<float>   coin(0.0f, 1.0f);
    uniform_int_distribution<uint64_t> random_bytes(0, UINT64_MAX);

    size_t base  = n_proofs / n_blocks; // minimum proofs per leaf.
    size_t extra = n_proofs % n_blocks; // first 'extra' leaves have one proof more.

    size_t proof_i = 0;
    for (size_t leaf = 0; leaf < n_blocks && proof_i < n_proofs; leaf++) {
        size_t count = base + (leaf < extra ? 1 : 0);
        for (size_t k = 0; k < count; k++) {
            uint8_t* dst_proof = batch->proofs + proof_i * BLOCK_SIZE;
            batch->leaf_index[proof_i] = (uint32_t) leaf;

            if (coin(rng) < tamper_rate) {
                // TAMPERED_DATA: random block, correct index
                for (size_t b = 0; b < BLOCK_SIZE; b += 8) {
                    uint64_t val = random_bytes(rng);
                    memcpy(dst_proof + b, &val, 8);
                }
                batch->expected[proof_i] = false;
            } else {
                // VALID: original block, correct index.
                memcpy(dst_proof, blocks + leaf * BLOCK_SIZE, BLOCK_SIZE);
                batch->expected[proof_i] = true;
            }
            proof_i++;
        }
    }

    return batch;
}

/* Free ProofBatch */
void free_proof_batch(ProofBatch* batch) {
    if (!batch) return;
    if (batch->proofs)       free(batch->proofs);
    if (batch->leaf_index) free(batch->leaf_index);
    if (batch->expected)   free(batch->expected);
    free(batch);
}