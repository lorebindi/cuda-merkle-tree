#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
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
 * Builds the per-leaf proof count array according to 'dist'.
 *
 * The returned vector has size n_blocks and sums exactly to n_proofs.
 * For DIST_ZIPF the counts follow a power-law with exponent 's';
 * the largest-remainder method ensures the total is exact.
 * Leaf popularity is randomised via shuffle so it does not correlate
 * with leaf index.
 */
static vector<size_t> build_leaf_counts(size_t n_blocks, size_t n_proofs, ProofDistribution dist, double s, mt19937_64& rng){
    
    vector<size_t> counts(n_blocks, 0);

    switch (dist) {

        case DIST_UNIFORM: {
            size_t base  = n_proofs / n_blocks;
            size_t extra = n_proofs % n_blocks;
            for (size_t i = 0; i < n_blocks; i++)
                counts[i] = base + (i < extra ? 1 : 0);
            break;
        }

        case DIST_ZIPF: {
            // computing weights w[i] = 1/(i+1)^s
            vector<double> weights(n_blocks);
            double total = 0.0;
            for (size_t i = 0; i < n_blocks; i++) {
                weights[i] = 1.0 / pow(static_cast<double>(i + 1), s);
                total += weights[i];
            }

            // weights shuffle: popularity assigned to casual leaves, it's not in relation with the leaf index
            shuffle(weights.begin(), weights.end(), rng);

            // set to 0 the leaf under the treshold, they will receive 0 proofs
            const double threshold_factor = 0.5;
            double mean_weight = total / static_cast<double>(n_blocks);
            double active_total = 0.0;
            for (size_t i = 0; i < n_blocks; i++) {
                if (weights[i] < mean_weight * threshold_factor)
                    weights[i] = 0.0;
                else
                    active_total += weights[i];
            }

            // guarantes that sum(counts) == n_proofs exactly
            vector<double> frac(n_blocks);
            size_t assigned = 0;
            for (size_t i = 0; i < n_blocks; i++) {
                if (weights[i] == 0.0) continue;
                // ideal number of proofs for leaf i.
                double exact = weights[i] / total * static_cast<double>(n_proofs);
                // truncking to integer part
                counts[i] = static_cast<size_t>(exact);
                // storing remaining decimal part
                frac[i] = exact - static_cast<double>(counts[i]);
                assigned += counts[i];
            }

            // assign the remaining proof
            size_t remainder = n_proofs - assigned;
            // creating indexes
            vector<size_t> idx(n_blocks);
            iota(idx.begin(), idx.end(), 0);
            // sorting for grater decimal part
            partial_sort(idx.begin(), idx.begin() + remainder, idx.end(),
                        [&](size_t a, size_t b){ return frac[a] > frac[b]; });
            // distribution of the rest
            for (size_t k = 0; k < remainder; k++)
                counts[idx[k]]++;
            break;
        }
    }

    return counts;
}


/*
* Generates n_proofs Merkle proof requests from n_blocks input blocks.
*
* Parameters:
*   - blocks:       pointer to input block data (size: n_blocks * BLOCK_SIZE)
*   - n_blocks:     number of leaves (blocks) in the dataset
*   - n_proofs:     total number of proof requests to generate
*   - tamper_rate:  fraction [0.0, 1.0] of proofs that will be invalid (random data)
*   - dist:         distribution strategy for assigning proofs to leaves 
*                      (DIST_UNIFORM/DIST_ZIPF randomized across leaves)
*   - zipf_s:       Zipf exponent (used only if dist == DIST_ZIPF). 
*                   Typical values: 
*                       - ~0.5 nearly uniform.
*                       - 1.0 standard.
*                       - 1.2-1.5 few 'hot' leaves.
*                       - >= 2.0 highly concentrated on few leaves.
*
* Per-leaf counts sum exactly to n_proofs.
* Proofs are emitted in increasing leaf order for good memory locality.
*
* Each proof is either valid (original data) or tampered (random data).
* The 'expected' array stores ground truth for verification.
*
* Uses a fixed-seed RNG for reproducibility.
*/
ProofBatch* generate_proof_requests(const uint8_t* blocks, size_t n_blocks, size_t n_proofs, float tamper_rate, ProofDistribution dist, double zipf_s){

    ProofBatch* batch = (ProofBatch*) malloc(sizeof(ProofBatch));
    batch->proofs     = (uint8_t*)  malloc(n_proofs * BLOCK_SIZE);
    batch->leaf_index = (uint32_t*) malloc(n_proofs * sizeof(uint32_t));
    batch->expected   = (bool*)     malloc(n_proofs * sizeof(bool));
    batch->n_proofs   = n_proofs;

    if (!batch->proofs || !batch->leaf_index || !batch->expected) {
        cerr << "Error allocating ProofBatch\n";
        exit(EXIT_FAILURE);
    }

    // random number generator
    mt19937_64 rng(42);
    uniform_real_distribution<float>   coin(0.0f, 1.0f);
    uniform_int_distribution<uint64_t> random_bytes(0, UINT64_MAX);

    // computing the number of proofs for each leaf based on the distribution
    vector<size_t> proof_per_leaf = build_leaf_counts(n_blocks, n_proofs, dist, zipf_s, rng);

    size_t proof_i = 0;
    for (size_t leaf_i = 0; leaf_i < n_blocks && proof_i < n_proofs; leaf_i++) {
        for (size_t k = 0; k < proof_per_leaf[leaf_i]; k++) {
            uint8_t* dst = batch->proofs + proof_i * BLOCK_SIZE;
            batch->leaf_index[proof_i] = static_cast<uint32_t>(leaf_i);

            if (coin(rng) < tamper_rate) {
                // TAMPERED_DATA: random block, correct index
                for (size_t b = 0; b < BLOCK_SIZE; b += 8) {
                    uint64_t val = random_bytes(rng);
                    memcpy(dst + b, &val, 8);
                }
                batch->expected[proof_i] = false;
            } else {
                // VALID: original block, correct index.
                memcpy(dst, blocks + leaf_i * BLOCK_SIZE, BLOCK_SIZE);
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