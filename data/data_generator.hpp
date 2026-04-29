#ifndef PROOF_BATCH_HPP
#define PROOF_BATCH_HPP

#include <cstdint>

#define BLOCK_SIZE 64

/* This structure represents a batch of Merkle proof requests.
 * Each entry corresponds to a single proof to be verified.
 *
 * Fields:
 *  - proofs:
 *      Contiguous array of input data blocks (one per proof) that will be
 *      hashed and verified against the Merkle tree.
 *
 *  - leaf_index:
 *      Array of leaf indices, one per proof. Each index identifies the
 *      position of the corresponding leaf in the Merkle tree (relative
 *      to the leaf level, i.e., must be offset by leaf_offset).
 *
 *  - expected:
 *      Host-side ground truth bitmap used for validation.
 *      expected[i] = true  -> proof i is valid
 *      expected[i] = false -> proof i is intentionally tampered
 *
 *  - n_proofs:
 *      Total number of proof requests in the batch.
 *
 * Notes:
 *  - The 'expected' array is intended for host-side verification and is
 *    not required for GPU computation.
 */
typedef struct ProofBatch {
    uint8_t*  proofs;
    uint32_t* leaf_index;
    bool*     expected;
    size_t    n_proofs;
} ProofBatch;

/*
 * ProofDistribution defines the strategy used to assign proofs across leaves.
 *
 * - DIST_UNIFORM: Each leaf is guaranteed at least one proof, representing
 *   a simple baseline where proofs are evenly distributed.
 *
 * - DIST_ZIPF: Proofs follow a Zipfian (power-law) distribution, meaning a small
 *   number of leaves receive many proofs while most receive few. This models
 *   more realistic scenarios where popularity or activity is highly skewed.
 */
typedef enum {
    DIST_UNIFORM,  // baseline: at least one proof per leaf.
    DIST_ZIPF,     // realistic: power-law distribution with casual popolarity
} ProofDistribution;

/*
 * Allocates a contiguous buffer of size n_blocks * BLOCK_SIZE bytes.
 * Each block is filled with pseudo-random data.
 */
uint8_t* generate_random_blocks(size_t n_blocks);
void free_blocks(uint8_t* ptr);

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
ProofBatch* generate_proof_requests(const uint8_t* blocks, size_t n_blocks, size_t n_proofs, float tamper_rate, ProofDistribution dist, double zipf_s);

/* Free ProofBatch */
void free_proof_batch(ProofBatch* batch);


#endif