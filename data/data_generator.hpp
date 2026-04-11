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
 * Allocates a contiguous buffer of size n_blocks * BLOCK_SIZE bytes.
 * Each block is filled with pseudo-random data.
 */
uint8_t* generate_random_blocks(size_t n_blocks);
void free_blocks(uint8_t* ptr);

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
ProofBatch* generate_proof_requests(const uint8_t* blocks, size_t n_blocks, size_t n_proofs, float tamper_rate);

/* Free ProofBatch */
void free_proof_batch(ProofBatch* batch);


#endif