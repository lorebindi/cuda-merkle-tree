# CUDA Merkle Tree

High-performance implementation of a Merkle Tree on GPU using CUDA.

This project explores how to leverage GPU parallelism to efficiently compute Merkle trees, commonly used in cryptography, blockchains, and data integrity systems. The implementation is supported by performance benchmarks, whose results are presented below.

## Repository Structure

- `benchmarks/`: this directory contains the source code for performance evaluation and benchmarking.
- `data/`: this directory contains the source code to generate both the data blocks (64 byte) used to build the leaves level of the merkle tree and the merkle proof.
- `merkle/`: this directory contains the source code both for the merkle tree building and for the merkle proof verification.
- `sha256/`: this directory contains both the CPU implementation and the GPU implementation of sha256 hash function (FIPS 180-4 compliant). 
- `tests/`: contains all tests used to validate the correctness of the proposed solution.

## SHA256: Standard vs Windowed Implementation

This project includes two variants of the SHA-256:

- Standard implementation: uses a full 64-word message schedule (`W[64]`)
- Windowed implementation: uses a rolling buffer of only 16 words.

In SHA-256, each word `W[i]` depends only on: `W[i-2]`, `W[i-7]`, `W[i-15]`, `W[i-16]`. This means that storing all 64 words is not strictly necessary. The windowed approach exploits this by keeping only the last 16 values. 

This design reduces the per-thread memory footprint. Since each GPU thread independently performs the hashing operation, it must maintain its own message schedule; shrinking it from 64 to 16 words significantly lowers register pressure and local memory usage.

The impact of this optimization is evaluated through the benchmarks reported below.

## Merkle Tree

In this repository, Merkle trees are represented as array-based heaps, with the root at the first position and the leaves at the end of the array. This approach avoids pointer-based structures and enables contiguous memory access, which is crucial for GPU performance.

On the GPU side, two different merkle tree building implementations are provided:
- **Naive implementation**: a kernel computes a single level of the Merkle tree at a time, starting from the lowest level. The kernel is launched repeatedly, once per tree level, until the root is reached.
- **SMEM-optimized implementation**: that builds multiple levels of the tree within a single launch. Each block is responsible for constructing a subtree stored in shared memory (SMEM), which is then written back to global memory (GMEM) once completed. The size of the subtree can be configured using the `leaves_per_block` parameter, and it is the same for all blocks.

The choice between the SHA-256 implementations used in Merkle tree construction is controlled via a compile-time macro `SHA256_WINDOWED` defined in `merkle/utils.h`. While this approach is not ideal from a software design perspective, it was deliberately chosen to simplify experimentation and focus on performance analysis.

## Merkle Proof

The repository includes a mechanism (`data/`) to generate batches of Merkle proof requests starting from a fixed set of input data blocks (64 bytes) previously used to construct the leaves level of the Merkle tree.

Merkle proof batches are generated using different sampling strategies over the leaf space.

- **Uniform coverage**: each leaf is guaranteed to appear at least once in the batch.
- **Zipf (skewed)**: selection follows a power-law distribution, controlled by an exponent, producing realistic workloads with a small number of frequently accessed leaves.

These modes allow benchmarking under both balanced and realistic access patterns.

On the GPU-side Merkle Proof verification is performed in parallel, each thread independently validates a single proof. For each request, the process starts from a leaf hash and iteratively recomputes parent hashes using the authentication path. At each level, the computed hash is compared against the corresponding node stored in the Merkle tree. If all intermediate hashes match and the final reconstructed root equals the stored root, the proof is considered valid; otherwise, it is rejected.


## Compilation

- `make test`: builds the test executables.
- `make bench`: builds the benchmark executables.

After compilation, the corresponding binaries can be run manually.
