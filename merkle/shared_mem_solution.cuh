#include <cstdint>

/*
* Builds a Merkle tree on the GPU using a shared memory (SMEM) optimized approach.
*
* The function first computes the leaf level from input data, then iteratively
* constructs the upper levels of the tree in horizontal bands. Each kernel launch
* processes a band of the tree, where each block builds a subtree in shared memory
* and writes the resulting nodes back to global memory (GMEM).
*
* At each iteration, the base band (current level) is reduced to a smaller band
* containing the roots of the computed subtrees, until the final root is produced.
*
* Parameters:
*  - n_blocks: number of input data blocks (leaves)
*  - host_data_bytes: pointer to input data (optional if generated internally)
*  - host_merkle_tree: optional output buffer for the full Merkle tree 
                       (used only when MERKLE_TEST is enabled).
*  - leaves_per_block: optional number of leaves processed per block, it must be power of 2
                       (used only when MERKLE_TEST is enabled). 
*  - sha256_windowed: selects the SHA-256 implementation variant
*/
void build_merkle_tree_SMEM(
    size_t n_blocks, 
    uint8_t* host_data_bytes, 
    uint8_t* host_merkle_tree = nullptr, 
    int leaves_per_block = -1, 
    bool sha256_windowed = true);