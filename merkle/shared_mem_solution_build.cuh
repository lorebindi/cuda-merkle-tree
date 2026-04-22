#include <cstdint>
#include "merkle_tree.cuh"

/*
* Determines the optimal leaves_per_block for the internal_level_build_SMEM kernel
* by evaluating occupancy. It tests power-of-two candidates, computes required
* shared memory, and selects the configuration that maximizes active blocks per SM.
*
* Parameters:
* - input_level_size: number of nodes in the current level.
* - threads_per_block: number of threads for each block.
*
* Returns:
* - optimal leaves_per_block (power of two)
*/
__host__ int compute_optimal_leaves_per_block(int input_level_size, int threads_per_block);

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
*  - n_blocks: number of input data blocks (leaves).
*  - host_data_bytes: pointer to input data.
*  - leaves_per_block: optional number of leaves processed per block, it must be power of 2
*                       (used only when MERKLE_TEST is enabled). 
*
* Returns:
*  - A pointer to a MerkleTreeGPU structure representing the tree stored in GPU memory.
*    The structure contains the device pointer to the tree and its metadata. The caller
*    is responsible for freeing it.
*/
MerkleTreeGPU* build_merkle_tree_SMEM(
    size_t n_blocks, 
    uint8_t* host_data_bytes, 
    int leaves_per_block = -1);