#include <iostream>
#include <cstdint>
#include <cassert>
#include "../sha256/sha256_GPU.cuh"
#include "../data/data_generator.hpp"
#include "utils.cuh"
#include "shared_mem_solution.cuh"

#define THREADS_PER_BLOCK 256

/*
* GPU Ampere 100 (from cudaGetDeviceProp): 
*   - SMEM per block: 48 KB (at most).
*   - SMEM per SM: 164 KB (at most).
*
* Each node of the merkle tree is 32 byte, hence:
*   - (48*1024)/32 = 1536 nodes in SMEM per block (at most).
*   - (164*1024)/32 = 5248 nodes in SMEM per block (at most).
*
* Notes:
*   - Obviously, it shouldn't be occupied the entire capacity of the SMEM because this limit the number 
*     of resident blocks and driving to a limited hardware multithreading.
*   - Hence, using SMEM for for multi-level merkle tree computation, we need to consider a subtree with all
*     all the children necessarly to compute the grandparent.
*
*                     GRANDPARENT              grandparents-level
*                     /          \
*                 PARENT       PARENT          parents-level
*                 /   \        /    \
*             CHILD  CHILD  CHILD  CHILD       children-level
*
*   - A good compromise could be using 30-40% of the SMEM capacity per block to storing the subtree,
*       this means around 450-600 nodes.
*
*/ 

/*
* Kernel that builds a horizontal band of the Merkle tree using SMEM.
* Each block is responsible for one subtree with a fixed power-of-two 
* number of leaves (leaves_per_block) loaded from global memory (GMEM), 
* storing intermediate nodes in shared memory (SMEM) during the computation
* and writing in GMEM, ONLY at the end, each level of the subtree using a 
* strided pattern across threads to enable parallel and (when possible) 
* coalesced memory writes.
*
* All blocks except the last one process exactly 'leaves_per_block' leaves,
* which is a power of 2, producing a perfectly balanced subtree with no
* odd node management needed. The last block processes the remaining
* leaves, which may be any number, and applies odd node duplication at 
* each level as needed.
*
* Each block operates on a distinct part of the merkle tree, and multiple
* kernel invocations can be used to iteratively build the tree toward the root.
*
* Example: 8 leaves, 2 levels per block, 4 blocks (THREADS_PER_BLOCK = 2)
*
* Heap layout:
*  Level 0                                              *[n0]    <- dev_merkle_tree points at * (beginning of this level)
*  Level 1:                         [n1]                                  [n2]      
*  Level 2:              *[n3]                 [n4]             
*  Level 3:         *[n7]       [n8]      [n9]       [n10]      <- base_band_GMEM_write_offset refers to * (beginning of this level)
*  Level 4:    *[n15][n16]  [n17][n18] [n19][n20] [n21][n22]    <- base_band_GMEM points at * (beginning of this level)
*  (base band) -----------------------------------------------------------------------------------------------
*
* Block 0 reads [n15][n16][n17][n18] from GMEM, computes [n7][n8] in SMEM, compute [n3] in SMEM, 
* writes [n3][n7][n8] in GMEM.
* Block 1 reads [n19][n20][n21][n22] from GMEM, computes [n9][n10] in SMEM, compute [n4] in SMEM, 
* writes [n4][n9][n10] in GMEM.
*
* After this kernel, a new invocation can treat [n3][n4] as the new
* leaves and compute the next band up to the root.
*
*
* Parameters:
* - base_band_GMEM: pointer to the input merkle tree nodes in GMEM to read to 
*        computing the first level of the subtree in SMEM
* - base_band_GMEM_size: number of nodes to consider starting from base_band_GMEM.
* - base_band_GMEM_write_offset: absolute offset for the leftmost node in the first
*         level in GMEM where to start writing
* - dev_merkle_tree: pointer to the root of the merkle tree.
* - leaves_per_block: number (power of 2) of leaves managed by each block.
* - sha256_windowed: selects the SHA-256 implementation variant
*
*/
__global__ void internal_level_build_SMEM(uint8_t *base_band_GMEM, int base_band_GMEM_size, int base_band_GMEM_write_offset, 
                                          uint8_t *dev_merkle_tree, int leaves_per_block, bool sha256_windowed) {

    int tid = threadIdx.x; // local block index
    // Base index of this block's leaves in the base_band_GMEM.
    // So the stride between consecutive blocks is leaves_per_block. 
    int block_leaf_base = blockIdx.x * leaves_per_block;

    bool is_last_block = (blockIdx.x == gridDim.x - 1);
    int children_level_size = is_last_block ? base_band_GMEM_size - (blockIdx.x * leaves_per_block) : leaves_per_block;
    int parent_level_size = (children_level_size + 1) / 2;
    int levels = __ffs(leaves_per_block) - 1;

    extern __shared__ uint8_t subtree_SMEM[];

    const int smem_nodes = compute_merkle_tree_size(leaves_per_block) - leaves_per_block;

    uint8_t *children_level_SMEM = nullptr;
    uint8_t *parents_level_SMEM = nullptr;
    // offset of the left-most node in the current parent level to compute
    size_t parents_offset = smem_nodes >> 1; 
    assert(parents_offset >= 0);

    bool active_thread = (tid < parent_level_size);

    // computing subtree in SMEM
    for(int curr_lev = 0 ; curr_lev < levels; curr_lev++) {
        // computing parents level
        parents_level_SMEM = subtree_SMEM + parents_offset * SHA256_OUTPUT_BLOCK_SIZE;
  
        if (active_thread) {

            uint8_t *left = nullptr;
            uint8_t *right = nullptr;

            if(curr_lev == 0) {
                // reading from GMEM (only the starting level)
                left = base_band_GMEM + (block_leaf_base + 2*tid)*SHA256_OUTPUT_BLOCK_SIZE;
                right = base_band_GMEM + (block_leaf_base + 2*tid+1)*SHA256_OUTPUT_BLOCK_SIZE; 
            } else{
                // reading from SMEM
                left = children_level_SMEM + (2*tid)*SHA256_OUTPUT_BLOCK_SIZE; 
                right = children_level_SMEM + (2*tid+1)*SHA256_OUTPUT_BLOCK_SIZE;
            }

            uint8_t *parent = parents_level_SMEM + tid*SHA256_OUTPUT_BLOCK_SIZE; // SMEM

            // Odd node management: the last node is duplicated
            bool is_last_odd = (children_level_size % 2 == 1) && (tid == parent_level_size - 1);
            if (is_last_odd) right = left;

            // SMEM write
            compute_parent_hash(parent, left, right, sha256_windowed);
        }

        __syncthreads(); // barrier

        if(active_thread) {
            children_level_SMEM = parents_level_SMEM; // update SMEM
        }
            
        // preparing for the next level
        children_level_size = parent_level_size;
        parent_level_size = (children_level_size + 1) /2;
        assert(parents_offset >= 0);
        parents_offset = (parents_offset - 1) >> 1; // Update to the left-most parent node of the next higher level
        active_thread = (tid < parent_level_size); // Update the status of active thread
        
        __syncthreads(); // barrier

    }

    // update GMEM
    int last_block_leaves = base_band_GMEM_size - (gridDim.x - 1) * leaves_per_block;
    bool last_block_is_different = (last_block_leaves != leaves_per_block);
    // level size for normal block (leaves are already in GMEM)
    int regular_block_lev_size = leaves_per_block >> 1; // = leaves / 2
    // level size for last block (leaves are already in GMEM)
    int last_block_lev_size = (last_block_leaves + 1) /2; // = ceil(last_block_leaves/2)
    // computing offset of the left-most node of the last level to read in SMEM                   
    int smem_offset = smem_nodes >> 1;
    // computing offset of the left-most node of the first level to write in GMEM (i.e. the level above the level 'base_band_GMEM')
    int gmem_lev_base_offset = base_band_GMEM_write_offset;
    int global_lev_size = (base_band_GMEM_size + 1) / 2; // global dimension of the first written level

    // update GMEM loop 
    for(int lev = 0 ; lev < levels; lev++) {
        // computing the level size in the current block
        int curr_block_lev_size = (is_last_block && last_block_is_different) ? last_block_lev_size : regular_block_lev_size;

        // block's starting position in the level
        int block_offset = blockIdx.x * regular_block_lev_size;

        // strided loop for generality (handles cases where curr_block_lev_size > blockDim.x)
        // strided loop: each thread writes nodes tid, tid+blockDim.x, tid+2*blockDim.x, ...
        for (int n = tid; n < curr_block_lev_size; n += blockDim.x) {
            // SMEM node to copy in GMEM
            uint8_t *src = subtree_SMEM + (size_t)(smem_offset + n) * SHA256_OUTPUT_BLOCK_SIZE;
            // GMEM position in which will be copied the node
            uint8_t *dst = dev_merkle_tree + (size_t)(gmem_lev_base_offset + block_offset + n) * SHA256_OUTPUT_BLOCK_SIZE;
            memcpy(dst, src, SHA256_OUTPUT_BLOCK_SIZE);
        }

        __syncthreads(); // barrier

        // update offset to point at the next level to copy
        smem_offset = (smem_offset - 1) >> 1;
        regular_block_lev_size  >>= 1;
        last_block_lev_size = (last_block_lev_size + 1) / 2;
        gmem_lev_base_offset -= (global_lev_size + 1) / 2;
        global_lev_size = (global_lev_size + 1) / 2;
    }
        
}

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
__host__ int compute_optimal_leaves_per_block(int input_level_size, int threads_per_block) {
    int optimal_leaves_per_block = threads_per_block * 2;
    int best_occupancy = 0;

    for (int leaves_per_block = threads_per_block * 2; leaves_per_block <= input_level_size; leaves_per_block *= 2) {
        size_t smem_needed = compute_merkle_tree_size(leaves_per_block) * SHA256_OUTPUT_BLOCK_SIZE;
        int active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            internal_level_build_SMEM,
            threads_per_block,
            smem_needed
        );
        if (active_blocks > best_occupancy) {
            best_occupancy = active_blocks;
            optimal_leaves_per_block = leaves_per_block;
        }
    }
    return optimal_leaves_per_block;
}

/*
 * Computes the total number of nodes in a Merkle subtree with a fixed number
 * of levels above the leaves, considering odd-node duplication when needed.
 *
 * Parameters:
 *  - n_leaf: number of leaf nodes.
 *  - levels: number of levels to build.
 *
 * Returns:
 *  - Total number of nodes in the subtree.
 */
__host__ __forceinline__ size_t compute_fixed_levels_merkle_tree_size(size_t n_leaf, int levels){
    size_t size = n_leaf;
    for (int lev = 0; lev < levels; lev++) { 
        n_leaf = (n_leaf + 1) / 2; 
        size += n_leaf; 
    }
    return size;
}

/*
* Computes the offset of the first node of the top level of a Merkle tree band,
* given the offset of the first node of the base level of the band and the
* number of leaves per block processed by the kernel.
*
* Parameters:
*  - base_band_offset: offset of the first node of the base band level.
*  - base_band_size: number of nodes in the base_band.
*  - leaves_per_block: number of leaves per block (power of 2).
*/
__host__ size_t compute_top_band_offset(size_t base_band_offset, size_t base_band_size, size_t leaves_per_block) {
    int blocks_per_grid = (base_band_size + leaves_per_block - 1) / leaves_per_block;
    size_t offset = base_band_offset;
    size_t size = base_band_size;
    while (size > blocks_per_grid) {
        size_t parent_size = (size + 1) / 2;
        offset -= parent_size;
        size = parent_size;
    }
    return offset;
}

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
void build_merkle_tree_SMEM(size_t n_blocks, uint8_t* host_data_bytes, uint8_t* host_merkle_tree, int leaves_per_block, bool sha256_windowed){ 
#ifndef MERKLE_TEST
    // allocation of the byte array of input data blocks.
    host_data_bytes = generate_random_blocks(n_blocks);
    // computing number of leaves per block
    leaves_per_block = compute_optimal_leaves_per_block(n_blocks, THREADS_PER_BLOCK);
#endif

    if (leaves_per_block <= 0 || ( (leaves_per_block & (leaves_per_block - 1)) != 0 )) {
        cout << "The parameter 'leaves_per_block' must be a power of two." << endl;
        return;
    }
   
    //computing the merkle_tree dimension
    const size_t merkle_tree_size = compute_merkle_tree_size(n_blocks);
    size_t leaf_offset = merkle_tree_size - n_blocks;
  
    // set the working device
    cudaSetDevice(0); 
    // allocation of GPU arrays
    uint8_t *dev_data_bytes;
    uint8_t *dev_merkle_tree;
    gpuErrchk(cudaMalloc((void**) &dev_merkle_tree, (merkle_tree_size)*SHA256_OUTPUT_BLOCK_SIZE));
    gpuErrchk(cudaMalloc((void**) &dev_data_bytes, (n_blocks)*SHA256_INPUT_BLOCK_SIZE));

    uint64_t initial_time = current_time_nsecs();

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_data_bytes, host_data_bytes, (n_blocks)*SHA256_INPUT_BLOCK_SIZE, cudaMemcpyHostToDevice));

    // computing the leaf level on GPU
    int blocks_per_grid = (n_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // rouding up
    leaf_level_build<<<blocks_per_grid, THREADS_PER_BLOCK>>>(n_blocks, leaf_offset, dev_data_bytes, dev_merkle_tree, sha256_windowed);
    cudaDeviceSynchronize();

    // deallocate GPU data bytes
    gpuErrchk(cudaFree(dev_data_bytes));

    size_t base_band_size = n_blocks;
    size_t base_band_offset = leaf_offset;
    //size_t top_band_offset = compute_top_band_offset(base_band_offset, base_band_size, leaves_per_block); 
    size_t base_band_write_offset = leaf_offset - (n_blocks + 1)/2;

    // loop to build internal levels
    int i = 1;
    while(base_band_size > 1){

        uint8_t *base_band = dev_merkle_tree + (base_band_offset * SHA256_OUTPUT_BLOCK_SIZE);
        // computing internal level 
        blocks_per_grid = (base_band_size + leaves_per_block - 1) / leaves_per_block; // rounding up for the last block
        size_t last_block_leaves = base_band_size - (blocks_per_grid - 1) * leaves_per_block;

        // Near the root: reduce leaves_per_block to next power of two ≥ base_band_size to avoid wasting SMEM
        int effective_leaves_per_block = leaves_per_block;
        if (base_band_size <= leaves_per_block) {
            effective_leaves_per_block = 1;
            while (effective_leaves_per_block < base_band_size)
                effective_leaves_per_block <<= 1;
        }
        // compute the dimension of the SMEM that will be used
        size_t size_SMEM = (compute_merkle_tree_size(effective_leaves_per_block) - effective_leaves_per_block) * SHA256_OUTPUT_BLOCK_SIZE;
        // computing the band of the merkle tree through the kernel
        internal_level_build_SMEM<<<blocks_per_grid, THREADS_PER_BLOCK, size_SMEM>>>(
            base_band, base_band_size, (int)base_band_write_offset, dev_merkle_tree, effective_leaves_per_block, sha256_windowed);

        cudaDeviceSynchronize();
        
        // update for the next iteration
        size_t old_base_band_size = base_band_size;
        base_band_size = blocks_per_grid; // because each block produce a subtree, hence the new base band is the subtree roots level.
        base_band_offset = compute_top_band_offset(base_band_offset, old_base_band_size, effective_leaves_per_block);
        base_band_write_offset = base_band_offset - (base_band_size + 1) / 2;
        i++;
    }

#ifdef MERKLE_TEST
    
    if(host_merkle_tree != nullptr)
        gpuErrchk(cudaMemcpy(host_merkle_tree, dev_merkle_tree,(merkle_tree_size)*SHA256_OUTPUT_BLOCK_SIZE,
            cudaMemcpyDeviceToHost));

#endif

    // deallocate GPU merkle_tree
    gpuErrchk(cudaFree(dev_merkle_tree));
}