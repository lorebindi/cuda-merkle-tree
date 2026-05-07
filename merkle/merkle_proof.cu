#include<iostream>
#include<stdlib.h>
#include<cstdint>
#include<string.h>

#include "../sha256/sha256_GPU.cuh"
#include "../data/data_generator.hpp"
#include "utils_gpu.cuh"
#include "utils_cpu.hpp"
#include "merkle_proof.cuh"

/*
* Parameters:
*   - dev_merkle_tree: pointer to the root of the merkle tree (stored in GMEM).
*   - merkle_tree_depth: number of levels of the merkle tree.
*   - leaves_offset: position of the first leaf in the merkle tree heap-style.
*   - leaves_number: number of leaves in the merkle tree.
*   - proofs: pointer to the array of merkle proofs, i.e. bytes representing
*             leaf nodes to check.  
*   - leaf_index: pointer to the integer array that contains the relative position 
*                 in which consider each merkle proof.
*   - n_proofs: number of merkle proof to compute.
*   - results: pointer to the results array.
*
*/
__global__ void merkle_proofs_verification (uint8_t *dev_merkle_tree, uint32_t merkle_tree_depth, int leaves_offset, 
                                                    int leaves_number, uint8_t* proofs, uint32_t* leaf_index, int n_proofs, 
                                                    bool* result){

    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i >= n_proofs) return;

    int curr_lev_size = leaves_number;
    // offset of the first node in the current level
    int curr_lev_offset = leaves_offset;
    // leaf position in the merkle tree for the merkle proof.
    int curr_node_offset = leaves_offset + leaf_index[i];
    // teoric sibling.
    int index_in_level = curr_node_offset - curr_lev_offset;
    int sibling_offset_step = (index_in_level & 1) ? -1 : +1;
    int sibling_node_offset = curr_node_offset + sibling_offset_step;
    // odd management.                                                    
    bool has_sibling = (sibling_node_offset >= leaves_offset) && (sibling_node_offset < leaves_offset + leaves_number); 
    if (!has_sibling){
        // executing only by the last thread of the last block (if 'leaves_number' is odd).
        sibling_node_offset = curr_node_offset;
    }

    
    // current node in the path leaf-to-root
    uint8_t* curr_node = proofs + i * SHA256_OUTPUT_BLOCK_SIZE;
    // sibling node of the current node necessary to go on in the path.
    uint8_t* sibling_node = dev_merkle_tree + sibling_node_offset * SHA256_OUTPUT_BLOCK_SIZE;
    // temporary parent node 
    __align__(16) uint8_t temp_parent_node [SHA256_OUTPUT_BLOCK_SIZE]; 
    // usefull pointers
    uint8_t* left, *right;
    int parent_lev_offset = 0, parent_offset = 0;

    // initializing result
    result[i] = true;

    // check for a merkle tree with only the root
    if (merkle_tree_depth == 0) {
        result[i] = device_memcmp32(curr_node, dev_merkle_tree);
        return;
    }

    for(int lev = 0; lev < merkle_tree_depth; lev++){
        
        // computing and storing the hash of the temporary parent node
        left  = (sibling_offset_step == +1) ? curr_node : sibling_node;
        right = (sibling_offset_step == +1) ? sibling_node : curr_node;
        compute_parent_hash(temp_parent_node, left, right);
        // computing parent offset 
        parent_lev_offset = curr_lev_offset - (curr_lev_size + 1) / 2;
        parent_offset = parent_lev_offset + ((curr_node_offset - curr_lev_offset) >> 1);       
   
        // comparing the just-computed parent hash with the one in the merkle tree
        if(result[i] && !device_memcmp32(temp_parent_node, dev_merkle_tree + parent_offset * SHA256_OUTPUT_BLOCK_SIZE))
            result[i] = false;
        
        curr_lev_size = (curr_lev_size + 1) / 2; // computing the upper level size.
        curr_lev_offset = parent_lev_offset; // parent lev offset became curr lev offset.
        curr_node = temp_parent_node; // the temporary parent node became the curr node.
        curr_node_offset = parent_offset; 
        
        // computing sibling for the upper level
        index_in_level = curr_node_offset - curr_lev_offset;
        sibling_offset_step = (index_in_level & 1) ? -1 : +1;
        sibling_node_offset = curr_node_offset + sibling_offset_step;
        // odd management.                                                    
        has_sibling = (sibling_node_offset >= curr_lev_offset) && (sibling_node_offset < curr_lev_offset + curr_lev_size); 
        if (!has_sibling){
            // executing only by the last thread of the last block (if 'leaves_number' is odd).
            sibling_node_offset = curr_node_offset;
        }
        sibling_node = dev_merkle_tree + sibling_node_offset * SHA256_OUTPUT_BLOCK_SIZE;
    }
}

/*
* Computes SHA-256 hashes for all proof inputs in a ProofBatch using the GPU.
*
* This function takes a batch of raw proof data stored on the host, transfers
* it to the GPU, computes the SHA-256 hash of each proof in parallel using
* a CUDA kernel, and returns the resulting hashes to the host.
*
* Parameters:
*  - proof_batch: batch containing raw proof inputs to be hashed
*  - n_proofs: number of proofs in the batch
*
* Returns:
*  - Pointer to a host-allocated array of size:
*      n_proofs * SHA256_OUTPUT_BLOCK_SIZE
*    containing the computed SHA-256 hashes.
*/
uint8_t* get_hashed_proofs(ProofBatch* proof_batch, size_t n_proofs){
    // allocate the return value
    uint8_t* host_hashed_proofs = (uint8_t*) malloc (sizeof(uint8_t)* SHA256_OUTPUT_BLOCK_SIZE * n_proofs);

    // set the working device
    cudaSetDevice(0); 
    // allocation of GPU arrays
    uint8_t *dev_hashed_proofs;
    uint8_t *dev_proofs;
    gpuErrchk(cudaMalloc((void**) &dev_hashed_proofs, ( n_proofs *SHA256_OUTPUT_BLOCK_SIZE)));
    gpuErrchk(cudaMalloc((void**) &dev_proofs, (n_proofs * SHA256_INPUT_BLOCK_SIZE)));

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_proofs, proof_batch->proofs, (n_proofs)*SHA256_INPUT_BLOCK_SIZE, cudaMemcpyHostToDevice));

    // computing the leaf level on GPU
    int blocks_per_grid = (n_proofs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // rouding up
    leaf_level_build<SHA256_WINDOWED><<<blocks_per_grid, THREADS_PER_BLOCK>>>(n_proofs, 0, dev_proofs, dev_hashed_proofs);
    cudaDeviceSynchronize();

    // copy of dev_hashed_proofs in host_hashed_proofs
    gpuErrchk(cudaMemcpy(host_hashed_proofs, dev_hashed_proofs, n_proofs * SHA256_OUTPUT_BLOCK_SIZE, cudaMemcpyDeviceToHost));

    // deallocate GPU data bytes
    gpuErrchk(cudaFree(dev_proofs));
    gpuErrchk(cudaFree(dev_hashed_proofs));

    return host_hashed_proofs;
}

/*
 * Launches a GPU kernel to verify a batch of Merkle proofs in parallel.
 *
 * Each proof in the batch is checked independently by a separate GPU thread,
 * which reconstructs the Merkle root from a leaf and its authentication path
 * and compares it against the known root stored in the Merkle tree.
 *
 * Parameters:
 *  - proof_batch: batch of Merkle proofs to verify
 *  - merkle_tree_gpu: GPU-resident Merkle tree used as reference
 *  - out_elapsed: elapsed merkle proofs verification time.
 *
 * Returns:
 *  - Pointer to an array of boolean values of size n_proofs:
 *    true  -> proof is valid
 *    false -> proof is invalid
 */
bool* compute_merkle_proofs(ProofBatch* proof_batch, MerkleTreeGPU* merkle_tree_gpu, uint64_t* out_elapsed){
    
    size_t n_proofs = proof_batch->n_proofs;
    uint8_t* host_hashed_proofs = get_hashed_proofs(proof_batch, n_proofs);

    uint8_t* dev_proofs;
    uint32_t* dev_leaf_index;
    bool* dev_result;

    gpuErrchk(cudaMalloc((void**)&dev_proofs, n_proofs * SHA256_OUTPUT_BLOCK_SIZE));
    gpuErrchk(cudaMalloc((void**)&dev_leaf_index, n_proofs * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc((void**)&dev_result, n_proofs * sizeof(bool)));

    uint64_t initial_time = current_time_nsecs();

    gpuErrchk(cudaMemcpy(dev_proofs, host_hashed_proofs,
    n_proofs * SHA256_OUTPUT_BLOCK_SIZE, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dev_leaf_index, proof_batch->leaf_index,
    n_proofs * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int threads_per_block = THREADS_PER_BLOCK;
    int blocks_per_grid = (n_proofs + threads_per_block - 1) / threads_per_block;

    merkle_proofs_verification<<<blocks_per_grid, threads_per_block>>>(
        merkle_tree_gpu -> dev_tree,
        merkle_tree_gpu -> base.depth,
        merkle_tree_gpu -> base.size - merkle_tree_gpu -> base.n_leaves,
        merkle_tree_gpu -> base.n_leaves,
        dev_proofs,
        dev_leaf_index,
        n_proofs,
        dev_result
    );

    cudaDeviceSynchronize();

    bool* host_result = (bool*) malloc (n_proofs * sizeof(bool));

    gpuErrchk(cudaMemcpy(host_result, dev_result, n_proofs * sizeof(bool), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    if (out_elapsed)
        *out_elapsed = end_time - initial_time;

    // free
    gpuErrchk(cudaFree(dev_proofs));
    gpuErrchk(cudaFree(dev_leaf_index));
    gpuErrchk(cudaFree(dev_result));

    free(host_hashed_proofs);

    return host_result;
    
}