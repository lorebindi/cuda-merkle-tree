#ifndef MERKLE_TREE_CUH
#define MERKLE_TREE_CUH

/*
 * merkle_tree.hpp
 *
 * This header defines the core data structures used to represent a Merkle tree
 * both on the GPU and on the CPU.
 *
 * It provides:
 *  - a common base structure (MerkleTreeBase) containing metadata such as
 *    total size, number of leaves, and tree depth
 *  - a GPU-specific representation (MerkleTreeGPU), storing the tree in device memory
 *  - a CPU-specific representation (MerkleTreeCPU), storing the tree in host memory
 *  - utility functions to create and destroy these structures
 *
 * The tree is stored as a contiguous array in heap-like layout:
 *  - root at index 0
 *  - internal nodes follow
 *  - leaves are stored in the last positions
 *
 * This header is shared across both CPU and GPU implementations and is used
 * by build routines and Merkle proof algorithms.
 */


#include <cstdint>
#include <cuda_runtime.h>


/*
 * MerkleTreeBase struct
 *
 * This struct contains the common metadata shared by both CPU and GPU
 * representations of a Merkle tree.
 *
 * - size: total number of nodes in the tree (including internal nodes and leaves)
 * - n_leaves: number of leaf nodes (input data blocks)
 * - depth: number of levels from leaves to root (root excluded from count of edges)
 *
 * This struct is embedded inside both MerkleTreeGPU and MerkleTreeCPU.
 */
typedef struct MerkleTreeBase {
    size_t size;
    size_t n_leaves;
    uint32_t depth;
} MerkleTreeBase;

static inline uint32_t compute_merkle_depth(size_t n_leaves) {
    uint32_t depth = 0;
    while (n_leaves > 1) {
        n_leaves = (n_leaves + 1) >> 1;
        depth++;
    }
    return depth;
}


/*
 * MerkleTreeGPU struct
 *
 * This struct represents a Merkle tree stored entirely in GPU global memory.
 *
 * - base: common metadata describing the structure of the tree
 * - dev_tree: pointer to the contiguous memory region in device (GMEM)
 *             containing all tree nodes stored in heap-like layout
 *
 * Instances of MerkleTreeGPU are returned by GPU-based build functions
 * (e.g., naive and shared-memory implementations) and are typically used
 * as input for GPU-based Merkle proof generation and verification.
 *
 * The lifetime of the underlying device memory is managed through the
 * provided create/destroy utility functions.
 */
typedef struct MerkleTreeGPU {
    MerkleTreeBase base;
    uint8_t* dev_tree;  // device pointer
} MerkleTreeGPU;

static inline MerkleTreeGPU* merkle_tree_gpu_create(uint8_t* dev_ptr, size_t tree_size, size_t n_leaves) {
    MerkleTreeGPU* tree = (MerkleTreeGPU*) malloc(sizeof(MerkleTreeGPU));
    if (!tree) { fprintf(stderr, "malloc failed in merkle_tree_gpu_create\n"); return NULL; }
    tree->base = { tree_size, n_leaves, compute_merkle_depth(n_leaves) };
    tree->dev_tree = dev_ptr;
    return tree;
}

static inline void merkle_tree_gpu_destroy(MerkleTreeGPU* tree) {
    if (tree) {
        if (tree->dev_tree) {
            cudaError_t err = cudaFree(tree->dev_tree);
            if (err != cudaSuccess)
                fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
            tree->dev_tree = NULL;
        }
        free(tree);
    }
}

/*
 * MerkleTreeCPU struct
 *
 * This struct represents a Merkle tree stored in host (CPU) memory.
 *
 * - base: common metadata describing the structure of the tree
 * - host_tree: pointer to the contiguous memory region in host memory
 *              containing all tree nodes stored in heap-like layout
 *
 * This implementation is mainly intended as a reference version for:
 *  - correctness validation of GPU implementations
 *  - testing and debugging
 *
 * The memory pointed by host_tree is allocated on the host and must be
 * managed through the provided create/destroy utility functions.
 */
typedef struct MerkleTreeCPU {
    MerkleTreeBase base;
    uint8_t* host_tree;  // host pointer
} MerkleTreeCPU;

static inline MerkleTreeCPU* merkle_tree_cpu_create(uint8_t* host_ptr, size_t tree_size, size_t n_leaves) {
    MerkleTreeCPU* tree = (MerkleTreeCPU*) malloc(sizeof(MerkleTreeCPU));
    if (!tree) { fprintf(stderr, "malloc failed in merkle_tree_cpu_create\n"); return NULL; }
    tree->base = { tree_size, n_leaves, compute_merkle_depth(n_leaves) };
    tree->host_tree = host_ptr;
    return tree;
}

static inline void merkle_tree_cpu_destroy(MerkleTreeCPU* tree) {
    if (tree) {
        if (tree->host_tree) {
            free(tree->host_tree);
            tree->host_tree = NULL;
        }
        free(tree);
    }
}


#endif
