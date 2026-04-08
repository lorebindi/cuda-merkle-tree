#ifndef MERKLE_TREE_GPU_CUH
#define MERKLE_TREE_GPU_CUH

#include <cstdint>

/*
 * MerkleTreeGPU struct and utility functions
 *
 * This struct represents a Merkle tree stored on the GPU.
 * - dev_tree: pointer to the GPU memory containing the tree nodes
 * - size: total number of nodes in the tree
 *
 * Instances of MerkleTreeGPU are returned by all merkle-tree-building functions
 * (both naive and shared-memory implementations), and are used as input
 * to merkle proof verification routines.
 */
typedef struct MerkleTreeGPU {
    uint8_t* dev_tree; // GPU pointer
    size_t size; // number of nodes
    size_t n_leaves;
    uint32_t depth;
} MerkleTreeGPU;

static inline uint32_t compute_merkle_depth(size_t n_leaves) {
    uint32_t depth = 0;
    while (n_leaves > 1) {
        n_leaves = (n_leaves + 1) >> 1;
        depth++;
    }
    return depth;
}

// Constructor: creates and initializes a MerkleTreeGPU instance
static inline MerkleTreeGPU* merkle_tree_gpu_create(uint8_t* dev_ptr, size_t tree_size, size_t n_leaves) {
    MerkleTreeGPU* tree = (MerkleTreeGPU*) malloc(sizeof(MerkleTreeGPU));
    if (!tree) {
        fprintf(stderr, "malloc failed in merkle_tree_gpu_create\n");
        return NULL;
    }
    tree->dev_tree = dev_ptr;
    tree->size = tree_size;
    tree->n_leaves = n_leaves;
    tree->depth = compute_merkle_depth(n_leaves);

    return tree;
}

// Destructor: frees GPU memory and deallocates the struct
static inline void merkle_tree_gpu_destroy(MerkleTreeGPU* tree) {
    if (tree) {
        if (tree->dev_tree) {
            cudaError_t err = cudaFree(tree->dev_tree);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
            }
            tree->dev_tree = NULL;
        }
        free(tree);
    }
}

#endif
