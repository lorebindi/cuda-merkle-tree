#include <cstdint>
#include "merkle_tree.cuh"

MerkleTreeCPU* host_build_merkle_tree(size_t n_blocks, uint8_t* host_data_bytes, bool sha256_windowed);