#include <cstdlib>
#include <cstdint>
#include <random>
#include <cstring>
#include <iostream>

#define BLOCK_SIZE 64  // each data block is 64 bytes

/*
 * Allocates a contiguous buffer of size n_blocks * BLOCK_SIZE bytes.
 * Each block is filled with pseudo-random data.
 */
uint8_t* generate_random_blocks(size_t n_blocks) {
    size_t total_size = n_blocks * BLOCK_SIZE;

    uint8_t* buffer = (uint8_t*) malloc(total_size);
    if (!buffer) {
        std::cerr << "Error allocating memory for data blocks\n";
        exit(EXIT_FAILURE);
    }

    std::mt19937_64 rng(12345);  // fixed seed for reproducibility
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    // Fill 8 bytes at a time
    for (size_t i = 0; i < total_size; i += 8) {
        uint64_t value = dist(rng);
        std::memcpy(buffer + i, &value, 8);
    }

    return buffer;
}

/*
 * Frees the buffer of blocks
 */
void free_blocks(uint8_t* ptr) {
    if (ptr) free(ptr);
}