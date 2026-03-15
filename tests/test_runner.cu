/*
* 
* This file contains all the set of tests used to verify the correctness of this work.
*
* In particular contains test on:
*  - SHA-256 implementation on CUDA. 
*  - Merkle tree built.
*
*/

#include <iostream>
#include <cstring>
#include <stdint.h>
#include "../sha256/sha256_GPU.cuh"
#include "../sha256/sha256_CPU.hpp"
#include "../merkle/utils.cuh"
#include "../merkle/naive_solution.cuh"
#include "../data/data_generator.hpp"

using namespace std;

struct TestVector {
    const char* msg;
    const char* expected;
};

TestVector test_vectors[] = {
    {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
    {"hello world", "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"},
    {"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // 55 a
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"}
};

#define NUM_TESTS (sizeof(test_vectors)/sizeof(TestVector))

/* Simple kernel that computes SHA-256 of a single block */
__global__ void single_test_sha256_kernel(const uint8_t* input, uint8_t* output, bool sha256_windowed) {
    sha256_single_block(input, output, sha256_windowed);
}

/* Utility function that converts a 32-byte hash to a hexadecimal string */
void hash_to_hex(const uint8_t hash[32], char hex[65]) {
    for (int i = 0; i < 32; i++)
        sprintf(hex + i*2, "%02x", hash[i]);
    hex[64] = 0;
}

__host__ void print_result(const char* msg, const char* expected, const char* actual) {
    cout << "Message: \"" << msg << "\"\n";
    if (strcmp(expected, actual) == 0)
        cout << "Result: PASS\n";
    else
        cout << "Result: FAIL\nExpected: " << expected << "\nGot     : " << actual << "\n";
    cout << "-----------------------------------\n";
}

/* 
* Function that runs a single 64 byte SHA-256 test using
* a kernel with 1D grid of 1D block with one thread, compares the result 
* with the expected hash, and prints PASS/FAIL.
*/
bool run_single_block_test(const char* msg, const char* expected, bool print, bool sha256_windowed) {
    size_t len = strlen(msg);

    uint8_t h_block[64];
    sha256_pad_single_block(reinterpret_cast<const uint8_t*>(msg), len, h_block);

    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, 64);
    cudaMalloc(&d_output, 32);
    cudaMemcpy(d_input, h_block, 64, cudaMemcpyHostToDevice);

    single_test_sha256_kernel<<<1,1>>>(d_input, d_output, sha256_windowed);
    cudaDeviceSynchronize();

    uint8_t h_hash[32];
    cudaMemcpy(h_hash, d_output, 32, cudaMemcpyDeviceToHost);

    char actual[65];
    hash_to_hex(h_hash, actual);

    if(print)
        print_result(msg, expected, actual);

    cudaFree(d_input);
    cudaFree(d_output);

    return strcmp(expected, actual) == 0;
}

void run_all_test_vectors(bool sha256_windowed){
    for (int t = 0; t < NUM_TESTS; t++) {
        run_single_block_test(test_vectors[t].msg, test_vectors[t].expected, true, sha256_windowed);
    }
}

/* Function that tests consistency of SHA-256 for repeated hashing */
void run_consistency_test(bool sha256_windowed) {
    for (int t = 0; t < NUM_TESTS; t++) {
        const char* msg = test_vectors[t].msg;
        const char* expected = test_vectors[t].expected;
        bool all_pass = true;

        for (int repeat = 0; repeat < 5; repeat++) {
            bool pass = run_single_block_test(msg, expected, false, sha256_windowed);
            if (!pass) {
                all_pass = false; // segna che c'è stato un fallimento
            }
        }

        // Stampa il risultato finale dopo tutti i tentativi
        if (all_pass)
            cout << "Message: \"" << msg << "\" -> PASS (all repetitions)\n";
        else
            cout << "Message: \"" << msg << "\" -> FAIL (some repetition failed)\n";

        cout << "-----------------------------------\n";
    }
}

/*
 * Test function for the naive Merkle tree implementation.
 * 
 * Steps:
 *  1. Generates n_blocks of random input data.
 *  2. Builds the Merkle tree on the GPU using the naive solution.
 *  3. Computes the SHA-256 hash of each leaf on the CPU.
 *  4. Compares the CPU-computed hashes with the GPU-computed leaf hashes.
 *  5. Computes the Merkle tree root on CPU level by level and compares with GPU root.
 * 
 * Reports mismatches if any, otherwise confirms all leaf hashes and root match.
 */
void test_naive_solution(size_t n_blocks, bool sha256_windowed) {
    // generate bytes of data.
    cout << "Data blocks (leaves) number: " << n_blocks << "\n" << endl;
    uint8_t* host_data = generate_random_blocks(n_blocks);

     // preparing host merkle tree.
    size_t merkle_tree_size = compute_merkle_tree_size(n_blocks);
    size_t leaf_offset = merkle_tree_size - n_blocks;
    cout << "Merkle tree size: " << merkle_tree_size << "\n" << endl;
    uint8_t* host_merkle_tree = (uint8_t*) malloc(merkle_tree_size * SHA256_OUTPUT_BLOCK_SIZE);
    // build the merkle tree on the GPU
    build_merkle_tree_naive(n_blocks, host_data, host_merkle_tree);

    cout << "GPU Merkle Tree computed. \n" << endl;

    // leafs verification
    bool correct = true;
    uint8_t* curr_lev = (uint8_t*) malloc(n_blocks * SHA256_OUTPUT_BLOCK_SIZE);
    for (size_t i = 0; i < n_blocks; i++) {
        // computing CPU hash of the i-th leaf
        sha256_single_block_CPU(host_data + i*SHA256_INPUT_BLOCK_SIZE, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, true);
        // retrieve the GPU hash of the i-th leaf
        uint8_t* gpu_leaf = host_merkle_tree + (leaf_offset + i)*SHA256_OUTPUT_BLOCK_SIZE;
        // compare
        if (memcmp(curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, gpu_leaf, SHA256_OUTPUT_BLOCK_SIZE) != 0) {
            correct = false;
            cout << "Mismatch at leaf " << i << endl;
        }
    }

    if (correct) {
        cout << "All leaf hashes match CPU calculation! \n" << endl;
    } else {
        cout << "Some leaf hashes mismatch!" << endl;
    }

    free(host_data);

    // root verification 
    uint8_t* prec_lev = curr_lev;
    size_t prec_lev_size = n_blocks;

    while (prec_lev_size > 1) {
        size_t curr_lev_size = (prec_lev_size + 1) / 2;
        uint8_t* curr_lev = (uint8_t*) malloc(curr_lev_size * SHA256_OUTPUT_BLOCK_SIZE);

        for (size_t i = 0; i < curr_lev_size; i++) {
            uint8_t* left = prec_lev + (2*i)*SHA256_OUTPUT_BLOCK_SIZE;
            uint8_t* right = prec_lev + (2*i+1)*SHA256_OUTPUT_BLOCK_SIZE;

            if ((prec_lev_size % 2 == 1) && (i == curr_lev_size - 1))
                right = left; 

            uint8_t concatenated[64];
            memcpy(concatenated, left, SHA256_OUTPUT_BLOCK_SIZE);
            memcpy(concatenated + SHA256_OUTPUT_BLOCK_SIZE, right, SHA256_OUTPUT_BLOCK_SIZE);

            sha256_single_block_CPU(concatenated, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, true);
        }

        free(prec_lev);
        prec_lev = curr_lev;
        prec_lev_size = curr_lev_size;
    }

    // compare
    if (memcmp(prec_lev, host_merkle_tree, SHA256_OUTPUT_BLOCK_SIZE) != 0) cout << "Roots mismatch" << endl;
    else cout << "Roots MATCH\n" << endl;

    free(host_merkle_tree);
    free(prec_lev);

    return;
}

int main() {
    cout << "================ Single Block SHA-256 Tests (traditional transform) ================\n";
    run_all_test_vectors(false);
    cout << "================ Single Block SHA-256 Tests (windowed transform) ================\n";
    run_all_test_vectors(true);
    cout << "\n================ Consistency Test (traditional transform) ================\n";
    run_consistency_test(false);
    cout << "\n================ Consistency Test (windowed transform) ================\n";
    run_consistency_test(true);
    cout << "\n================ Testing naive Merkle tree build ================\n" << endl;
    test_naive_solution(50000, true);

    cudaDeviceReset();

    return 0;
}