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
#include "../merkle/naive_solution.cuh"
#include "../data/data_generator.hpp"

using namespace std;

#define N_BLOCKS 128  // piccolo numero per debug

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
__global__ void single_test_sha256_kernel(const uint8_t* input, uint8_t* output, bool window) {
    sha256_single_block(input, output, window);
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
bool run_single_block_test(const char* msg, const char* expected, bool print, bool window) {
    size_t len = strlen(msg);

    uint8_t h_block[64];
    sha256_pad_single_block(reinterpret_cast<const uint8_t*>(msg), len, h_block);

    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, 64);
    cudaMalloc(&d_output, 32);
    cudaMemcpy(d_input, h_block, 64, cudaMemcpyHostToDevice);

    single_test_sha256_kernel<<<1,1>>>(d_input, d_output, window);
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

void run_all_test_vectors(bool window){
    for (int t = 0; t < NUM_TESTS; t++) {
        run_single_block_test(test_vectors[t].msg, test_vectors[t].expected, true, window);
    }
}

/* Function that tests consistency of SHA-256 for repeated hashing */
void run_consistency_test(bool window) {
    for (int t = 0; t < NUM_TESTS; t++) {
        const char* msg = test_vectors[t].msg;
        const char* expected = test_vectors[t].expected;
        bool all_pass = true;

        for (int repeat = 0; repeat < 5; repeat++) {
            bool pass = run_single_block_test(msg, expected, false, window);
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
 *  1. Generates N_BLOCKS of random input data.
 *  2. Builds the Merkle tree on the GPU using the naive solution.
 *  3. Computes the SHA-256 hash of each leaf on the CPU.
 *  4. Compares the CPU-computed hashes with the GPU-computed leaf hashes.
 * 
 * Reports mismatches if any, otherwise confirms all leaf hashes match.
 */
void test_naive_solution() {
    // generate bytes of data.
    uint8_t* host_data = generate_random_blocks(N_BLOCKS);
    // preparing host merkle tree.
    uint8_t* host_merkle_tree = (uint8_t*) malloc((2*N_BLOCKS - 1) * SHA256_OUTPUT_BLOCK_SIZE);
    // build the merkle tree on the GPU
    build_merkle_tree_naive(N_BLOCKS, host_merkle_tree);

    // compute the same on the CPU to compare
    bool correct = true;
    for (size_t i = 0; i < N_BLOCKS; i++) {
        uint8_t cpu_hash[SHA256_OUTPUT_BLOCK_SIZE];
        sha256_single_block_CPU(host_data + i*SHA256_INPUT_BLOCK_SIZE, cpu_hash, false);

        uint8_t* gpu_leaf = host_merkle_tree + (N_BLOCKS - 1 + i)*SHA256_OUTPUT_BLOCK_SIZE;

        if (memcmp(cpu_hash, gpu_leaf, SHA256_OUTPUT_BLOCK_SIZE) != 0) {
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
    free(host_merkle_tree);

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
    test_naive_solution();
    return 0;
}