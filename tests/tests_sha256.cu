/*
 * test_sha256.cu
 *
 * This file contains a set of tests to verify the correctness of the SHA-256
 * implementation on CUDA. Each test hashes a single 512-bit block using
 * a kernel with 1D grid of 1D block with one thread, compares the result 
 * with the expected hash, and prints PASS/FAIL.
 */

#include <iostream>
#include <cstring>
#include <stdint.h>
#include "../sha256/sha256.cuh"

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
__global__ void single_test_sha256_kernel(const uint8_t* input, uint8_t* output) {
    sha256_single_block(input, output);
}

/* Utility function that converts a 32-byte hash to a hexadecimal string */
void hash_to_hex(const uint8_t hash[32], char hex[65]) {
    for (int i = 0; i < 32; i++)
        sprintf(hex + i*2, "%02x", hash[i]);
    hex[64] = 0;
}

__host__ void print_result(const char* msg, const char* expected, const char* actual) {
    std::cout << "Message: \"" << msg << "\"\n";
    if (strcmp(expected, actual) == 0)
        std::cout << "Result: PASS\n";
    else
        std::cout << "Result: FAIL\nExpected: " << expected << "\nGot     : " << actual << "\n";
    std::cout << "-----------------------------------\n";
}

/* Function that runs a single SHA-256 test */
bool run_single_block_test(const char* msg, const char* expected, bool print) {
    size_t len = strlen(msg);

    uint8_t h_block[64];
    sha256_pad_single_block(reinterpret_cast<const uint8_t*>(msg), len, h_block);

    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, 64);
    cudaMalloc(&d_output, 32);
    cudaMemcpy(d_input, h_block, 64, cudaMemcpyHostToDevice);

    single_test_sha256_kernel<<<1,1>>>(d_input, d_output);
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

void run_all_test_vectors(){
    for (int t = 0; t < NUM_TESTS; t++) {
        run_single_block_test(test_vectors[t].msg, test_vectors[t].expected, true);
    }
}

/* Function that tests consistency of SHA-256 for repeated hashing */
void run_consistency_test() {
    for (int t = 0; t < NUM_TESTS; t++) {
        const char* msg = test_vectors[t].msg;
        const char* expected = test_vectors[t].expected;
        bool all_pass = true;

        for (int repeat = 0; repeat < 5; repeat++) {
            bool pass = run_single_block_test(msg, expected, false);
            if (!pass) {
                all_pass = false; // segna che c'è stato un fallimento
            }
        }

        // Stampa il risultato finale dopo tutti i tentativi
        if (all_pass)
            std::cout << "Message: \"" << msg << "\" -> PASS (all repetitions)\n";
        else
            std::cout << "Message: \"" << msg << "\" -> FAIL (some repetition failed)\n";

        std::cout << "-----------------------------------\n";
    }
}

int main() {
    std::cout << "================ Single Block SHA-256 Tests ================\n";
    run_all_test_vectors();
    std::cout << "\n================ Consistency Test ================\n";
    run_consistency_test();
    return 0;
}