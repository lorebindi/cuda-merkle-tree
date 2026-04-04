/*
* This file contains all the set of tests used to verify the correctness of SHA-256 implementation on CUDA. 
*
*/

#include <iostream>
#include <cstring>
#include <stdint.h>
#include <vector>
#include "../sha256/sha256_GPU.cuh"
#include "../sha256/sha256_CPU.hpp"
#include "../merkle/utils.cuh"
#include "../data/data_generator.hpp"

#define CUDA_CHECK(call) { cudaError_t e = call; if(e != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(e) << " at line " << __LINE__ << "\n"; exit(1); }}


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

bool run_all_test_vectors(bool sha256_windowed){
    bool outcome = true;
    for (int t = 0; t < NUM_TESTS; t++) {
        bool temp = run_single_block_test(test_vectors[t].msg, test_vectors[t].expected, true, sha256_windowed);
        if (!temp)
            outcome = false;
    }
    return outcome;
}

/* Function that tests consistency of SHA-256 for repeated hashing */
bool run_consistency_test(bool sha256_windowed) {
    bool outcome = true;
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
        if (all_pass) {
            cout << "Message: \"" << msg << "\" -> PASS (all repetitions)\n";
        }
        else {
            cout << "Message: \"" << msg << "\" -> FAIL (some repetition failed)\n";
            outcome = false;
        }
            

        cout << "-----------------------------------\n";
    }
    return outcome;
}

bool run_cpu_gpu_consistency_test(bool windowed) {
    bool outcome = true;

    for (int t = 0; t < NUM_TESTS; t++) {
        const char* msg = test_vectors[t].msg;
        size_t len = strlen(msg);

        // --- Padding (comune a CPU e GPU) ---
        uint8_t h_block[SHA256_INPUT_BLOCK_SIZE];
        sha256_pad_single_block(reinterpret_cast<const uint8_t*>(msg), len, h_block);

        // --- CPU ---
        uint8_t cpu_hash[SHA256_OUTPUT_BLOCK_SIZE];
        sha256_single_block_CPU(h_block, cpu_hash, windowed);

        // --- GPU ---
        uint8_t *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input,  SHA256_INPUT_BLOCK_SIZE));
        CUDA_CHECK(cudaMalloc(&d_output, SHA256_OUTPUT_BLOCK_SIZE));
        CUDA_CHECK(cudaMemcpy(d_input, h_block, SHA256_INPUT_BLOCK_SIZE, cudaMemcpyHostToDevice));

        single_test_sha256_kernel<<<1,1>>>(d_input, d_output, windowed);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint8_t gpu_hash[SHA256_OUTPUT_BLOCK_SIZE];
        CUDA_CHECK(cudaMemcpy(gpu_hash, d_output, SHA256_OUTPUT_BLOCK_SIZE, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);

        // --- byte compare ---
        bool match = (memcmp(cpu_hash, gpu_hash, SHA256_OUTPUT_BLOCK_SIZE) == 0);

        char cpu_hex[65], gpu_hex[65];
        hash_to_hex(cpu_hash, cpu_hex);
        hash_to_hex(gpu_hash, gpu_hex);

        cout << "Message: \"" << msg << "\"\n";
        if (match) {
            cout << "CPU vs GPU: PASS (" << cpu_hex << ")\n";
        } else {
            cout << "CPU vs GPU: FAIL\n";
            cout << "  CPU: " << cpu_hex << "\n";
            cout << "  GPU: " << gpu_hex << "\n";
            outcome = false;
        }
        cout << "-----------------------------------\n";
    }

    return outcome;
}


int main() {
    srand(time(NULL));

    bool outcome = false;

    cout << "================ Single Block SHA-256 Tests (traditional transform) ================\n";
    bool outcome1 = run_all_test_vectors(false);
    cout << "================ Single Block SHA-256 Tests (windowed transform) ================\n";
    bool outcome2 = run_all_test_vectors(true);
    cout << "\n================ Consistency Test (traditional transform) ================\n";
    bool outcome3 = run_consistency_test(false);
    cout << "\n================ Consistency Test (windowed transform) ================\n";
    bool outcome4 = run_consistency_test(true);
    cout << "\n================ CPU vs GPU Test (traditional transform) ================\n";
    bool outcome5 = run_cpu_gpu_consistency_test(false);
    cout << "\n================ CPU vs GPU Test (windowed transform) ================\n";
    bool outcome6 = run_cpu_gpu_consistency_test(true);

    cout << "\n\n#####################################################################\n\n";
    cout << "================ SHA256 TESTS SUMMARY ====================\n";
    if(outcome1 && outcome2 && outcome3 && outcome4 && outcome5 && outcome6) {
        cout << "All tests PASSED!\n";
    }
    else{
        cout << "Some test FAILED!\n";
    }
    cout << "================ END SHA256 TESTS SUMMARY =================\n\n";

    
    
    cudaDeviceReset();

    return 0;
}