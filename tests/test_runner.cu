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
#include <vector>
#include "../sha256/sha256_GPU.cuh"
#include "../sha256/sha256_CPU.hpp"
#include "../merkle/utils.cuh"
#include "../merkle/naive_solution.cuh"
#include "../merkle/shared_mem_solution.cuh"
#include "../data/data_generator.hpp"

using namespace std;

enum MerkleTestMode {
    ROOT_ONLY,
    FULL_TREE
};

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

            sha256_single_block_CPU(concatenated, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, sha256_windowed);
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

bool run_all_merkle_tests_naive(bool sha256_windowed) {

    cout << "\n================ MERKLE TREE NAIVE SOLUTION TEST SUITE ================\n";
    vector<string> failed_tests;

    auto run_test = [&](size_t n_blocks, const string& desc) {
        cout << "\n[TEST] " << desc << " n_blocks = " << n_blocks << "\n";
        try {
            test_naive_solution(n_blocks, sha256_windowed);
        } catch (...) {
            cout << "Test failed due to an exception!\n";
            failed_tests.push_back(desc + " (n_blocks=" + to_string(n_blocks) + ")");
        }
    };

    // --- SMALL TESTS ---
    vector<size_t> small_sizes = {1, 2, 4, 8, 16};
    for (auto n : small_sizes) run_test(n, "Small test");

    // --- MEDIUM TESTS ---
    vector<size_t> medium_sizes = {100, 500, 1000};
    for (auto n : medium_sizes) run_test(n, "Medium test");

    // --- POWER-OF-TWO EDGE ---
    vector<size_t> pow2_edges = {31, 32, 33, 1023, 1024};
    for (auto n : pow2_edges) run_test(n, "Power-of-two edge test");

    // --- RANDOM STRESS ---
    for (int i = 0; i < 5; i++) {
        size_t n_blocks = rand() % 2000 + 1;
        run_test(n_blocks, "Random stress test");
    }

    // --- SUMMARY ---
    cout << "\n================ NAIVE SOLUTION TEST SUMMARY ================\n";
    if (failed_tests.empty()) {
        cout << "All tests passed!\n";
        return true;
    } else {
        cout << "Some tests failed:\n";
        for (auto& s : failed_tests) cout << "- " << s << "\n";
        return false;
    }
    cout << "================ END TESTS ================\n";
}

void compute_merkle_levels_layout(size_t n_leaves,
                                  size_t leaf_offset,
                                  std::vector<size_t>& level_sizes,
                                  std::vector<size_t>& level_offsets) {
    
    level_sizes.clear();
    level_offsets.clear();

    size_t curr_size = n_leaves;
    size_t curr_offset = leaf_offset;

    // livello foglie
    level_sizes.push_back(curr_size);
    level_offsets.push_back(curr_offset);

    // livelli superiori
    while (curr_size > 1) {
        size_t parent_size = (curr_size + 1) / 2;
        curr_offset -= parent_size;

        level_sizes.push_back(parent_size);
        level_offsets.push_back(curr_offset);

        curr_size = parent_size;
    }
}

/*
 * Test function for the SMEM optimized Merkle tree implementation.
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
bool test_SMEM_solution(size_t n_blocks, int leaves_per_block, bool sha256_windowed, MerkleTestMode mode) {
    // generate bytes of data.
    cout << "Data blocks (leaves) number: " << n_blocks << "\n" << endl;
    uint8_t* host_data = generate_random_blocks(n_blocks);

    cout << "Number of leaves per block: " << leaves_per_block << "\n" << endl;

     // preparing host merkle tree.
    size_t merkle_tree_size = compute_merkle_tree_size(n_blocks);
    size_t leaf_offset = merkle_tree_size - n_blocks;

    std::vector<size_t> level_sizes;
    std::vector<size_t> level_offsets;

    compute_merkle_levels_layout(n_blocks, leaf_offset, level_sizes, level_offsets
    );


    cout << "Merkle tree size: " << merkle_tree_size << "\n" << endl;
    uint8_t* host_merkle_tree = (uint8_t*) malloc(merkle_tree_size * SHA256_OUTPUT_BLOCK_SIZE);
    // build the merkle tree on the GPU
    build_merkle_tree_SMEM(n_blocks, host_data, host_merkle_tree, leaves_per_block);

    cout << "GPU Merkle Tree computed. \n" << endl;

    if(mode == FULL_TREE) {
        cout << "\nGPU Merkle tree heap (linear):\n";
        for (size_t i = 0; i < merkle_tree_size; i++) {
            printf("Node %zu: ", i);
            for (int j = 0; j < SHA256_OUTPUT_BLOCK_SIZE; j++) {
                printf("%02x", host_merkle_tree[i * SHA256_OUTPUT_BLOCK_SIZE + j]);
            }
            printf("\n");
        }
    }

    // leafs verification
    bool correct = true;
    uint8_t* curr_lev = (uint8_t*) malloc(n_blocks * SHA256_OUTPUT_BLOCK_SIZE);
    for (size_t i = 0; i < n_blocks; i++) {
        // computing CPU hash of the i-th leaf
        sha256_single_block_CPU(host_data + i*SHA256_INPUT_BLOCK_SIZE, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, sha256_windowed);
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
        return false;
    }

    free(host_data);

    // root verification

    bool outcome = false;
    
    if(mode == ROOT_ONLY) {
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

                sha256_single_block_CPU(concatenated, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, sha256_windowed);
            }

            free(prec_lev);
            prec_lev = curr_lev;
            prec_lev_size = curr_lev_size;
        }

        // compare
        if (memcmp(prec_lev, host_merkle_tree, SHA256_OUTPUT_BLOCK_SIZE) != 0){
            cout << "Roots mismatch" << endl;
            outcome = false;
        } 
        else{
            cout << "Roots MATCH\n" << endl;
            outcome = true;
        } 

        free(host_merkle_tree);
        free(prec_lev);

    }
    else {
        bool all_levels_match = true;

        // CPU buffer iniziale = foglie
        uint8_t* cpu_prev = curr_lev;
        size_t cpu_prev_size = n_blocks;

        for (size_t level = 1; level < level_sizes.size(); level++) {

            size_t cpu_curr_size = (cpu_prev_size + 1) / 2;
            uint8_t* cpu_curr = (uint8_t*) malloc(cpu_curr_size * SHA256_OUTPUT_BLOCK_SIZE);

            // costruzione livello CPU
            for (size_t i = 0; i < cpu_curr_size; i++) {
                uint8_t* left = cpu_prev + (2*i)*SHA256_OUTPUT_BLOCK_SIZE;
                uint8_t* right = cpu_prev + (2*i+1)*SHA256_OUTPUT_BLOCK_SIZE;

                if ((cpu_prev_size % 2 == 1) && (i == cpu_curr_size - 1))
                    right = left;

                uint8_t concatenated[64];
                memcpy(concatenated, left, SHA256_OUTPUT_BLOCK_SIZE);
                memcpy(concatenated + SHA256_OUTPUT_BLOCK_SIZE, right, SHA256_OUTPUT_BLOCK_SIZE);

                sha256_single_block_CPU(concatenated, cpu_curr + i*SHA256_OUTPUT_BLOCK_SIZE, sha256_windowed);
            }

            // GPU livello corrispondente
            size_t gpu_offset = level_offsets[level];
            uint8_t* gpu_level = host_merkle_tree + gpu_offset * SHA256_OUTPUT_BLOCK_SIZE;

            // confronto
            for (size_t i = 0; i < cpu_curr_size; i++) {
                uint8_t* cpu_node = cpu_curr + i*SHA256_OUTPUT_BLOCK_SIZE;
                uint8_t* gpu_node = gpu_level + i*SHA256_OUTPUT_BLOCK_SIZE;

                if (memcmp(cpu_node, gpu_node, SHA256_OUTPUT_BLOCK_SIZE) != 0) {
                    all_levels_match = false;
                }
            }

            free(cpu_prev);
            cpu_prev = cpu_curr;
            cpu_prev_size = cpu_curr_size;
        }

        if (all_levels_match) {
            cout << "\nRoots MATCH\n" << endl;
            outcome = true;
        }
        else{
            cout << "\nSome nodes mismatch!\n" << endl;
            outcome = false;
        }
            

    }

    return outcome;    
}

bool run_all_merkle_tests_SMEM(MerkleTestMode mode_small_size) {
 cout << "\n================ MERKLE TREE TEST SUITE ================\n";
    vector<string> failed_tests;

    auto run_test = [&](size_t n_blocks, int leaves_per_block, MerkleTestMode mode, const string& desc) {
        cout << "\n[TEST] " << desc << " n_blocks = " << n_blocks
             << ", leaves_per_block = " << leaves_per_block << "\n";
        bool passed = test_SMEM_solution(n_blocks, leaves_per_block, true, mode);
        if (!passed) {
            failed_tests.push_back(desc + " (n_blocks=" + to_string(n_blocks) +
                                     ", leaves_per_block=" + to_string(leaves_per_block) + ")");
        }
    };

    // --- SMALL TESTS ---
    vector<size_t> small_sizes = {1, 2, 4, 8, 16};
    for (auto n : small_sizes) run_test(n, 8, mode_small_size, "Small test");

    // --- MEDIUM TESTS ---
    vector<size_t> medium_sizes = {100, 1000};
    for (auto n : medium_sizes) run_test(n, 32, ROOT_ONLY, "Medium test");

    // --- POWER-OF-TWO EDGE TESTS ---
    vector<size_t> pow2_edges = {31, 32, 33, 1023, 1024};
    for (auto n : pow2_edges) run_test(n, 32, ROOT_ONLY, "Power-of-two edge test");

    // --- RANDOM STRESS TEST ---
    for (int i = 0; i < 5; i++) {
        size_t merkle_tree_leaves = rand() % 2000 + 1;
        int leaves_per_block = 1 << (rand() % 6 + 1); // 2–64
        run_test(merkle_tree_leaves, leaves_per_block, ROOT_ONLY, "Random stress test");
    }

    // --- SUMMARY ---
    cout << "\n================ TEST SMEM SOLUTION SUMMARY ================\n";
    if (failed_tests.empty()) {
        cout << "All tests passed!\n";
        cout << "================ END TEST SMEM SOLUTION ================\n";
        return true;
    } else {
        cout << "Some tests failed:\n";
        for (auto& s : failed_tests) cout << "- " << s << "\n";
        cout << "================ END TEST SMEM SOLUTION ================\n";
        return false;
    }
    
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
    
    // merkle tree building tests naive solution
    bool outcome5 = run_all_merkle_tests_naive(true);
    
    // merkle tree building tests SMEM solution
    bool outcome6 = run_all_merkle_tests_SMEM(ROOT_ONLY);

    cout << "\n\n#####################################################################\n\n";
    cout << "================ TESTS SUMMARY ====================\n";
    if(outcome1 && outcome2 && outcome3 && outcome4 && outcome5 && outcome6) {
        cout << "All tests PASSED!\n";
    }
    else{
        cout << "Some test FAILED!\n";
    }
    cout << "================ END TESTS SUMMARY ================\n\n";

    
    
    cudaDeviceReset();

    return 0;
}