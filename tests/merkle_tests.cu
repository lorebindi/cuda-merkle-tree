/*
* 
* This file contains all the set of tests used to verify the correctness of the merkle building
* process (both naive and SMEM solution).
*
*/

#include <iostream>
#include <cstring>
#include <stdint.h>
#include <vector>
#include "../sha256/sha256_GPU.cuh"
#include "../sha256/sha256_CPU.hpp"
#include "../merkle/utils.cuh"
#include "../merkle/naive_solution_build.cuh"
#include "../merkle/shared_mem_solution_build.cuh"
#include "../merkle/merkle_tree_cpu.hpp"
#include "../merkle/merkle_proof.cuh"
#include "../data/data_generator.hpp"

using namespace std;

#define THREADS_PER_BLOCK 256

enum MerkleTestMode {
    ROOT_ONLY,
    FULL_TREE
};


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
bool test_naive_solution(size_t n_blocks) {
    // generate bytes of data.
    cout << "Data blocks (leaves) number: " << n_blocks << "\n" << endl;
    uint8_t* host_data = generate_random_blocks(n_blocks);
   
    // build the merkle tree on the GPU
    MerkleTreeGPU* merkle_tree_gpu = build_merkle_tree_naive(n_blocks, host_data);

    // preparing host merkle tree.
    size_t merkle_tree_size = merkle_tree_gpu->base.size;
    size_t leaf_offset = merkle_tree_gpu->base.size - merkle_tree_gpu->base.n_leaves;
    cout << "Merkle tree size: " << merkle_tree_size << "\n" << endl;
    uint8_t* host_merkle_tree = (uint8_t*) malloc(merkle_tree_size * SHA256_OUTPUT_BLOCK_SIZE);

    gpuErrchk(cudaMemcpy(host_merkle_tree, merkle_tree_gpu->dev_tree,(merkle_tree_size)*SHA256_OUTPUT_BLOCK_SIZE,
            cudaMemcpyDeviceToHost));

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
        free(host_data);
        free(curr_lev);
        free(host_merkle_tree);
        merkle_tree_gpu_destroy(merkle_tree_gpu);
        return false;
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

            sha256_single_block_CPU(concatenated, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, SHA256_WINDOWED);
        }

        free(prec_lev);
        prec_lev = curr_lev;
        prec_lev_size = curr_lev_size;
    }

    bool outcome = false;

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

    merkle_tree_gpu_destroy(merkle_tree_gpu);

    return outcome;
}

bool run_all_merkle_tests_naive() {

    cout << "\n================ MERKLE TREE NAIVE SOLUTION TEST SUITE ================\n";
    vector<string> failed_tests;

    auto run_test = [&](size_t n_blocks, const string& desc) {
        cout << "\n[TEST] " << desc << " n_blocks = " << n_blocks << "\n";
        try {
            bool passed = test_naive_solution(n_blocks);
            if (!passed)
                failed_tests.push_back(desc + " (n_blocks=" + to_string(n_blocks) + ")");
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
bool test_SMEM_solution(size_t n_blocks, int leaves_per_block, MerkleTestMode mode) {
    // generate bytes of data.
    cout << "Data blocks (leaves) number: " << n_blocks << "\n" << endl;
    uint8_t* host_data = generate_random_blocks(n_blocks);

    cout << "Number of leaves per block: " << leaves_per_block << "\n" << endl;

  

    // build the merkle tree on the GPU
    MerkleTreeGPU* merkle_tree_gpu = build_merkle_tree_SMEM(n_blocks, host_data, leaves_per_block);

    cout << "Merkle tree size: " << merkle_tree_gpu->base.size << "\n" << endl;
    uint8_t* host_merkle_tree = (uint8_t*) malloc(merkle_tree_gpu->base.size * SHA256_OUTPUT_BLOCK_SIZE);

    gpuErrchk(cudaMemcpy(host_merkle_tree, merkle_tree_gpu->dev_tree,(merkle_tree_gpu->base.size)*SHA256_OUTPUT_BLOCK_SIZE,
            cudaMemcpyDeviceToHost));

    size_t merkle_tree_size = merkle_tree_gpu->base.size;
    size_t leaf_offset = merkle_tree_gpu->base.size - merkle_tree_gpu->base.n_leaves;

    
    std::vector<size_t> level_sizes;
    std::vector<size_t> level_offsets;

    compute_merkle_levels_layout(n_blocks, leaf_offset, level_sizes, level_offsets
    );

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
        sha256_single_block_CPU(host_data + i*SHA256_INPUT_BLOCK_SIZE, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, SHA256_WINDOWED);
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

                sha256_single_block_CPU(concatenated, curr_lev + i*SHA256_OUTPUT_BLOCK_SIZE, SHA256_WINDOWED);
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
        // building the entire merkle tree on host side
        MerkleTreeCPU* cpu_tree = host_build_merkle_tree(n_blocks, host_data, SHA256_WINDOWED);
        
        // comparing the merkle tree computed in the host side with the one computed 
        // in gpu side
        bool all_match = memcmp(cpu_tree->host_tree, host_merkle_tree,
                                merkle_tree_size * SHA256_OUTPUT_BLOCK_SIZE) == 0;
        
        merkle_tree_cpu_destroy(cpu_tree);
        free(host_merkle_tree);

        if (all_match) {
            cout << "Full tree MATCH\n" << endl;
            outcome = true;
        } else {
            cout << "Some nodes mismatch!\n" << endl;
            outcome = false;
        }
    }

    free(host_data);

    merkle_tree_gpu_destroy(merkle_tree_gpu);
    return outcome;    
}

bool run_all_merkle_tests_SMEM(MerkleTestMode mode_small_size) {
 cout << "\n================ MERKLE TREE TEST SUITE ================\n";
    vector<string> failed_tests;

    auto run_test = [&](size_t n_blocks, int leaves_per_block, MerkleTestMode mode, const string& desc) {
        cout << "\n[TEST] " << desc << " n_blocks = " << n_blocks
             << ", leaves_per_block = " << leaves_per_block << "\n";
        bool passed = test_SMEM_solution(n_blocks, leaves_per_block, mode);
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
        size_t merkle_tree_leaves = rand() % 5000 + 1;
        int leaves_per_block = 1 << (rand() % 9 + 1); // 2–512
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

bool test_merkle_proof(size_t n_blocks, size_t n_proofs, float tamper_rate, bool smem, ProofDistribution distribution, double zipf_s, bool check_with_cpu){

    uint8_t* host_data_blocks = generate_random_blocks(n_blocks);

    ProofBatch* proof_batch = generate_proof_requests(host_data_blocks, n_blocks, n_proofs, tamper_rate, distribution, zipf_s);
    
    // building the merkle tree
    int leaves_per_block = 0;
    MerkleTreeGPU* merkle_tree_gpu = NULL;

    if(smem) {
        leaves_per_block = compute_optimal_leaves_per_block(n_blocks, THREADS_PER_BLOCK);
        merkle_tree_gpu = build_merkle_tree_SMEM(n_blocks, host_data_blocks, leaves_per_block);
    }
    else{
        merkle_tree_gpu = build_merkle_tree_naive(n_blocks, host_data_blocks);
    }

    bool* gpu_result = compute_merkle_proofs (proof_batch, merkle_tree_gpu);

    bool outcome = false;

    // Check against the work of the CPU side
    MerkleTreeCPU* merkle_tree_cpu = NULL;
    bool* cpu_result = NULL;

    if (check_with_cpu) {
        merkle_tree_cpu = host_build_merkle_tree(n_blocks, host_data_blocks, SHA256_WINDOWED);
        cpu_result = host_compute_merkle_proofs(proof_batch, merkle_tree_cpu, SHA256_WINDOWED);

        int gpu_vs_expected = memcmp(proof_batch->expected, gpu_result, sizeof(bool) * n_proofs);
        int cpu_vs_expected = memcmp(proof_batch->expected, cpu_result, sizeof(bool) * n_proofs);

        if(gpu_vs_expected == 0 && cpu_vs_expected == 0){
                cout << "Merkle proof both in the GPU and CPU computed correctly" << endl;
                outcome = true;
                merkle_tree_cpu_destroy(merkle_tree_cpu);
                free(cpu_result);
            }
        else{
            
            if(gpu_vs_expected != 0 && cpu_vs_expected == 0)
                cout << "Some mistakes in the GPU computed merkle proof." << endl;
            
            if(gpu_vs_expected == 0 && cpu_vs_expected != 0)
                cout << "Some mistakes in the CPU computed merkle proof." << endl;
            
            if(gpu_vs_expected != 0 && cpu_vs_expected != 0)
                cout << "Some mistakes both in the GPU and CPU computed merkle proof." << endl;
                
            outcome = false;
            
        }
    }
    else{
        if(memcmp(proof_batch->expected, gpu_result, sizeof(bool) * n_proofs) == 0){
            cout << "Merkle proof in the GPU computed correctly" << endl;
            outcome = true;
        }
        else{
            cout << "Some mistakes in the GPU computed merkle proof." << endl;
            outcome = false;
        }
    }


    free(host_data_blocks);
    free(gpu_result);
    free_proof_batch(proof_batch);
    merkle_tree_gpu_destroy(merkle_tree_gpu);

    return outcome;
    
}

bool run_all_merkle_proof_tests(bool smem, ProofDistribution distribution, double zipf_s) {
    cout << "\n================ MERKLE PROOF TEST SUITE ================\n";
    vector<string> failed_tests;

    auto run_test = [&](size_t n_blocks, size_t n_proofs, float tamper_rate, bool check_with_cpu, const string& desc) {
        cout << "\n[TEST] " << desc
             << " | n_blocks=" << n_blocks
             << " | n_proofs=" << n_proofs
             << " | tamper_rate=" << tamper_rate << "\n";
        bool passed = test_merkle_proof(n_blocks, n_proofs, tamper_rate, smem, distribution, zipf_s, check_with_cpu);
        if (!passed)
            failed_tests.push_back(desc + " (n_blocks=" + to_string(n_blocks) +
                                   ", n_proofs=" + to_string(n_proofs) + ")");
    };

    auto rand_tamper = [&]() { return (float)(rand() % 101) / 100.0f; };

    // --- EDGE CASES ---
    run_test(1,  1,  0.0f, true, "Single block, single proof, no tamper");
    run_test(1,  1,  1.0f, true, "Single block, single proof, all tamper");
    run_test(2,  2,  0.0f, true, "Two blocks, all valid");
    run_test(2,  2,  1.0f, true, "Two blocks, all tampered");

    // --- n_proofs < n_blocks ---
    vector<size_t> small_blocks = {5, 13, 17, 32, 64};
    for (auto n : small_blocks) {
        size_t n_proofs = max((size_t)1, n / 2);
        run_test(n, n_proofs, rand_tamper(), true, "n_proofs < n_blocks");
    }

    // --- n_proofs == n_blocks ---
    vector<size_t> medium_blocks = {8, 16, 31, 33, 100};
    for (auto n : medium_blocks)
        run_test(n, n, rand_tamper(), true, "n_proofs == n_blocks");

       
    // --- n_proofs > n_blocks ---
    vector<size_t> large_blocks = {10, 50, 128, 257, 1000};
    for (auto n : large_blocks) {
        size_t n_proofs = n * 3;
        run_test(n, n_proofs, rand_tamper(), false, "n_proofs > n_blocks");
    }

    // --- MIXED TAMPER RATE ---
    vector<size_t> mixed_blocks = {500, 1000, 5000, 10000};
    for (auto n : mixed_blocks)
        run_test(n, n * 2, 0.5f, false, "Mixed tamper rate (50%)");

    // --- POWER OF TWO EDGE ---
    vector<size_t> pow2 = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (auto n : pow2)
        run_test(n, n, rand_tamper(), false, "Power-of-two n_blocks");

    // --- RANDOM STRESS ---
    for (int i = 0; i < 10; i++) {
        size_t n_blocks = rand() % 2000 + 1;
        size_t n_proofs = rand() % (n_blocks * 2) + 1;
        run_test(n_blocks, n_proofs, rand_tamper(), false, "Random stress test");
    }

    // --- SUMMARY ---
    cout << "\n================ MERKLE PROOF TEST SUMMARY ================\n";
    if (failed_tests.empty()) {
        cout << "All tests passed!\n";
        cout << "================ END MERKLE PROOF TESTS ================\n";
        return true;
    } else {
        cout << "Some tests failed:\n";
        for (auto& s : failed_tests) cout << "- " << s << "\n";
        cout << "================ END MERKLE PROOF TESTS ================\n";
        return false;
    }
}

int main() {
    srand(time(NULL));

    const bool use_smem = true;
    
    // merkle tree building tests naive solution
    bool outcome1 = run_all_merkle_tests_naive();
    
    // merkle tree building tests SMEM solution
    bool outcome2 = run_all_merkle_tests_SMEM(ROOT_ONLY);

    bool outcome3 = run_all_merkle_proof_tests(use_smem, DIST_ZIPF, 1.0 );

    cout << "\n\n#####################################################################\n\n";
    cout << "================ MERKLE TESTS SUMMARY ====================\n";
    if(outcome1 && outcome2 && outcome3) {
        cout << "All tests PASSED!\n";
    }
    else{
        cout << "Some test FAILED!\n";
    }
    cout << "================ END MERKLE TESTS SUMMARY ================\n\n";    
    
    cudaDeviceReset();

    return 0;

    /*
    ******************IMPORTANTE*************************+
    Trovare il modo di testare 'host_compute_merkle_proofs' prima del testing delle prestazioni
    */

}