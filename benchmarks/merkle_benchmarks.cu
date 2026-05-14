/*
* This file contains Merkle Tree building performance benchmarks divided into three groups:
*
* 1) CPU vs GPU naive:
*    Compares CPU execution against GPU (naive implementation)
*    to evaluate acceleration and scalability.
*
* 2) GPU naive vs GPU SMEM:
*    Compares two GPU implementations to evaluate the impact of
*    the SMEM optimization
*
* 3) GPU SMEM changing the size of leaves_per_block.
*
* It also contains Merkle Proof verification performance benchmarks:
*
* 1) CPU vs GPU naive:
*    Compares CPU execution against GPU (naive implementation)
*    to evaluate acceleration and scalability.
*
* Benchmarks are designed to isolate performance factors and provide
* meaningful comparisons.
*/

#include <iostream>
#include <cstring>
#include <stdint.h>
#include <vector>
#include <iomanip>
#include <inttypes.h>
#include <stdio.h>
#include <cassert>
#include "../data/data_generator.hpp"
#include "../merkle/utils_gpu.cuh"
#include "../merkle/utils_cpu.hpp"
#include "../merkle/naive_solution_build.cuh"
#include "../merkle/shared_mem_solution_build.cuh"
#include "../merkle/merkle_tree_cpu.hpp"
#include "../merkle/merkle_tree_cpu_openmp.hpp"
#include "../merkle/merkle_proof.cuh"
#include "bench_utils.hpp"

using namespace std;

/*
 * Benchmarks Merkle tree construction on CPU vs GPU across multiple sizes.
 *
 * For each size, the function performs several runs, measures execution time,
 * and reports average performance, variability, and relative speedup.
 *
 * Results are printed to stdout and saved as a CSV file.
 */
void build_merkle_tree_CPU_vs_GPU(int runs, const std::string& out_dir = "bench_output_files"){

    // preparing data to hash
    vector<size_t> leaf_lev_sizes = {/*4096,*/ 16384, 65536, 131072, 262144, 1048576, 4194304, 8388608};
    vector<uint8_t*> data_blocks_list;

    for (size_t n_blocks : leaf_lev_sizes) {
        uint8_t* data_blocks = generate_random_blocks(n_blocks);
        data_blocks_list.push_back(data_blocks);
    }

    // preparing for storing results
    vector<BenchResult> cpu_serial_results(leaf_lev_sizes.size());
    vector<BenchResult> cpu_parallel_results(leaf_lev_sizes.size());
    vector<BenchResult> gpu_results(leaf_lev_sizes.size());

    //gpu collecting results
    for(size_t i = 0; i < leaf_lev_sizes.size(); i++) {

        // GPU warmup 
        {
            uint64_t dummy = 0;
            MerkleTreeGPU* t = build_merkle_tree_naive(leaf_lev_sizes[i], data_blocks_list[i], &dummy);
            cudaDeviceSynchronize();
            merkle_tree_gpu_destroy(t);
        }

        vector<uint64_t> samples(runs);
        for (int r = 0; r < runs; r++) {
            uint64_t elapsed = 0;
            MerkleTreeGPU *merkle_tree_gpu = build_merkle_tree_naive(leaf_lev_sizes[i], data_blocks_list[i], &elapsed);
            merkle_tree_gpu_destroy(merkle_tree_gpu);
            samples[r] = elapsed;
        }
        gpu_results[i] = BenchResult::from_samples(samples);
    }

    // cpu serial collecting results
    for(size_t i = 0; i < leaf_lev_sizes.size(); i++) {
        vector<uint64_t> samples(runs);
        for (int r = 0; r < runs; r++) {
            uint64_t elapsed = 0;
            MerkleTreeCPU *merkle_tree_cpu = host_build_merkle_tree_serial(leaf_lev_sizes[i], data_blocks_list[i], SHA256_WINDOWED, &elapsed);
            merkle_tree_cpu_destroy(merkle_tree_cpu);
            samples[r] = elapsed;
        }
        cpu_serial_results[i] = BenchResult::from_samples(samples);
    }

    // CPU parallel warmup
    int omp_threads_used = 0;

    // cpu parallel collecting results
    for(size_t i = 0; i < leaf_lev_sizes.size(); i++) {

        // warmup per questo size specifico
        {
            MerkleTreeCPU* t = host_build_merkle_tree_parallel(
                leaf_lev_sizes[i], data_blocks_list[i],
                SHA256_WINDOWED, nullptr, nullptr);
            merkle_tree_cpu_destroy(t);
        }

        vector<uint64_t> samples(runs);
        for (int r = 0; r < runs; r++) {
            uint64_t elapsed = 0;
            // campiona i thread solo al primo run del primo size
            int* thread_out = (i == 0 && r == 0) ? &omp_threads_used : nullptr;
            MerkleTreeCPU* t = host_build_merkle_tree_parallel(leaf_lev_sizes[i], data_blocks_list[i], SHA256_WINDOWED, &elapsed, thread_out);
            merkle_tree_cpu_destroy(t);
            samples[r] = elapsed;
        }
        cpu_parallel_results[i] = BenchResult::from_samples(samples);
    }

    BenchmarkTable table(
        "build_merkle_tree_CPU_vs_GPU",
        {"size",
         "cpu_ser_ns", "cpu_ser_stddev", "cpu_ser_cv%",
         "omp_threads",
         "cpu_par_ns", "cpu_par_stddev", "cpu_par_cv%",
         "gpu_ns",     "gpu_stddev",     "gpu_cv%",
        }
    );

    for (size_t i = 0; i < leaf_lev_sizes.size(); i++) {
        std::ostringstream cpu_ser_std, cpu_ser_cv, cpu_par_std, cpu_par_cv, gpu_std, gpu_cv;
        cpu_ser_std << std::fixed << std::setprecision(2) << cpu_serial_results[i].stddev;
        cpu_ser_cv  << std::fixed << std::setprecision(2) << cpu_serial_results[i].cv;
        cpu_par_std << std::fixed << std::setprecision(2) << cpu_parallel_results[i].stddev;
        cpu_par_cv  << std::fixed << std::setprecision(2) << cpu_parallel_results[i].cv;
        gpu_std     << std::fixed << std::setprecision(2) << gpu_results[i].stddev;
        gpu_cv      << std::fixed << std::setprecision(2) << gpu_results[i].cv;

        table.add_row({
            std::to_string(leaf_lev_sizes[i]),
            std::to_string(cpu_serial_results[i].mean), cpu_ser_std.str(), cpu_ser_cv.str(),
            std::to_string(omp_threads_used),
            std::to_string(cpu_parallel_results[i].mean), cpu_par_std.str(), cpu_par_cv.str(),
            std::to_string(gpu_results[i].mean), gpu_std.str(), gpu_cv.str()
        });
    }

    table.dump();
    for (auto* p : data_blocks_list) free(p);
}

/*
 * Benchmarks GPU Merkle tree construction comparing naive and shared-memory implementations.
 *
 * For each input size, multiple runs are executed to measure execution time,
 * variability, and relative performance improvement of the shared-memory approach.
 *
 * Results are printed to stdout and saved as a CSV file.
 */
void build_merkle_tree_GPU_naive_vs_smem(int runs, int leaves_per_block, const std::string& out_dir = "bench_output_files"){

    // preparing data to hash
    vector<size_t> leaf_lev_sizes = {1024, 4096, 16384, 65536, 262144, 4194304, 8388608, 33554432};
    vector<uint8_t*> data_blocks_list;

    for (size_t n_blocks : leaf_lev_sizes) {
        uint8_t* data_blocks = generate_random_blocks(n_blocks);
        data_blocks_list.push_back(data_blocks);
    }

    // GPU warmup
    {
        uint64_t dummy = 0;
        MerkleTreeGPU* t = build_merkle_tree_naive(leaf_lev_sizes[0], data_blocks_list[0], &dummy);
        cudaDeviceSynchronize();
        merkle_tree_gpu_destroy(t);
    }

    // preparing for storing results
    vector<BenchResult> gpu_naive_results(leaf_lev_sizes.size());
    vector<BenchResult> gpu_smem_results(leaf_lev_sizes.size());

    // collecting results
    for(size_t i = 0; i < leaf_lev_sizes.size(); i++) {
        uint8_t* data = data_blocks_list[i];
        size_t   n = leaf_lev_sizes[i];

        // naive
        cudaDeviceReset();

        // warmup
        { uint64_t dummy = 0; auto* t = build_merkle_tree_naive(n, data, &dummy); merkle_tree_gpu_destroy(t); }
        { uint64_t dummy = 0; auto* t = build_merkle_tree_naive(n, data, &dummy); merkle_tree_gpu_destroy(t); }
        { uint64_t dummy = 0; auto* t = build_merkle_tree_naive(n, data, &dummy); merkle_tree_gpu_destroy(t); }
        cudaDeviceSynchronize();

        vector<uint64_t> naive_samples(runs);
        for (int r = 0; r < runs; r++) {
            uint64_t elapsed = 0;
            auto* t = build_merkle_tree_naive(n, data, &elapsed);
            merkle_tree_gpu_destroy(t);
            naive_samples[r] = elapsed;
        }
        gpu_naive_results[i] = BenchResult::from_samples(naive_samples);

        // smem 
        cudaDeviceReset();

        // warmup
        { uint64_t dummy = 0; auto* t = build_merkle_tree_SMEM(n, data, leaves_per_block, &dummy); merkle_tree_gpu_destroy(t); }
        { uint64_t dummy = 0; auto* t = build_merkle_tree_SMEM(n, data, leaves_per_block, &dummy); merkle_tree_gpu_destroy(t); }
        { uint64_t dummy = 0; auto* t = build_merkle_tree_SMEM(n, data, leaves_per_block, &dummy); merkle_tree_gpu_destroy(t); }
        cudaDeviceSynchronize();

        vector<uint64_t> smem_samples(runs);
        for (int r = 0; r < runs; r++) {
            uint64_t elapsed = 0;
            auto* t = build_merkle_tree_SMEM(n, data, leaves_per_block, &elapsed);
            merkle_tree_gpu_destroy(t);
            smem_samples[r] = elapsed;
        }
        gpu_smem_results[i] = BenchResult::from_samples(smem_samples);
    }

    BenchmarkTable table(
        "build_merkle_tree_GPU_naive_vs_smem",
        {"size", "gpu_naive_ns", "gpu_naive_stddev", "gpu_naive_cv%", "gpu_smem_ns", "gpu_smem_stddev", "gpu_smem_cv%", "variation%"}
    );

    for (size_t i = 0; i < leaf_lev_sizes.size(); i++) {
        double variation = (static_cast<double>(gpu_naive_results[i].mean) - gpu_smem_results[i].mean)
                           / gpu_naive_results[i].mean * 100.0;
        std::ostringstream var, gpu_naive_cv, gpu_smem_cv, naive_std, smem_std;
        var    << std::fixed << std::setprecision(2) << variation;
        gpu_naive_cv << std::fixed << std::setprecision(2) << gpu_naive_results[i].cv;
        gpu_smem_cv << std::fixed << std::setprecision(2) << gpu_smem_results[i].cv;
        naive_std << std::fixed << std::setprecision(2) << gpu_naive_results[i].stddev;
        smem_std << std::fixed << std::setprecision(2) << gpu_smem_results[i].stddev;

        table.add_row({
            std::to_string(leaf_lev_sizes[i]),
            std::to_string(gpu_naive_results[i].mean),
            naive_std.str(),
            gpu_naive_cv.str(),
            std::to_string(gpu_smem_results[i].mean),
            smem_std.str(),
            gpu_smem_cv.str(),
            var.str()
        });
    }

    table.dump();
    for (auto* p : data_blocks_list) free(p);
}

/* Benchmark to study the impact of the sizing of 'leves_per_block parameter */
void build_merkle_tree_GPU_smem_leaves_per_block(int runs, int n_leaves, const std::string& out_dir = "bench_output_files"){
    
    vector<size_t> leaves_per_block_sizes = {16, 32, 64, 128, 256, 512};
    uint8_t* data_blocks = generate_random_blocks(n_leaves);

    vector<BenchResult> gpu_smem_results(leaves_per_block_sizes.size());

    // collecting results
    for(size_t i = 0; i < leaves_per_block_sizes.size(); i++) {
        cudaDeviceReset();        

        vector<uint64_t> smem_samples(runs);
        for (int r = 0; r < runs; r++) {

            if(r == 0) {
                uint64_t dummy = 0; auto* t = build_merkle_tree_SMEM(n_leaves, data_blocks, leaves_per_block_sizes[i], &dummy); merkle_tree_gpu_destroy(t);
            }

            uint64_t elapsed = 0;
            auto* t = build_merkle_tree_SMEM(n_leaves, data_blocks, leaves_per_block_sizes[i], &elapsed);
            merkle_tree_gpu_destroy(t);
            smem_samples[r] = elapsed;
        }
        gpu_smem_results[i] = BenchResult::from_samples(smem_samples);
    }

    BenchmarkTable table(
        "build_merkle_tree_GPU_smem_" + std::to_string(n_leaves) + "_leaves",
        {"size", "leaves_per_block", "gpu_smem_ns", "gpu_smem_stddev", "gpu_smem_cv%"}
    );

    for (size_t i = 0; i < leaves_per_block_sizes.size(); i++) {

        std::ostringstream cv, stddev;
        cv << std::fixed << std::setprecision(2) << gpu_smem_results[i].cv;
        stddev << std::fixed << std::setprecision(2) << gpu_smem_results[i].stddev;

        table.add_row({
            std::to_string(n_leaves),
            std::to_string(leaves_per_block_sizes[i]),
            std::to_string(gpu_smem_results[i].mean),
            stddev.str(),
            cv.str()
        });
    }

    table.dump();
    free(data_blocks);
}

/*
 * Benchmarks Merkle proof verification on CPU vs GPU across multiple tree sizes.
 *
 * For each configuration, a batch of proofs is generated and verified multiple times
 * to measure execution time, variability, and relative performance difference.
 *
 * Results are printed to stdout and saved as a CSV file.
 */
void merkle_proof_CPU_vs_GPU(int runs, const std::string& out_dir = "bench_output_files"){

    // preparing data to hash
    vector<size_t> leaf_lev_sizes = {1024, 4096, 16384, 65536, 262144, 1048576/*, 4194304, 8388608, 33554432*/};
    vector<uint8_t*> data_blocks_list;

    for (size_t n_blocks : leaf_lev_sizes) {
        uint8_t* data_blocks = generate_random_blocks(n_blocks);
        data_blocks_list.push_back(data_blocks);
    }

    // preparing for storing results
    vector<BenchResult> cpu_serial_results(leaf_lev_sizes.size());
    vector<BenchResult> cpu_parallel_results(leaf_lev_sizes.size());
    vector<BenchResult> gpu_results(leaf_lev_sizes.size());

    // merkle trees building and proof
    MerkleTreeCPU* merkle_tree_cpu;
    MerkleTreeGPU* merkle_tree_gpu;
    ProofBatch* proof_batch;

    int omp_threads_used = 0;

    for(int i=0; i< leaf_lev_sizes.size(); i++) {

        // The number of proof for each merkle tree is three times its numer of leaves to ensures realistic tests
        size_t n_proofs = leaf_lev_sizes[i] * 3;
        float tamper_rate = 0.3; 

        merkle_tree_cpu = host_build_merkle_tree_serial(leaf_lev_sizes[i], data_blocks_list[i], SHA256_WINDOWED);
        merkle_tree_gpu = build_merkle_tree_naive(leaf_lev_sizes[i], data_blocks_list[i]);
        proof_batch = generate_proof_requests(data_blocks_list[i], leaf_lev_sizes[i], n_proofs, tamper_rate, DIST_ZIPF, 1.2);

        // GPU warmup
        {
            uint64_t dummy = 0;
            bool* result = compute_merkle_proofs(proof_batch, merkle_tree_gpu, &dummy);
            free(result);
        }

        // GPU benchmark
        vector<uint64_t> gpu_samples(runs);
        for(int run = 0; run < runs; run++){
            uint64_t elapsed = 0;
            bool* result = compute_merkle_proofs(proof_batch, merkle_tree_gpu, &elapsed);
            gpu_samples[run] = elapsed;
            assert(memcmp(proof_batch->expected, result, sizeof(bool) * n_proofs) == 0);
            free(result);
        }
        gpu_results[i] = BenchResult::from_samples(gpu_samples);
        
        cout << "GPU finita" << endl;
        
        // CPU serial benchmark
        vector<uint64_t> cpu_samples(runs);
        for(int run = 0; run < runs; run++){
            uint64_t elapsed = 0;
            bool* result = host_compute_merkle_proofs_serial(proof_batch, merkle_tree_cpu, SHA256_WINDOWED, &elapsed);
            cpu_samples[run] = elapsed;
            assert(memcmp(proof_batch->expected, result, sizeof(bool) * n_proofs) == 0);
            free(result);
        }
        cpu_serial_results[i] = BenchResult::from_samples(cpu_samples);

        cout << "CPU seriale finita" << endl;

        // CPU parallel warmup
        {
            bool* result = host_compute_merkle_proofs_parallel(proof_batch, merkle_tree_cpu, SHA256_WINDOWED, nullptr, &omp_threads_used);
            free(result);
        }

        // CPU parallel benchmark
        vector<uint64_t> cpu_parallel_samples(runs);
        for(int run = 0; run < runs; run++){
            uint64_t elapsed = 0;
            int* thread_out = (i == 0) ? &omp_threads_used : nullptr;
            bool* result = host_compute_merkle_proofs_parallel(proof_batch, merkle_tree_cpu, SHA256_WINDOWED, &elapsed, thread_out);
            cpu_parallel_samples[run] = elapsed;
            assert(memcmp(proof_batch->expected, result, sizeof(bool) * n_proofs) == 0);
            free(result);
        }
        cpu_parallel_results[i] = BenchResult::from_samples(cpu_parallel_samples);

        cout << "CPU parallela finita" << endl;

        merkle_tree_cpu_destroy(merkle_tree_cpu);
        merkle_tree_gpu_destroy(merkle_tree_gpu);
        free_proof_batch(proof_batch);

        cout << "######### Size: " << leaf_lev_sizes[i] << " finita. ###########" << endl;
    }

    BenchmarkTable table(
        "merkle_proof_CPU_vs_GPU",
        {"merkle_tree_size", "merkle_proofs_number",
         "cpu_ser_ns", "cpu_ser_stddev", "cpu_ser_cv%",
         "omp_threads",
         "cpu_par_ns", "cpu_par_stddev", "cpu_par_cv%",
         "gpu_ns", "gpu_stddev", "gpu_cv%"}
    );

    for (size_t i = 0; i < leaf_lev_sizes.size(); i++) {

        size_t n_proofs = leaf_lev_sizes[i] * 3;

        std::ostringstream cpu_ser_std, cpu_ser_cv, cpu_par_std, cpu_par_cv, gpu_std, gpu_cv;
        cpu_ser_std << std::fixed << std::setprecision(2) << cpu_serial_results[i].stddev;
        cpu_ser_cv  << std::fixed << std::setprecision(2) << cpu_serial_results[i].cv;
        cpu_par_std << std::fixed << std::setprecision(2) << cpu_parallel_results[i].stddev;
        cpu_par_cv  << std::fixed << std::setprecision(2) << cpu_parallel_results[i].cv;
        gpu_std     << std::fixed << std::setprecision(2) << gpu_results[i].stddev;
        gpu_cv      << std::fixed << std::setprecision(2) << gpu_results[i].cv;

        table.add_row({
            std::to_string(leaf_lev_sizes[i]),
            std::to_string(n_proofs),
            std::to_string(cpu_serial_results[i].mean),   cpu_ser_std.str(), cpu_ser_cv.str(),
            std::to_string(omp_threads_used),
            std::to_string(cpu_parallel_results[i].mean), cpu_par_std.str(), cpu_par_cv.str(),
            std::to_string(gpu_results[i].mean),          gpu_std.str(),     gpu_cv.str()
        });
    }

    table.dump();
    for (auto* p : data_blocks_list) free(p);

}

uint64_t profiling_merkle_build_naive_GPU(size_t n_leaves) {
    uint8_t* data_blocks = generate_random_blocks(n_leaves);
    uint64_t elapsed = 0;

    auto* t = build_merkle_tree_naive(n_leaves, data_blocks, &elapsed);
    merkle_tree_gpu_destroy(t);

    free(data_blocks);
    return elapsed;
}

uint64_t profiling_merkle_build_SMEM_GPU(size_t n_leaves, int leaves_per_block) {
    uint8_t* data_blocks = generate_random_blocks(n_leaves);
    uint64_t elapsed = 0;

    auto* t = build_merkle_tree_SMEM(n_leaves, data_blocks, leaves_per_block, &elapsed);
    merkle_tree_gpu_destroy(t);

    free(data_blocks);
    return elapsed;
}

int main() {
    //build_merkle_tree_CPU_vs_GPU(20);
    //build_merkle_tree_GPU_naive_vs_smem(20,256); // to do both with default sha256 and windowed

    //build_merkle_tree_GPU_smem_leaves_per_block(20, 65536); // 2^16 merkle tree leaves -> 2^16 + 2^16-1 merkle tree nodes
    //build_merkle_tree_GPU_smem_leaves_per_block(20, 262144); // 2^18 merkle tree leaves -> 2^18 + 2^18-1 merkle tree nodes
    //build_merkle_tree_GPU_smem_leaves_per_block(20, 4194304); // 2^22 merkle tree leaves -> 2^22 + 2^22-1 merkle tree nodes
    //build_merkle_tree_GPU_smem_leaves_per_block(20, 8388608); // 2^23 merkle tree leaves -> 2^23 + 2^23-1 merkle tree nodes
    //build_merkle_tree_GPU_smem_leaves_per_block(20, 33554432); // 2^25 merkle tree leaves -> 2^25 + 2^25-1 merkle tree nodes

    merkle_proof_CPU_vs_GPU(20);

    //uint64_t elapsed = profiling_merkle_build_naive_GPU(4194304);
    //uint64_t elapsed = profiling_merkle_build_SMEM_GPU(4194304, 256);
}