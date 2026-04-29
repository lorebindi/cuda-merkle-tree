/*
* This file contains SHA-256 performance benchmarks divided into two groups:
*
* 1) CPU vs GPU:
*    Compares CPU execution against GPU (windowed implementation)
*    to evaluate acceleration and scalability.
*
* 2) GPU naive vs GPU windowed:
*    Compares two GPU implementations to evaluate the impact of
*    the windowed optimization, which reduces the message schedule
*    from 64 to 16 uint32_t elements reused.
*
* Benchmarks are designed to isolate performance factors and provide
* meaningful comparisons.
*/

#include <iostream>
#include <cstring>
#include <stdint.h>
#include <vector>
#include <iomanip>
#include <cmath>
#include "../data/data_generator.hpp"
#include "../merkle/utils.cuh"
#include "../sha256/sha256_CPU.hpp"
#include "bench_utils.hpp"

using namespace std;

/* Returns the total time necessary to hashing each block of the array provided */
uint64_t sha256_cpu_array_benchmark(const uint8_t* data, size_t n_blocks, bool sha256_windowed) {

    uint8_t* hashed_data = (uint8_t*) malloc(SHA256_OUTPUT_BLOCK_SIZE * n_blocks);

    uint64_t start = current_time_nsecs();

    for (size_t i = 0; i < n_blocks; i++)
        sha256_single_block_CPU(data + i*SHA256_INPUT_BLOCK_SIZE, hashed_data + i*SHA256_OUTPUT_BLOCK_SIZE, sha256_windowed);

    uint64_t end = current_time_nsecs();

    free(hashed_data);
    return end - start;
}

/* Compute the hashing of an entire array of data blocks on the device and returns the total time
   (loading data + computing) */
uint64_t sha256_gpu_array_benchmark(uint8_t* host_data, uint8_t* dev_data, uint8_t* dev_hashed_data, size_t n_blocks, const bool sha256_windowed) {

    uint64_t initial_time = current_time_nsecs();

    // memcpy dentro la misura: è parte del costo reale
    gpuErrchk(cudaMemcpy(dev_data, host_data,
              n_blocks * SHA256_INPUT_BLOCK_SIZE,
              cudaMemcpyHostToDevice));

    int blocks_per_grid = (n_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if(sha256_windowed){
        leaf_level_build<true><<<blocks_per_grid, THREADS_PER_BLOCK>>>(
        n_blocks, 0, dev_data, dev_hashed_data);
    }
    else{
        leaf_level_build<false><<<blocks_per_grid, THREADS_PER_BLOCK>>>(
        n_blocks, 0, dev_data, dev_hashed_data);
    }
        
    cudaDeviceSynchronize();

    return current_time_nsecs() - initial_time;
}

/* Collects 'runs' samples for the CPU benchmark and returns a BenchResult */
BenchResult collect_cpu(const uint8_t* data, size_t n_blocks, bool windowed, int runs) {
    vector<uint64_t> samples(runs);
    for (int r = 0; r < runs; r++)
        samples[r] = sha256_cpu_array_benchmark(data, n_blocks, windowed);
    return BenchResult::from_samples(samples);
}

/* Collects 'runs' samples for the GPU benchmark and returns a BenchResult */
BenchResult collect_gpu(uint8_t* host_data, size_t n_blocks, bool windowed, int runs) {

    uint8_t *dev_data, *dev_hashed_data;
    gpuErrchk(cudaMalloc((void**) &dev_data,        n_blocks * SHA256_INPUT_BLOCK_SIZE));
    gpuErrchk(cudaMalloc((void**) &dev_hashed_data, n_blocks * SHA256_OUTPUT_BLOCK_SIZE));

    vector<uint64_t> samples(runs);
    for (int r = 0; r < runs; r++)
        samples[r] = sha256_gpu_array_benchmark(host_data, dev_data, dev_hashed_data, n_blocks, windowed);

    gpuErrchk(cudaFree(dev_data));
    gpuErrchk(cudaFree(dev_hashed_data));

    return BenchResult::from_samples(samples);
}

/*
 * Benchmarks SHA-256 hashing on CPU vs GPU across multiple input sizes.
 *
 * For each size, multiple runs are executed to measure execution time,
 * variability, and relative performance difference between implementations.
 *
 * Results are printed to stdout and saved as a CSV file.
 */
void sha256_CPU_vs_GPU(int runs, bool sha256_windowed, const std::string& out_dir = "bench_output_files") {

    vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 4194304, 8388608};
    vector<uint8_t*> data_blocks_list;
    for (size_t n : sizes)
        data_blocks_list.push_back(generate_random_blocks(n));

    // GPU warmup
    {
        uint8_t *dev_data, *dev_hashed_data;
        gpuErrchk(cudaMalloc((void**) &dev_data,        sizes[0] * SHA256_INPUT_BLOCK_SIZE));
        gpuErrchk(cudaMalloc((void**) &dev_hashed_data, sizes[0] * SHA256_OUTPUT_BLOCK_SIZE));
        sha256_gpu_array_benchmark(data_blocks_list[0], dev_data, dev_hashed_data, sizes[0], sha256_windowed);
        gpuErrchk(cudaFree(dev_data));
        gpuErrchk(cudaFree(dev_hashed_data));
        cudaDeviceSynchronize();
    }

    vector<BenchResult> cpu_results(sizes.size());
    vector<BenchResult> gpu_results(sizes.size());

    for (size_t i = 0; i < sizes.size(); i++)
        cpu_results[i] = collect_cpu(data_blocks_list[i], sizes[i], sha256_windowed, runs);

    for (size_t i = 0; i < sizes.size(); i++)
        gpu_results[i] = collect_gpu(data_blocks_list[i], sizes[i], sha256_windowed, runs);

    std::string mode = sha256_windowed ? "windowed" : "naive";
    BenchmarkTable table(
        "sha256_CPU_vs_GPU_" + mode,
        {"size", "cpu_ns", "cpu_stddev", "cpu_cv%", "gpu_ns", "gpu_stddev", "gpu_cv%", "variation%"}
    );

    for (size_t i = 0; i < sizes.size(); i++) {
        double variation = (static_cast<double>(cpu_results[i].mean) - gpu_results[i].mean)
                           / cpu_results[i].mean * 100.0;
        std::ostringstream var, cpu_windowed_cv, gpu_windowed_cv, cpu_windowed_stddev, gpu_windowed_stddev;
        var    << std::fixed << std::setprecision(2) << variation;
        cpu_windowed_cv << std::fixed << std::setprecision(2) << cpu_results[i].cv;
        gpu_windowed_cv << std::fixed << std::setprecision(2) << gpu_results[i].cv;
        cpu_windowed_stddev<< std::fixed << std::setprecision(2) << cpu_results[i].stddev;
        gpu_windowed_stddev << std::fixed << std::setprecision(2) << gpu_results[i].stddev;

        table.add_row({
            std::to_string(sizes[i]),
            std::to_string(cpu_results[i].mean),
            cpu_windowed_stddev.str(),
            cpu_windowed_cv.str(),
            std::to_string(gpu_results[i].mean),
            gpu_windowed_stddev.str(),
            gpu_windowed_cv.str(),
            var.str()
        });
    }

    table.dump();
    for (auto* p : data_blocks_list) free(p);
}

/*
 * Benchmarks GPU SHA-256 hashing comparing naive and windowed implementations.
 *
 * For each input size, multiple runs are executed to measure execution time,
 * variability, and relative performance improvement of the windowed approach.
 *
 * Results are printed to stdout and saved as a CSV file.
 */
void sha256_GPU_naive_vs_windowed(int runs, const std::string& out_dir = "bench_output_files") {

    vector<size_t> sizes = {1024, 4096, 16384, 65536,  262144 , 4194304, 8388608, 33554432, 67108864};
    vector<uint8_t*> data_blocks_list;
    for (size_t n : sizes)
        data_blocks_list.push_back(generate_random_blocks(n));

    vector<BenchResult> naive_results(sizes.size());
    vector<BenchResult> windowed_results(sizes.size());

    for (size_t i = 0; i < sizes.size(); i++) {
    
        uint8_t* data = data_blocks_list[i];

        cudaDeviceReset(); // clean state
        collect_gpu(data, sizes[i], false, 3);  // warmup naive
        naive_results[i] = collect_gpu(data, sizes[i], false, runs);
        
        cudaDeviceReset(); // clean state
        collect_gpu(data, sizes[i], true,  3); // warmup naive
        windowed_results[i] = collect_gpu(data, sizes[i], true, runs);
        
    }
        
    BenchmarkTable table(
        "sha256_GPU_naive_vs_windowed",
        {"size", "gpu_naive_ns", "gpu_naive_stddev", "gpu_naive_cv%", "gpu_smem_ns", "gpu_smem_stddev", "gpu_smem_cv%", "variation%"}
    );

    for (size_t i = 0; i < sizes.size(); i++) {
        double variation = (static_cast<double>(naive_results[i].mean) - windowed_results[i].mean)
                           / naive_results[i].mean * 100.0;
        std::ostringstream var, naive_cv, smem_cv, naive_std, smem_std;
        var      << std::fixed << std::setprecision(2) << variation;
        naive_cv << std::fixed << std::setprecision(2) << naive_results[i].cv;
        smem_cv  << std::fixed << std::setprecision(2) << windowed_results[i].cv;
        naive_std << std::fixed << std::setprecision(2) << naive_results[i].stddev;
        smem_std << std::fixed << std::setprecision(2) << windowed_results[i].stddev;

        table.add_row({
            std::to_string(sizes[i]),
            std::to_string(naive_results[i].mean),
            naive_std.str(),
            naive_cv.str(),
            std::to_string(windowed_results[i].mean),
            smem_std.str(),
            smem_cv.str(),
            var.str()
        });
    }

    table.dump();
    for (auto* p : data_blocks_list) free(p);
}

/*
 * This function is used for profiling the two sha256 implementations.
 */
void profiling_sha256_GPU(size_t n_blocks, bool windowed) {

    uint8_t* host_data = generate_random_blocks(n_blocks);

    uint8_t *dev_data, *dev_out;
    gpuErrchk(cudaMalloc(&dev_data, n_blocks * SHA256_INPUT_BLOCK_SIZE));
    gpuErrchk(cudaMalloc(&dev_out,  n_blocks * SHA256_OUTPUT_BLOCK_SIZE));

    // transfer outside profiling
    gpuErrchk(cudaMemcpy(dev_data, host_data,
              n_blocks * SHA256_INPUT_BLOCK_SIZE, cudaMemcpyHostToDevice));

    int bpg = (n_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // only kernel under profiling
    if (windowed)
        leaf_level_build<true><<<bpg, THREADS_PER_BLOCK>>>(n_blocks, 0, dev_data, dev_out);
    else
        leaf_level_build<false><<<bpg, THREADS_PER_BLOCK>>>(n_blocks, 0, dev_data, dev_out);
    cudaDeviceSynchronize();

    gpuErrchk(cudaFree(dev_data));
    gpuErrchk(cudaFree(dev_out));
    free(host_data);
}

int main() {
    
    sha256_CPU_vs_GPU(20, true);
    //sha256_GPU_naive_vs_windowed(20);
    //profiling_sha256_GPU(33554432, true);
}