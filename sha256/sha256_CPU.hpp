/* 
 * This SHA-256 CPU implementation is used for testing purposes only.
 * It computes the hash of single 64-byte blocks on the host to verify
 * that the GPU-based Merkle tree produces correct leaf hashes.
*/

#ifndef SHA256_CPU_H
#define SHA256_CPU_H

#include <stdint.h>
#include <stdlib.h>

#define SHA256_OUTPUT_BLOCK_SIZE 32
#define SHA256_INPUT_BLOCK_SIZE 64

/* 
* This function computes the SHA-256 hash of a single block of data. 
* The function does not handle messages longer than 64 bytes and 
* does not perform padding (this is handled from the host side if needed).
* 
* Parameters:
*  - 'input': input data of 64 byte.
*  - 'output': sha256 digest of 32 byte.
*  - 'window': true -> windowed trasform, false -> traditional transform
*/
void sha256_single_block_CPU(const uint8_t input[SHA256_INPUT_BLOCK_SIZE], uint8_t output[SHA256_OUTPUT_BLOCK_SIZE], bool window);

#endif // SHA256_CPU_H