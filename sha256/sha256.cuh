#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>
#include <stdlib.h>

#define SHA256_OUTPUT_BLOCK_SIZE 32
#define SHA256_INPUT_BLOCK_SIZE 64

/*
 * Pads a message to a single 512-bit (64-byte) block according to SHA-256.
 *
 * Parametri:
 * 	'msg': pointer to the input message (host memory).
 * 	'len': length of the message in bytes (must be <= 55 for single-block padding).
 * 	'block': output 64-byte buffer where the padded block is written.
*/
__host__ void sha256_pad_single_block(const uint8_t* msg, size_t len, uint8_t block[64]);

/* 
 * This function computes the SHA-256 hash of a single block of data. 
 * The function does not handle messages longer than 64 bytes and 
 * does not perform padding (this is handled from the host side if needed).
 */
__device__ void sha256_single_block(const uint8_t input[SHA256_INPUT_BLOCK_SIZE], uint8_t output[SHA256_OUTPUT_BLOCK_SIZE]);

#endif // SHA256_H