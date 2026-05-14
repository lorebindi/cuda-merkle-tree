#ifndef SHA256_GPU_H
#define SHA256_GPU_H

#include <iostream>
#include <memory.h>
#include <stdint.h>
#include <stdlib.h>

#define SHA256_OUTPUT_BLOCK_SIZE 32
#define SHA256_INPUT_BLOCK_SIZE 64

typedef struct {
    uint32_t hash[8]; // internal state, initialized with initial value of sha256
} CUDA_SHA256_CTX;

/*############################### Utility Function ###############################*/

/* Left rotate the 32-bit integer 'a' by 'b' bits 
 (Not strictly needed for SHA-256, but included for completeness.) */
__device__ __forceinline__ uint32_t left_rotate(uint32_t a, uint32_t b) {
    return (a << b) | (a >> (32 - b));
}

/* Right rotate a 32-bit integer 'a' by 'b' bits. */
__device__ __forceinline__ uint32_t right_rotate(uint32_t a, uint32_t b) {
    return (a >> b) | (a << (32 - b));
}

/* 'Choose' function: selects bits from y or z depending on x
 SHA-256 literature: Ch(x,y,z) = (x AND y) XOR (NOT x AND z)*/
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

/* 'Majority' function: selects the bit that appears in the majority among x, y, z
 SHA-256 literature: Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z) */
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

/* Big Sigma 0 = ROTR(x,2) XOR ROTR(x,13) XOR ROTR(x,22)
 used in the main compression loop. */
__device__ __forceinline__ uint32_t Big_sigma0(uint32_t x) {
    return right_rotate(x, 2) ^ right_rotate(x, 13) ^ right_rotate(x, 22);
}

/* Big Sigma 1 (Σ1) = ROTR(x,6) XOR ROTR(x,11) XOR ROTR(x,25) */
__device__ __forceinline__ uint32_t Big_sigma1(uint32_t x) {
    return right_rotate(x, 6) ^ right_rotate(x, 11) ^ right_rotate(x, 25);
}

/* Small Sigma 0 (σ0) = ROTR(x,7) XOR ROTR(x,18) XOR SHR(x,3)
 Used for extending the first 16 words into w[0..63]. */
__device__ __forceinline__ uint32_t Small_sigma0(uint32_t x) {
    return right_rotate(x, 7) ^ right_rotate(x, 18) ^ (x >> 3);
}

/* Small Sigma 1 (σ1) = ROTR(x,17) XOR ROTR(x,19) XOR SHR(x,10) */
__device__ __forceinline__ uint32_t Small_sigma1(uint32_t x) {
    return right_rotate(x, 17) ^ right_rotate(x, 19) ^ (x >> 10);
}

/*
 * Pads a message to a single 512-bit (64-byte) block according to SHA-256.
 *
 * Parametri:
 * 	'msg': pointer to the input message (host memory).
 * 	'len': length of the message in bytes (must be <= 55 for single-block padding).
 * 	'block': output 64-byte buffer where the padded block is written.*/

__host__ inline void sha256_pad_single_block(const uint8_t* msg, size_t len, uint8_t block[64]) {
    
    // Only message that fit in a block.
    if (len > 55) {
        std::cerr << "Message too long for single-block test\n";
        exit(1);
    }

	//Fills the block with zeros. 
	// Only the bytes after the message up to byte 55 will be used for 0 padding.
    memset(block, 0, 64);
	// Copies the original message bytes into the block.
    memcpy(block, msg, len);
	// Appends the mandatory 0x80 byte after the message.
    block[len] = 0x80;

  
    // Appends the original message length in bits in big-endian format
	//  in the last 8 bytes (bytes 56..63).
    uint64_t bit_len = len * 8;
    block[63] = (bit_len) & 0xff;
    block[62] = (bit_len >> 8) & 0xff;
    block[61] = (bit_len >> 16) & 0xff;
    block[60] = (bit_len >> 24) & 0xff;
    block[59] = (bit_len >> 32) & 0xff;
    block[58] = (bit_len >> 40) & 0xff;
    block[57] = (bit_len >> 48) & 0xff;
    block[56] = (bit_len >> 56) & 0xff;
}

/*############################### Costant ###############################*/

extern __constant__ uint32_t k[64];

/*############################### Functions ###############################*/

/* This function initialize the struct context */
__device__ __forceinline__ void sha256_init(CUDA_SHA256_CTX *ctx) {
    /* The following are the first 32 bits of the fractional parts of the 
     square roots of the first 8 prime numbers */
	ctx->hash[0] = 0x6a09e667;
	ctx->hash[1] = 0xbb67ae85;
	ctx->hash[2] = 0x3c6ef372;
	ctx->hash[3] = 0xa54ff53a;
	ctx->hash[4] = 0x510e527f;
	ctx->hash[5] = 0x9b05688c;
	ctx->hash[6] = 0x1f83d9ab;
	ctx->hash[7] = 0x5be0cd19;
}

/* 
* This function implements the so-called 'compression function' in the literature. This function 
* is executed for the only 512 bit block. 
* 
* Parameters:
*  - 'ctx': pointer to the sha context.
*  - 'data': read-only pointer to the data.
*/
__device__  __forceinline__ void sha256_transform(CUDA_SHA256_CTX *ctx, const uint8_t data[]) {
	
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    /* Initialize the first 16 words (m[0..15]) of the message schedule array. */
	const uint32_t* data32 = (const uint32_t*)data;
	for (i = 0; i < 16; i++)
		m[i] = __byte_perm(data32[i], 0, 0x0123); // Load message word m[i] from the input block (preserve original byte order)
    /* Extend the first 16 words into the remaining 48 words of the message schedule (m[16..63]) */
	for ( ; i < 64; i++)
		m[i] = Small_sigma1(m[i - 2]) + m[i - 7] + Small_sigma0(m[i - 15]) + m[i - 16];

	a = ctx->hash[0];
	b = ctx->hash[1];
	c = ctx->hash[2];
	d = ctx->hash[3];
	e = ctx->hash[4];
	f = ctx->hash[5];
	g = ctx->hash[6];
	h = ctx->hash[7];

    /* Compression loop */
	for (i = 0; i < 64; i++) {
		t1 = h + Big_sigma1(e) + Ch(e,f,g) + k[i] + m[i];
		t2 = Big_sigma0(a) + Maj(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->hash[0] += a;
	ctx->hash[1] += b;
	ctx->hash[2] += c;
	ctx->hash[3] += d;
	ctx->hash[4] += e;
	ctx->hash[5] += f;
	ctx->hash[6] += g;
	ctx->hash[7] += h;
}

/* 
* This function implements the so-called 'compression function' in the literature. This 
* function is executed for the only 512 bit block. 
*
* Instead of storing the full 64-word message schedule, this version uses a
* 16-word "window" because computing W[i] depends only on the previous 16 words 
* (i.e i-2, i-7, i-15, i-16). Compared to the traditional 'transform', this
* implementation reduces per-thread memory usage (only) for the message schedule by 75%.
*
* Parameters:
*  - 'ctx': pointer to the sha context.
*  - 'data': read-only pointer to the data.
*/
__device__  __forceinline__ void sha256_transform_windowed(CUDA_SHA256_CTX *ctx, const uint8_t data[]) {
	
    uint32_t a, b, c, d, e, f, g, h, j, t1, t2;
	uint32_t m[16]; // messagge schedule

    a = ctx->hash[0];
	b = ctx->hash[1];
	c = ctx->hash[2];
	d = ctx->hash[3];
	e = ctx->hash[4];
	f = ctx->hash[5];
	g = ctx->hash[6];
	h = ctx->hash[7];

	int i = 0;

	const uint32_t* data32 = (const uint32_t*)data;

	for (; i < 16; i++) {
        int idx = i & 15;
		// Computing messagge schedule
		j = i * 4;
        m[idx] = __byte_perm(data32[i], 0, 0x0123); // Load message word m[i] from the input block (preserve original byte order)

		// Compression function
        uint32_t t1 = h + Big_sigma1(e) + Ch(e, f, g) + k[i] + m[idx];
        uint32_t t2 = Big_sigma0(a) + Maj(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	for (; i < 64; i++) {
        int idx = i & 15;
		// For i >= 16: rolling update.
		m[idx] = Small_sigma1(m[(i - 2) & 15])
				+ m[(i - 7) & 15]
				+ Small_sigma0(m[(i - 15) & 15])
				+ m[(i - 16) & 15];

		// Compression function
        uint32_t t1 = h + Big_sigma1(e) + Ch(e, f, g) + k[i] + m[idx];
        uint32_t t2 = Big_sigma0(a) + Maj(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->hash[0] += a;
	ctx->hash[1] += b;
	ctx->hash[2] += c;
	ctx->hash[3] += d;
	ctx->hash[4] += e;
	ctx->hash[5] += f;
	ctx->hash[6] += g;
	ctx->hash[7] += h;
}

/* 
* This function computes the SHA-256 hash of a single block of data. 
* The function does not handle messages longer than 64 bytes and 
* does not perform padding (this is handled from the host side if needed).
* 
* Parameters:
*  - 'input': input data of 64 byte.
*  - 'output': sha256 digest of 32 byte.
*/

template<bool sha256_windowed>
__device__ __forceinline__ void sha256_single_block(const uint8_t input[SHA256_INPUT_BLOCK_SIZE], uint8_t output[SHA256_OUTPUT_BLOCK_SIZE]) {

    CUDA_SHA256_CTX ctx;
    sha256_init(&ctx);
	if constexpr (sha256_windowed)
		sha256_transform_windowed(&ctx, input);
	else
    	sha256_transform(&ctx, input);

    /* Since GPU NVIDIA use little endian uint8_t ordering and SHA uses big endian,
	 reverse all the uint8_ts when copying the final hash to the output hash. */
	uint32_t* out32 = (uint32_t*)output;
    for (int i = 0; i < 8; i++)
        out32[i] = __byte_perm(ctx.hash[i], 0, 0x0123);
}

#endif // SHA256_GPU_H