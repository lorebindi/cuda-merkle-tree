#include <iostream>
#include <memory.h>
#include <stdint.h>
#include "sha256.cuh"

typedef struct {
    uint32_t hash[8]; // internal state, initialized with initial value of sha256
} CUDA_SHA256_CTX;

/*############################### Utility Function ###############################*/

/* Left rotate the 32-bit integer 'a' by 'b' bits 
 (Not strictly needed for SHA-256, but included for completeness.) */
__device__ __inline__ uint32_t left_rotate(uint32_t a, uint32_t b) {
    return (a << b) | (a >> (32 - b));
}

/* Right rotate a 32-bit integer 'a' by 'b' bits. */
__device__ __inline__ uint32_t right_rotate(uint32_t a, uint32_t b) {
    return (a >> b) | (a << (32 - b));
}

/* 'Choose' function: selects bits from y or z depending on x
 SHA-256 literature: Ch(x,y,z) = (x AND y) XOR (NOT x AND z)*/
__device__ __inline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

/* 'Majority' function: selects the bit that appears in the majority among x, y, z
 SHA-256 literature: Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z) */
__device__ __inline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

/* Big Sigma 0 = ROTR(x,2) XOR ROTR(x,13) XOR ROTR(x,22)
 used in the main compression loop. */
__device__ __inline__ uint32_t Big_sigma0(uint32_t x) {
    return right_rotate(x, 2) ^ right_rotate(x, 13) ^ right_rotate(x, 22);
}

/* Big Sigma 1 (Σ1) = ROTR(x,6) XOR ROTR(x,11) XOR ROTR(x,25) */
__device__ __inline__ uint32_t Big_sigma1(uint32_t x) {
    return right_rotate(x, 6) ^ right_rotate(x, 11) ^ right_rotate(x, 25);
}

/* Small Sigma 0 (σ0) = ROTR(x,7) XOR ROTR(x,18) XOR SHR(x,3)
 Used for extending the first 16 words into w[0..63]. */
__device__ __inline__ uint32_t Small_sigma0(uint32_t x) {
    return right_rotate(x, 7) ^ right_rotate(x, 18) ^ (x >> 3);
}

/* Small Sigma 1 (σ1) = ROTR(x,17) XOR ROTR(x,19) XOR SHR(x,10) */
__device__ __inline__ uint32_t Small_sigma1(uint32_t x) {
    return right_rotate(x, 17) ^ right_rotate(x, 19) ^ (x >> 10);
}

/*
 * Pads a message to a single 512-bit (64-byte) block according to SHA-256.
 *
 * Parametri:
 * 	'msg': pointer to the input message (host memory).
 * 	'len': length of the message in bytes (must be <= 55 for single-block padding).
 * 	'block': output 64-byte buffer where the padded block is written.
*/
__host__ void sha256_pad_single_block(const uint8_t* msg, size_t len, uint8_t block[64]) {
    
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

__constant__ uint32_t k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*############################### Functions ###############################*/

/* This function initialize the struct context */
__device__ void sha256_init(CUDA_SHA256_CTX *ctx) {
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

/* This function implements the so-called 'compression function' in the literature. This function 
 is executed for the only 512 bit block. 
 
 Parameters:
  - 'ctx': pointer to the sha context.
  - 'data': read-only pointer to the data.
 */
__device__  __forceinline__ void sha256_transform(CUDA_SHA256_CTX *ctx, const uint8_t data[]) {
	
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    /* Initialize the first 16 words (m[0..15]) of the message schedule array. */
	for (i = 0, j = 0; i < 16; i++, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
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
 * This function computes the SHA-256 hash of a single block of data. 
 * The function does not handle messages longer than 64 bytes and 
 * does not perform padding (this is handled from the host side if needed).
 */
__device__ void sha256_single_block(const uint8_t input[SHA256_INPUT_BLOCK_SIZE], uint8_t output[SHA256_OUTPUT_BLOCK_SIZE]) {

    CUDA_SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_transform(&ctx, input);

    /* Since GPU NVIDIA use little endian uint8_t ordering and SHA uses big endian,
	 reverse all the uint8_ts when copying the final hash to the output hash. */
    for(int i=0;i<4;i++){
        output[i]      = (ctx.hash[0] >> (24-i*8)) & 0xff;
        output[i+4]    = (ctx.hash[1] >> (24-i*8)) & 0xff;
        output[i+8]    = (ctx.hash[2] >> (24-i*8)) & 0xff;
        output[i+12]   = (ctx.hash[3] >> (24-i*8)) & 0xff;
        output[i+16]   = (ctx.hash[4] >> (24-i*8)) & 0xff;
        output[i+20]   = (ctx.hash[5] >> (24-i*8)) & 0xff;
        output[i+24]   = (ctx.hash[6] >> (24-i*8)) & 0xff;
        output[i+28]   = (ctx.hash[7] >> (24-i*8)) & 0xff;
    }
}

/* 
 * This function implement the so-called 'chunk-loop' in the literature. 
 *
 * This function insert the input data into the SHA-256 algorithm in a streaming fashion.
 * It buffers incoming bytes until 64 bytes (512 bits) are collected, then calls
 * the compression function (`consume_chunk`) to process that chunk. 
 
__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const uint8_t data[], size_t len) {
	
	for (uint32_t i = 0; i < len; i++) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			consume_chunk(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

 This function completes the SHA-256 hash calculation for the last chunk of the messagge.
 * It performs the final padding of the remaining data in the buffer, appends the total length
 * of the message in bits, and transforms the final chunk to produce the 32-byte hash. 
__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, uint8_t hash[]) {

	uint32_t i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		consume_chunk(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	consume_chunk(ctx, ctx->data);

	/* Since GPU NVIDIA use little endian uint8_t ordering and SHA uses big endian,
	 reverse all the uint8_ts when copying the final hash to the output hash. 
	for (i = 0; i < 4; i++) {
		hash[i]      = (ctx->hash[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->hash[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->hash[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->hash[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->hash[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->hash[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->hash[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->hash[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

__global__ void kernel_sha256_hash(uint8_t* indata, uint32_t inlen, uint8_t* outdata, uint32_t n_batch) {
	uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread >= n_batch)
	{
		return;
	}
	uint8_t* in = indata  + thread * inlen;
	uint8_t* out = outdata  + thread * SHA256_BLOCK_SIZE;
	CUDA_SHA256_CTX ctx;
	cuda_sha256_init(&ctx);
	cuda_sha256_update(&ctx, in, inlen);
	cuda_sha256_final(&ctx, out);
}

extern "C"
{
void mcm_cuda_sha256_hash_batch(uint8_t* in, uint32_t inlen, uint8_t* out, uint32_t n_batch) {
	uint8_t *cuda_indata;
	uint8_t *cuda_outdata;
	cudaMalloc(&cuda_indata, inlen * n_batch);
	cudaMalloc(&cuda_outdata, SHA256_BLOCK_SIZE * n_batch);
	cudaMemcpy(cuda_indata, in, inlen * n_batch, cudaMemcpyHostToDevice);

	uint32_t thread = 256;
	uint32_t block = (n_batch + thread - 1) / thread;

	kernel_sha256_hash << < block, thread >> > (cuda_indata, inlen, cuda_outdata, n_batch);
	cudaMemcpy(out, cuda_outdata, SHA256_BLOCK_SIZE * n_batch, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error cuda sha256 hash: %s \n", cudaGetErrorString(error));
	}
	cudaFree(cuda_indata);
	cudaFree(cuda_outdata);
}
}*/