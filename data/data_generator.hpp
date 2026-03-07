#include <cstdint>

using namespace std;

#define BLOCK_SIZE 64

uint8_t* generate_random_blocks(size_t n_blocks);
void free_blocks(uint8_t* ptr);