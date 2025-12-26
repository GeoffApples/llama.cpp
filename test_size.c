#include <stdio.h>
#include "ggml/include/ggml.h"
int main() { printf("block_q3_hifi size: %zu bytes\\n", sizeof(block_q3_hifi)); printf("outliers: %d\\n", Q3_HIFI_OUTFIERS_PER_BLOCK); return 0; }
