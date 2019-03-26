#include "random_normal_sample.h"
#include "pcg-c-0.94/include/pcg_variants.h"

void init_random_seeds(){
  pcg64_random_t rng1, rng2, rng3;
  pcg64_srandom_r(&rng1, time(NULL), (intptr_t)&rng1);
  pcg64_srandom_r(&rng2, time(NULL), (intptr_t)&rng2);
  pcg64_srandom_r(&rng3, time(NULL), (intptr_t)&rng3);
}
