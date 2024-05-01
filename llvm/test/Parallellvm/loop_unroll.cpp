

#include <cassert>
#include <stdio.h>
#include <string>

constexpr uint64_t default_n = 10'000;

int main(int argc, char** argv) {

    const uint64_t n = argc > 1 ? atoi(argv[1]) : default_n;
    uint64_t sum = 0;
    #pragma unroll 200
    for (uint64_t i = 1; i <= n; i++) {
        sum += i;
    }
    assert(sum == n * (n + 1) / 2);
    printf("sum: %ld\n", sum);
}