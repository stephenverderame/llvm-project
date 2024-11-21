// RUN: cat %s | %clang -x c - -c -arch arm64 -arch x86_64 2> %t0 ||:
// RUN: FileCheck %s < %t0
// RUN: cat %s | %clang -x c /dev/stdin -c -arch arm64 -arch x86_64 2> %t1 ||:
// RUN: FileCheck %s < %t1

// CHECK: error: this compilation requires multiple jobs, but non-regular input {{[^ ]+}} cannot be read more than once
int main(void) {
    return 0;
}
