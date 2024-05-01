#include <cstdio>
// NOLINTBEGIN

int main(int argc, char** argv) {
    int x = 'h' * argc;
    int z = x + argc;
    printf("Hello, World! %d %d\n", x, z);
    return 0;
}

// NOLINTEND