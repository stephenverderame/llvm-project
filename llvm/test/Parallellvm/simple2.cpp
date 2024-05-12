#include <cstdio>
int main(int argc, char** argv) {
    auto f = argc;
    auto e = f + 5;
    auto i = e + f;
    auto a = i * 2;
    auto b = a * 7;
    auto g = i + f;
    auto c = b + 3;
    auto d = 2 * b;
    auto h = g + c + d;
    printf("g = %d, h = %d, c = %d, d = %d\n", g, h, c, d);
}