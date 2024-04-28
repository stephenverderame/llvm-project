#include <cstdlib>
#include <cstdio>
int main(int argc, char **argv) {
    const int nums = argc <= 1 ? 1000 : atoi(argv[1]);
    int a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
    for (int loop_idx = 0; loop_idx < nums; ++loop_idx) {
        switch(loop_idx % 16) {
            case 0: a = loop_idx; break;
            case 1: b = loop_idx; break;
            case 2: c = loop_idx; break;
            case 3: d = loop_idx; break;
            case 4: e = loop_idx; break;
            case 5: f = loop_idx; break;
            case 6: g = loop_idx; break;
            case 7: h = loop_idx; break;
            case 8: i = loop_idx; break;
            case 9: j = loop_idx; break;
            case 10: k = loop_idx; break;
            case 11: l = loop_idx; break;
            case 12: m = loop_idx; break;
            case 13: n = loop_idx; break;
            case 14: o = loop_idx; break;
            case 15: p = loop_idx; break;
        }
    }
    printf("a = %d\n", a);
    printf("b = %d\n", b);
    printf("c = %d\n", c);
    printf("d = %d\n", d);
    printf("e = %d\n", e);
    printf("f = %d\n", f);
    printf("g = %d\n", g);
    printf("h = %d\n", h);
    printf("i = %d\n", i);
    printf("j = %d\n", j);
    printf("k = %d\n", k);
    printf("l = %d\n", l);
    printf("m = %d\n", m);
    printf("n = %d\n", n);
    printf("o = %d\n", o);
    printf("p = %d\n", p);
    return 0;
}