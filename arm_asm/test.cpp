#include <stdio.h>

/*
  http://www.ethernut.de/en/documents/arm-inline-asm.html
 */

// qadd is a saturation add
int qadd(int a, int b)
{
    int c;

    asm("qadd %[c], %[a], %[b]"
        : [c] "=r" (c)
        : [a] "r" (a), [b] "r" (b)
        :);

    return c;
}

int main()
{
    int a = 0x7fffffff;
    int b = 10;
    int c = qadd(a, b);

    printf("%x\n", c);

    a = 0x80000000;
    b = -10;
    printf("%x\n", qadd(a, b));
    
    return 0;
}
