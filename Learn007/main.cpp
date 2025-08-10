#include <cstdio>
#include "include/Library.h"
#include "Calc.h"

int main()
{
    Test();
    printf("Hello CMake! Learn007\n");
    Calc c;
    int value = c.Add(51, 5);
    printf("Value = %d \n",value);
}