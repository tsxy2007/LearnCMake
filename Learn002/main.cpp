#include <cstdio>
#include "include/Library.h"
#include "Calc.h"

int main()
{
    Test();
    printf("Hello CMake!\n");
    Calc c;
    int value = c.Add(51, 5);
    printf("Value = %d \n",value);
}