#include "add.h"
#include "common.h"

int FAdd::exec_Add(int a, int b)
{
    int result = a + b;
    printresult(result);
    return result;
}

int FAdd::exec_Test(int a, int b)
{
    int result = a + b;
    printresult(result);
    return result;
}