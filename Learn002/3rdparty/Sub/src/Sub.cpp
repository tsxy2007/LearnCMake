#include "sub.h"
#include "common.h"

int FSub::exec_Sub(int a, int b)
{
    int result = a - b;
    printresult(result);
    return result;
}