#include "Calc.h"
#include "Sub.h"
#include "add.h"

int Calc::Add(int a, int b)
{
    FAdd add;
    return add.exec_Add(a, b);
}

int Calc::Sub(int a, int b)
{
    FSub sub;
    return sub.exec_Sub(a,b);
}