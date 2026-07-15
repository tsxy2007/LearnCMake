#include <cstdio>

int climbStairs(int n) {
    if (n <= 2) return n;
    int prev1 = 2, prev2 = 1, result = 0;
    for (int i = 3; i <= n; i++) {
        result = prev1 + prev2;
        prev2 = prev1;
        prev1 = result;
    }
    return result;
}

int main() {
    int n = 10;
    printf("Climbing %d stairs: %d ways\n", n, climbStairs(n));
    return 0;
}
