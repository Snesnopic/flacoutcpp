#include <iostream>
#include <limits>
int main() {
    uint32_t cost = std::numeric_limits<uint32_t>::max();
    uint32_t bits = 16 + 4096 * 16;
    uint32_t sum = cost + bits;
    std::cout << sum << "\n";
    return 0;
}
