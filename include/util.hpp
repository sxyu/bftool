#include <cstddef>
namespace bftool {
namespace util {
// Cross-platform fast popcount (number of 1's in binary number)
int popcount(size_t val);
// Cross-platform fast highest set bit, 4->4, 3->2
size_t high_bit(size_t val);
// Find lowest set bit, 4->4, 3->1
size_t low_bit(size_t val);
// Integer binary exponentiation
size_t intpow(size_t base, size_t expo);
}  // namespace util
}  // namespace bftool
