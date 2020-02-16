#include "util.hpp"
namespace bftool {
namespace util {
int popcount(size_t val) {
#if defined(__GNUC__)
    return __builtin_popcountll(val);
#elif defined(_MSC_VER)
#  include <intrin.h>
    return __popcnt64(val);
#else
    int cnt = 0;
    while (val) {
        cnt += val & 1;
        val >>= 1;
    } return cnt;
#endif
}

size_t high_bit(size_t val) {
#if defined(__GNUC__)
    return 1ULL << (63-__builtin_clzll(val));
#else
    size_t ans = 1;
    while (ans <= val && (ans & val) == 0) {
        ans <<= 1;
    }
    return ans;
#endif
}

size_t low_bit(size_t val) {
    return (val & -val);
}

size_t intpow(size_t base, size_t expo) {
    size_t r = 1;
    while (expo) {
        if (expo & 1) {
            r *= base;
        }
        base *= base;
        expo >>= 1;
    }
    return r;
}
}  // namespace util
}  // namespace bftool
