#include "util.hpp"
#include <random>
#include <algorithm>
#include <vector>
#include <string>

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

// Random
std::random_device r;
std::default_random_engine e1(1);//r());

double uniform(double a, double b) {
    std::uniform_real_distribution<double> uniform_dist(a, b);
    return uniform_dist(e1);
}

template<class T>
T randint(T a, T b) {
    std::uniform_int_distribution<T> uniform_dist(a, b);
    return uniform_dist(e1);
}
template int8_t randint<int8_t>(int8_t a, int8_t b);
template uint8_t randint<uint8_t>(uint8_t a, uint8_t b);
template int16_t randint<int16_t>(int16_t a, int16_t b);
template uint16_t randint<uint16_t>(uint16_t a, uint16_t b);
template int32_t randint<int32_t>(int32_t a, int32_t b);
template uint32_t randint<uint32_t>(uint32_t a, uint32_t b);
template int64_t randint<int64_t>(int64_t a, int64_t b);
template uint64_t randint<uint64_t>(uint64_t a, uint64_t b);

std::uniform_int_distribution<uint8_t> uniform_byte_dist(0, 255);
uint8_t randbyte() {
    return uniform_byte_dist(e1);
}

std::uniform_int_distribution<uint8_t> uniform_bit_dist(0, 1);
uint8_t randbit() {
    return uniform_bit_dist(e1);
}

uint8_t randn(double mu, double sigma) {
    std::normal_distribution<double> normal_dist(mu, sigma);
    return normal_dist(e1);
} 

template<class Iterator>
void shuffle(Iterator begin, Iterator end) {
    std::shuffle(begin, end, e1);
}

template void shuffle<uint8_t*>(uint8_t* begin, uint8_t* end);
template void shuffle<int*>(int* begin, int* end);
template void shuffle<int64_t*>(int64_t* begin, int64_t* end);
template void shuffle<size_t*>(size_t* begin, size_t* end);
template void shuffle<double*>(double* begin, double* end);
template void shuffle<std::vector<uint8_t>::iterator>(std::vector<uint8_t>::iterator begin, std::vector<uint8_t>::iterator end);
template void shuffle<std::vector<int>::iterator>(std::vector<int>::iterator begin, std::vector<int>::iterator end);
template void shuffle<std::vector<uint32_t>::iterator>(std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end);
template void shuffle<std::vector<int64_t>::iterator>(std::vector<int64_t>::iterator begin, std::vector<int64_t>::iterator end);
template void shuffle<std::vector<uint64_t>::iterator>(std::vector<uint64_t>::iterator begin, std::vector<uint64_t>::iterator end);
template void shuffle<std::vector<double>::iterator>(std::vector<double>::iterator begin, std::vector<double>::iterator end);
template void shuffle<std::vector<float>::iterator>(std::vector<float>::iterator begin, std::vector<float>::iterator end);
template void shuffle<std::string::iterator>(std::string::iterator begin, std::string::iterator end);
}  // namespace util
}  // namespace bftool
