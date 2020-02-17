#include "boolfun.hpp"

#include "util.hpp"
#include <cassert>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace {
    auto plus = std::plus<double>();
}  // namespace

#include "boolfun_macros.hpp"

namespace bftool {
BoolFun::ValueSetter::ValueSetter(uint8_t& data, uint8_t mask) :
    data(data), mask(mask) {};

BoolFun::ValueSetter::operator int() const {
    return data & mask;
}
BoolFun::ValueSetter::operator bool() const {
    return data & mask;
}

template<> void BoolFun::ValueSetter::set<1>() { data |= mask; }
template<> void BoolFun::ValueSetter::set<0>() { data &= ~mask; }

int BoolFun::ValueSetter::operator =(int val) {
    if (val == 1) set<1>();
    else set<0>();
    return val;
}

void BoolFun::ValueSetter::invert() {
    data ^= mask;
}

void BoolFun::ValueSetter::randomize() {
    data ^= mask * util::randbit();
}


// BoolFun
BoolFun::BoolFun(size_t input_size, int init_val) :
    input_size(input_size),
    table_size(1ULL << input_size),
    table_size_bytes(((table_size - 1) >> 3ULL) + 1ULL),
    restriction_mask_size(util::intpow(3, input_size)) {
    static_assert(sizeof(bool) == 1, "We do not support systems with bool width != 1 byte");

    data.resize(table_size_bytes);
    *this = init_val;
}
BoolFun& BoolFun::operator=(int val) {
    std::memset(data.data(), -val, table_size_bytes);
    return *this;
}

BoolFun& BoolFun::operator=(const std::vector<int64_t>& tab) {
    input_size = 0;
    while ((1ULL << input_size) <= tab.size()) {
        ++input_size;
    }
    --input_size;
    table_size = (1ULL << input_size);
    for (size_t i = 0; i < table_size; ++i) {
        if (tab[i]) {
            data[i >> 3] |= (1ULL << (i & 7));
        }
    }
    return *this;
}
BoolFun::ValueSetter BoolFun::operator[](size_t pos) {
    pos &= table_size - 1;
    return BoolFun::ValueSetter(data[pos >> 3], uint8_t(1) << (pos & 7));
}

bool BoolFun::at(size_t pos) const {
    return (data[pos >> 3] >> (pos & 7)) & 1;
}
bool BoolFun::operator[](size_t pos) const {
    pos &= table_size - 1;
    return at(pos);
}
bool BoolFun::operator()(size_t pos) const {
    pos &= table_size - 1;
    return at(pos);
}

BoolFun BoolFun::operator& (const BoolFun& other) const {
    assert(input_size == other.input_size);
    BoolFun prod = *this;
    for (size_t i = 0; i < table_size_bytes; ++i) {
        prod.data[i] = prod.data[i] & other.data[i];
    }
    return prod;
}
BoolFun BoolFun::operator* (const BoolFun& other) const {
    return (*this) & other;
}

BoolFun BoolFun::operator| (const BoolFun& other) const {
    assert(input_size == other.input_size);
    BoolFun prod = *this;
    for (size_t i = 0; i < table_size_bytes; ++i) {
        prod.data[i] = prod.data[i] | other.data[i];
    }
    return prod;
}

BoolFun BoolFun::operator^ (const BoolFun& other) const {
    assert(input_size == other.input_size);
    BoolFun prod = *this;
    for (size_t i = 0; i < table_size_bytes; ++i) {
        prod.data[i] = prod.data[i] ^ other.data[i];
    }
    return prod;
}
BoolFun BoolFun::operator+ (const BoolFun& other) const {
    return *this ^ other;
}
BoolFun BoolFun::operator- (const BoolFun& other) const {
    return *this ^ other;
}

BoolFun BoolFun::operator% (const BoolFun& other) const {
    BoolFun composed(input_size * other.input_size);
    for (size_t i = 0; i < composed.table_size; ++i) {
        size_t tmp = i;
        size_t cur_fun_input = 0;
        for (size_t j = 0; j < input_size; ++j) {
            cur_fun_input |= static_cast<size_t>(
                    other(tmp & (other.table_size - 1))) << j;
            tmp >>= other.input_size;
        }
        if (at(cur_fun_input)) {
            composed.data[i >> 3] |= (1ULL << (i & 7));
        }
    }
    return composed;
}

bool BoolFun::operator== (const BoolFun& other) const {
    assert(input_size == other.input_size);
    return data == other.data;
}
bool BoolFun::operator!= (const BoolFun& other) const {
    return !(*this == other);
}

BoolFun& BoolFun::invert() {
    for (size_t i = 0; i < table_size_bytes; ++i) {
        data[i] ^= uint8_t(-1);
    }
    return *this;
}

BoolFun& BoolFun::mul_parity(size_t mask) {
    for (size_t i = 0; i < table_size; ++i) {
        if (util::popcount(i & mask) & 1) {
            data[i >> 3] ^= (1ULL << (i & 7));
        }
    }
    return *this;
}

BoolFun& BoolFun::set_parity(size_t mask) {
    std::memset(data.data(), 0, table_size_bytes);
    for (size_t i = 0; i < table_size; ++i) {
        if (util::popcount(i & mask) & 1) {
            data[i >> 3] |= (1ULL << (i & 7));
        }
    }
    return *this;
}

void BoolFun::randomize() {
    for (size_t i = 0; i < table_size_bytes; ++i) {
        data[i] = util::randbyte();
    }
}

void BoolFun::choose_randomize(size_t num_ones) {
    std::vector<size_t> ind(table_size);
    std::iota(ind.begin(), ind.end(), 0);
    util::shuffle(ind.begin(), ind.end());
    for (size_t i = 0; i < std::min(num_ones, ind.size()); ++i) {
        data[ind[i] >> 3] |= (1ULL << (ind[i] & 7));
    }
    for (size_t i = num_ones; i < ind.size(); ++i) {
        data[ind[i] >> 3] &= ~(1ULL << (ind[i] & 7));
    }
}

size_t BoolFun::balanced_randomize(size_t k) {
    std::memset(data.data(), 0, table_size_bytes);
    size_t success_cnt = 0;
    for (size_t i = 0; i < input_size; ++i) {
        size_t j = (1ULL << i);
        data[j >> 3] |= (1ULL << (j & 7));
        size_t j2 = (table_size - 1) - (1ULL << i);
        data[j2 >> 3] |= (1ULL << (j2 & 7));
    }
    for (size_t i = 0; i < k; ++i) {
        size_t x = util::randint<size_t>(3, table_size-1);
        int px = util::popcount(x);
        if(px == 1 || px == input_size - 1) continue;

        size_t y = util::randint<size_t>(3, table_size-1);
        int py = util::popcount(y);
        if ((px & 1) == (py & 1)) {
            y ^= (1ULL << util::randint<size_t>(0, input_size-1));
        }
        py = util::popcount(y);
        if (py == 1 || py == input_size - 1) continue;
        bool atx = at(x);
        if(at(y) != atx) continue;

        if (atx == 0) {
            data[x >> 3] |= (1ULL << (x & 7));
            data[y >> 3] |= (1ULL << (y & 7));
        } else {
            data[x >> 3] &= ~(1ULL << (x & 7));
            data[y >> 3] &= ~(1ULL << (y & 7));
        }
        ++success_cnt;
    }
    return success_cnt;
}

BoolFun BoolFun::andn(size_t size) {
    BoolFun f(size, 0);
    f[-1] = 1;
    return f;
}

BoolFun BoolFun::orn(size_t size) {
    BoolFun f(size, 1);
    f[0] = 0;
    return f;
}

BoolFun BoolFun::addr(size_t addr_size) {
    BoolFun f(addr_size + (1ULL << addr_size));
    size_t addr_reg_mask = ((1ULL << addr_size) - 1);
    for (size_t i = 0; i < f.table_size; ++i) {
        size_t addr_bit = (i & addr_reg_mask);
        if ((i >> (addr_size + addr_bit)) & 1) {
            f.data[i >> 3] |= (1ULL << (i & 7));
        }
    }
    return f;
}

BoolFun BoolFun::threshold(size_t size, size_t level) {
    BoolFun f(size);
    for (size_t i = 0; i < f.table_size; ++i) {
        if (util::popcount(i) >= level) {
            f.data[i >> 3] |= (1ULL << (i & 7));
        }
    }
    return f;
}

BoolFun BoolFun::parity(size_t size, size_t mask) {
    BoolFun f(size);
    f.set_parity(mask);
    return f;
}

BoolFun BoolFun::thresh_parity(size_t size, size_t level) {
    BoolFun f(size);
    f.set_parity();
    if (level & 1) {
        f.invert();
    }
    for (size_t i = 0; i < f.table_size; ++i) {
        if (util::popcount(i) >= level) {
            f.data[i >> 3] &= ~(1ULL << (i & 7));
        }
    }
    return f;
}

BoolFun BoolFun::kushilevitz() {
    BoolFun f(6, 1);
    for (size_t i = 0 ; i < f.table_size; ++i) {
        int cnt = util::popcount(i);
        if (cnt == 0 || cnt == 4 || cnt == 5) {
            f[i] = 0;
        }
    }
    f[0b000111] = f[0b001110] = f[0b011100] = f[0b011001] = f[0b010011] = 0;
    f[0b100101] = f[0b101001] = f[0b101010] = f[0b110010] = f[0b110100] = 0;
    return f;
}

BoolFun BoolFun::nae(size_t size) {
    BoolFun f(size, 1);
    f[0] = f[-1] = 0;
    return f;
}

BoolFun BoolFun::random(size_t size) {
    BoolFun f(size);
    f.randomize();
    return f;
}

BoolFun BoolFun::from_poly(const std::vector<int64_t>& poly) {
    size_t input_size = 0;
    while ((1ULL << input_size) < poly.size()) ++input_size;
    BoolFun f(input_size);
    for (size_t i = 0 ; i < f.table_size; ++i) {
        // Not the most efficient possible
        int64_t value = poly[i] + poly[0];
        for (size_t j = 1; j < i; ++j) {
            if ((i & j) == j) {
                value += poly[j];
            }
        }
        if (value == 1) {
            f.data[i >> 3] |= (1ULL << (i & 7));
        } else if (value != 0) {
            f.invalid = true;
            break;
        }
    }
    return f;
}

bool BoolFun::next() {
    size_t i = 0;
    for (; i < table_size_bytes-1; ++i) {
        ++data[i];
        if (data[i]) return true;
    }
    ++data[i];
    if (input_size < 3) {
        data[i] &= (1ULL << table_size) - 1;
    }
    return data[i];
}

BoolFun BoolFun::restrict(size_t i, int val) {
    BoolFun fr(input_size - 1);
    size_t mlo = (1ULL<<i)-1;
    size_t mhi = ~((1ULL<<(i+1))-1);
    for (size_t x = 0; x < table_size; ++x) {
        if (((x >> i)& 1) == val) {
            if (at(x)) {
                size_t xl = x & mlo;
                size_t xu = (x & mhi) >> 1;
                size_t xx = (xl | xu);
                fr.data[xx >> 3] |= (1ULL << (xx & 7));
            }
        }
    }
    
    return fr;
}

double BoolFun::expectation() const {
    return static_cast<double>(sum()) / table_size;
}

size_t BoolFun::sum() const {
    size_t ans = 0;
    for (size_t x = 0; x < table_size; ++x) {
        if(at(x)) ++ans;
    }
    return ans;
}

double BoolFun::variance() const {
    double ex = expectation();
    return ex - ex * ex;
}

double BoolFun::expectation_pm1() const {
    return static_cast<double>(sum_pm1()) / table_size;
}

int64_t BoolFun::sum_pm1() const {
    int64_t ans = 0;
    for (size_t x = 0; x < table_size; ++x) {
        if(at(x)) ++ans;
        else --ans;
    }
    return ans;
}

double BoolFun::variance_pm1() const {
    double ex = expectation();
    return 1.0 - ex * ex;
}

size_t BoolFun::s0() const {
    AUTO_REDUCE_MEASURE_ONE_SIDED(s, size_t, std::max, 0, 1);
}

size_t BoolFun::s1() const {
    AUTO_REDUCE_MEASURE_ONE_SIDED(s, size_t, std::max, 1, 1);
}

size_t BoolFun::s() const {
    AUTO_REDUCE_MEASURE(s, size_t, std::max, 1);
}

double BoolFun::su() const {
    AUTO_REDUCE_MEASURE(s, double, plus, 1. / table_size);
}

double BoolFun::inf(size_t i) const {
    int64_t cnt = 0;
    for (size_t x = 0; x < table_size; ++x) {
        bool val = at(x);
        if (!val && val != at(x^i)) {
            ++ cnt;
        }
    }
    return static_cast<double>(cnt) / (table_size / 2);
}

double BoolFun::fbs(size_t pos) const {
    std::vector<double> _garbage;
    return fbs(pos, _garbage);
}

double BoolFun::fbs0() const {
    AUTO_REDUCE_MEASURE_ONE_SIDED(fbs, double, std::max, 0, 1);
}

double BoolFun::fbs1() const {
    AUTO_REDUCE_MEASURE_ONE_SIDED(fbs, double, std::max, 1, 1);
}

double BoolFun::fbs() const {
    AUTO_REDUCE_MEASURE(fbs, double, std::max, 1);
}

bool BoolFun::is_balanced() const {
    int ones_minus_zeros = 0;
    for (size_t i = 0; i < table_size; ++i) {
        int val = at(i);
        if (val == 1) {
            ++ones_minus_zeros;
        } else {
            --ones_minus_zeros;
        }
    }
    return ones_minus_zeros == 0;
}

bool BoolFun::is_parity_balanced() const {
    int ones_odds_minus_evens = 0;
    for (size_t i = 0; i < table_size; ++i) {
        if (at(i) == 1) {
            if (util::popcount(i) & 1) ++ones_odds_minus_evens;
            else --ones_odds_minus_evens;
        }
    }
    return ones_odds_minus_evens == 0;
}

std::string BoolFun::input_to_bin(size_t x) const {
    std::string ans;
    for (size_t i = input_size - 1; ~i; --i) {
        if (x & (1ULL << i)) {
            ans.push_back('1');
        } else {
            ans.push_back('0');
        }
    }
    return ans;
}

// OStream
std::ostream& operator << (std::ostream& os, const BoolFun& bf) {
    os << "BoolFun {\n";
    for (size_t i = 0; i < bf.table_size; ++i) {
        os << " ";
        for (size_t j = 0; j < bf.input_size; ++j) {
            os << ((i >> j) & 1ULL);
        }
        os << " -> " << static_cast<int>(bf(i)) << "\n";
    }
    return os << "}";
}
std::ostream& operator << (std::ostream& os, const BoolFun::ValueSetter& vs) {
    return os << static_cast<bool>(vs);
}
}  // namespace bftool
