#include "boolfun.hpp"

#include "util.hpp"
#include "boolfun_macros.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include "lpsolver.hpp"

namespace {
inline int64_t neg1p(int64_t val01) {
    return -(((val01 & 1) << 1) - 1);
}
}  // namespace

namespace bftool {
// RealBoolFun
RealBoolFun::RealBoolFun(size_t input_size, double init_val) :
    input_size(input_size),
    table_size(1ULL << input_size),
    restriction_mask_size(util::intpow(3, input_size)) {
    static_assert(sizeof(bool) == 1, "We do not support systems with bool width != 1 byte");

    data.resize(table_size, init_val);
}
RealBoolFun::RealBoolFun(const BoolFun& f) {
    (*this) = f;
}
RealBoolFun::RealBoolFun(const std::vector<double>& tab) {
    (*this) = tab;
}
RealBoolFun& RealBoolFun::operator=(double val) {
    std::fill(data.begin(), data.end(), val);
    return *this;
}
RealBoolFun& RealBoolFun::operator=(const std::vector<double>& val) {
    data.resize(val.size());
    std::copy(val.begin(), val.end(), data.begin());

    input_size = 0;
    while ((1ULL << input_size) < val.size()) ++input_size;
    table_size = val.size();
    restriction_mask_size = util::intpow(3, input_size);
    return *this;
}
RealBoolFun& RealBoolFun::operator=(const BoolFun& f) {
    input_size = f.input_size;
    table_size = f.table_size;
    restriction_mask_size = util::intpow(3, input_size);
    data.resize(f.table_size);
    for (size_t i = 0; i < f.table_size; ++i) {
        data[i] = f(i);
    }
    return *this;
}

double& RealBoolFun::operator[](size_t pos) {
    pos &= table_size - 1;
    return data[pos];
}
double RealBoolFun::operator()(size_t pos) const {
    pos &= table_size - 1;
    return data[pos];
}
double RealBoolFun::operator[](size_t pos) const {
    pos &= table_size - 1;
    return data[pos];
}
double RealBoolFun::at(size_t pos) const {
    pos &= table_size - 1;
    return data[pos];
}

void RealBoolFun::randomize(double lo, double hi) {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] = util::uniform(lo, hi);
    }
}

void RealBoolFun::gaussian_randomize(double mu, double sigma) {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] = util::randn(mu, sigma);
    }
}

void RealBoolFun::invert() {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] = 1.0 - data[i];
    }
}

void RealBoolFun::negate() {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] -= data[i];
    }
}

void RealBoolFun::neg1p() {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] = ::neg1p(static_cast<int64_t>(data[i]));
    }
}

void RealBoolFun::sgn() {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] = (data[i] >= 0.0 ? 1.0 : -1.0);
    }
}

void RealBoolFun::round() {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] = std::round(data[i]);
    }
}

BoolFun RealBoolFun::sgn_to_boolfun() const {
    BoolFun f(input_size);
    for (size_t i = 0; i < table_size; ++i) {
        f[i] = data[i] < 0 ? 1 : 0;
    }
    return f;
}

BoolFun RealBoolFun::round_to_boolfun() const {
    BoolFun f(input_size);
    for (size_t i = 0; i < table_size; ++i) {
        f[i] = data[i] >= 0.5 ? 1 : 0;
    }
    return f;
}

RealBoolFun& RealBoolFun::operator *=(double t) {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] *= t;
    }
    return *this;
}

RealBoolFun& RealBoolFun::operator /=(double t) {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] /= t;
    }
    return *this;
}

RealBoolFun& RealBoolFun::operator +=(double t) {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] += t;
    }
    return *this;
}

RealBoolFun& RealBoolFun::operator -=(double t) {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] -= t;
    }
    return *this;
}

size_t RealBoolFun::deg() const {
    const std::vector<double>& four = fourier(); 
    for (size_t i = four.size()-1; ~i; --i) {
        if (four[i] != 0.0) {
            return util::popcount(i);
        }
    }
    return 0;
}

const std::vector<double>& RealBoolFun::fourier() const {
    if (fourier_data.empty()) {
        fourier_data.resize(table_size);
        if (input_size < 4) {
            for (size_t s = 0; s < table_size; ++s) {
                for (size_t i = 0; i < table_size; ++i) {
                    double parity_val = -(((util::popcount(i&s) & 1) << 1) - 1);
                    double f_val = data[i];
                    fourier_data[s] += parity_val * f_val;
                }
                fourier_data[s] /= table_size;
            }
        }
        else {
            fourier_data = data;
            fft(fourier_data);
            for (size_t i = 0; i < table_size; ++i) {
                fourier_data[i] /= table_size;
            }
        }
    }
    return fourier_data;
}

size_t RealBoolFun::adeg(double eps0, double eps1) const {
    std::vector<std::vector<size_t> > popcnt_order(input_size + 1);
    std::vector<size_t> num_of_degree(input_size + 1);
    for (size_t i = 0; i < table_size; ++i) {
        popcnt_order[util::popcount(i)].push_back(i);
    }
    num_of_degree[0] = 1;
    for (size_t i = 1; i <= input_size; ++i) {
        num_of_degree[i] = num_of_degree[i-1] + popcnt_order[i].size();
    }

    auto check = [&] (size_t deg) {
        const size_t num_vars = num_of_degree[deg] * 2 + 1;
        LPSolverd::Matrix A(table_size * 2, LPSolverd::Vector(num_vars));
        LPSolverd::Vector b(table_size * 2), c(num_vars), x;
        for (size_t i = 0; i < table_size; ++i) {
            int64_t fx = static_cast<int64_t>(at(i));
            if (fx == 1) {
                b[i<<1] = 1.;
                b[(i<<1)|1] = -1.;
                A[i<<1][0] = A[(i<<1)|1][0] = -1.;
            } else {
                b[i<<1] = eps0;
                b[(i<<1)|1] = eps0;
            }
        }
        c[0] = -1; // Slack
        // Constraints for each input
        for (size_t x = 0; x < table_size; ++x) {
            auto& row = A[x<<1];
            // Enter each table
            size_t k = 0;
            for (size_t d = 0; d <= deg; ++d) {
                for (size_t it = 0; it < popcnt_order[d].size(); ++it) {
                    size_t i = popcnt_order[d][it];
                    if ((i & x) == i) {
                        // Is subset
                        row[(k<<1) + 1] = 1;
                        row[(k<<1) + 2] = -1;
                    } else {
                        row[(k<<1) + 1] = row[(k<<1) + 2] = 0;
                    }
                    ++k;
                }
            }

            auto& row2 = A[(x<<1)|1];
            for (size_t i = 1; i < num_vars; ++i) {
                row2[i] = -row[i];
            }
        }
        double ans = -LPSolverd(A, b, c).solve(x);
        return ans <= eps1;
    };

    size_t lo = 0, hi = input_size + 1;
    while (hi - lo > 1) {
        size_t mi = lo + (hi - lo - 1) / 2;
        if (check(mi)) {
            hi = mi+1;
        } else {
            lo = mi+1;
        }
    }
    return lo;
}

void RealBoolFun::print_fourier(const std::vector<double>* fourier_dist_poly) const {
    if (fourier_dist_poly == nullptr) {
        const std::vector<double> four = fourier();
        return print_fourier(&four);
    }

    std::vector<std::vector<size_t> > popcnt_order(input_size + 1);
    for (size_t i = 0; i < table_size; ++i) {
        popcnt_order[util::popcount(i)].push_back(i);
    }
    bool first = true;
    for (size_t i = 0; i < popcnt_order.size(); ++i) {
        for (size_t j : popcnt_order[i]) {
            if (fourier_dist_poly->at(j) == 0.0) continue;
            if (!first) {
                if (fourier_dist_poly->at(j) > 0) std::cout << "+ ";
                else std::cout << "- ";
            } else {
                first = false;
                if (fourier_dist_poly->at(j) < 0) {
                    std::cout << "- ";
                }
            }
            std::cout << std::fabs(fourier_dist_poly->at(j));
            std::cout << " ";
            for (size_t k = 0; k < input_size; ++k) {
                if ((j & (1ULL << k))) {
                    std::cout << "x" << k+1 << " ";
                }
            }
        }
    }
    std::cout << "\n";
}

static RealBoolFun from_poly(const std::vector<double>& poly) {
    size_t input_size = 0;
    while ((1ULL << input_size) < poly.size()) ++input_size;
    RealBoolFun f(input_size);
    for (size_t i = 0 ; i < f.table_size; ++i) {
        // Not the most efficient possible
        double value = poly[i] + poly[0];
        for (size_t j = 1; j < i; ++j) {
            if ((i & j) == j) {
                value += poly[j];
            }
        }
        f.data[i] = value;
    }
    return f;
}

static RealBoolFun from_poly_pm1(const std::vector<double>& poly) {
    size_t input_size = 0;
    while ((1ULL << input_size) < poly.size()) ++input_size;
    RealBoolFun f(input_size);
    for (size_t i = 0 ; i < f.table_size; ++i) {
        // Not the most efficient possible
        double value = 0.;
        for (size_t j = 0; j < f.table_size; ++j) {
            if (util::popcount(i & j) & 1) {
                value -= poly[j];
            } else {
                value += poly[j];
            }
        }
        f.data[i] = value;
    }
    return f;
}

static RealBoolFun random(size_t size, double lo = 0.0, double hi = 1.0) {
    RealBoolFun f(size);
    f.randomize(lo, hi);
    return f;
}

double RealBoolFun::expectation() const {
    double ans = 0.;
    for (size_t x = 0; x < table_size; ++x) {
        ans += data[x];
    }
    return ans / table_size;
}

double RealBoolFun::variance() const {
    double m2 = 0., m1 = 0.;
    for (size_t x = 0; x < table_size; ++x) {
        m2 += data[x] * data[x];
        m1 += data[x];
    }
    return m2 - m1 * m1;
}

double RealBoolFun::s(size_t pos) const {
    double ans = 0;
    double cur_val = at(pos);
    for (size_t i = 0; i < input_size; ++i) {
        size_t nei = pos ^ (1ULL << i);
        ans += std::fabs(cur_val - at(nei));
    }
    return ans;
}

double RealBoolFun::s0() const {
    AUTO_REDUCE_MEASURE_ONE_SIDED(s, double, std::max, 0, 1);
}

double RealBoolFun::s1() const {
    AUTO_REDUCE_MEASURE_ONE_SIDED(s, double, std::max, 1, 1);
}

double RealBoolFun::s() const {
    AUTO_REDUCE_MEASURE(s, double, std::max, 1);
}

// OStream
std::ostream& operator << (std::ostream& os, const RealBoolFun& bf) {
    os << "RealBoolFun {\n";
    for (size_t i = 0; i < bf.table_size; ++i) {
        os << " ";
        for (size_t j = 0; j < bf.input_size; ++j) {
            os << ((i >> j) & 1ULL);
        }
        os << " -> " << static_cast<double>(bf(i)) << "\n";
    }
    return os << "}";
}

}  // namespace bftool
