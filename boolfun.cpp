#include "boolfun.hpp"

#include <cstring>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "lpsolver.hpp"
#include "util.hpp"

namespace {
    auto plus = std::plus<double>();

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
}  // namespace

// Create reduced measure from local measure e.g. make f.bs() from f.bs(x)
#define AUTO_REDUCE_MEASURE_ONE_SIDED(measure, reduced_type, reducer, func_value, final_multiplier) do{ \
                         reduced_type ans = 0; \
                         for (size_t i = 0; i < table_size; ++i) { \
                            if (at(i) == func_value) \
                                ans = reducer(ans, static_cast<reduced_type>(measure(i)));\
                         } \
                         return ans * (final_multiplier); \
                        } while(false);
#define AUTO_REDUCE_MEASURE(measure, reduced_type, reducer, final_multiplier) do{ \
                         reduced_type ans = 0; \
                         for (size_t i = 0; i < table_size; ++i) { \
                            ans = reducer(ans, static_cast<reduced_type>(measure(i)));\
                         } \
                         return ans * (final_multiplier); \
                        } while(false);

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
    data ^= mask * randbit();
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

BoolFun BoolFun::operator* (const BoolFun& other) const {
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

void BoolFun::invert() {
    for (size_t i = 0; i < table_size_bytes; ++i) {
        data[i] ^= uint8_t(-1);
    }
}

void BoolFun::set_parity(size_t mask) {
    std::memset(data.data(), 0, table_size_bytes);
    for (size_t i = 0; i < table_size; ++i) {
        if (util::popcount(i & mask) & 1) {
            data[i >> 3] |= (1ULL << (i & 7));
        }
    }
}

void BoolFun::randomize() {
    for (size_t i = 0; i < table_size_bytes; ++i) {
        data[i] = randbyte();
    }
}

void BoolFun::choose_randomize(size_t num_ones) {
    std::vector<size_t> ind(table_size);
    std::iota(ind.begin(), ind.end(), 0);
    std::shuffle(ind.begin(), ind.end(), e1);
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
        size_t x = randint<size_t>(3, table_size-1);
        int px = util::popcount(x);
        if(px == 1 || px == input_size - 1) continue;

        size_t y = randint<size_t>(3, table_size-1);
        int py = util::popcount(y);
        if ((px & 1) == (py & 1)) {
            y ^= (1ULL << randint<size_t>(0, input_size-1));
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
    size_t ans = 0;
    for (size_t x = 0; x < table_size; ++x) {
        if(at(x)) ++ans;
    }
    return static_cast<double>(ans) / table_size;
}

double BoolFun::variance() const {
    double ex = expectation();
    return ex - ex * ex;
}

double BoolFun::expectation_pm1() const {
    int64_t ans = 0;
    for (size_t x = 0; x < table_size; ++x) {
        if(at(x)) ++ans;
        else --ans;
    }
    return static_cast<double>(ans) / table_size;
}

double BoolFun::variance_pm1() const {
    double ex = expectation();
    return 1.0 - ex * ex;
}

size_t BoolFun::s(size_t pos) const {
    size_t ans = 0;
    size_t cur_val = at(pos);
    for (size_t i = 0; i < input_size; ++i) {
        size_t nei = pos ^ (1ULL << i);
        ans += cur_val != at(nei);
    }
    return ans;
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

double BoolFun::fbs(size_t pos, std::vector<double>& x_out) const {
    LPSolverd::Matrix A;
    size_t cur_val = at(pos);
    for (size_t i = 0; i < table_size; ++i) {
        if (at(i) != cur_val) {
            A.emplace_back(input_size);
            LPSolverd::Vector & row = A.back();
            for (size_t j = 0, jmask = 1; j < input_size; ++j, jmask <<= 1) {
                if ((i & jmask) != (pos & jmask)) {
                    row[j] = -1;
                }
            }
        }
    }
    return -LPSolverd(A,
            LPSolverd::ConstVector(-1, static_cast<int>(A.size())),
            LPSolverd::ConstVector(-1, input_size)).solve(x_out);
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

size_t BoolFun::deg(bool f2, std::vector<int64_t>* poly) const {
    std::vector<int64_t> dp(restriction_mask_size);
    size_t best_deg_term = 0;
    if (poly != nullptr) poly->resize(table_size);
    for (size_t i = 0; i < restriction_mask_size; ++i) {
        size_t tmp = i, pos = 0, non_restricted_mask = 0;
        size_t p3j = 1;
        size_t last_non_restricted, non_restricted_cnt = 0, restricted_1_cnt = 0;;
        // Note num of non-restricted indices is equal to degree of term;
        // by Moebius inversion formula, (# even ones - # odd ones) under restriction
        //  is nonzero iff Fourier coefficient of term is nonzero
        for (size_t j = 0; j < input_size; ++ j) {
            size_t dig = tmp % 3;
            if (dig == 2) {
                last_non_restricted = p3j;
                ++non_restricted_cnt;
                non_restricted_mask |= (1ULL << j);
            } else if (dig == 1) {
                pos |= (1ULL << j);
                ++restricted_1_cnt;
            }
            tmp /= 3;
            p3j *= 3;
        }
        if (non_restricted_cnt == 0) {
            dp[i] = static_cast<int64_t>(at(pos));
        } else {
            dp[i] = dp[i - 2 * last_non_restricted] - dp[i - last_non_restricted];
            if ((dp[i] & 1) || (!f2 && dp[i])) {
                best_deg_term = std::max(best_deg_term, non_restricted_cnt);
            }
        }
        if (restricted_1_cnt == 0 && poly != nullptr) {
            poly->at(non_restricted_mask) = dp[i] * 
                -((((non_restricted_cnt) & 1) << 1) - 1);
        }
    }
    return best_deg_term;
}

std::vector<double> BoolFun::fourier() const {
    std::vector<double> ans(table_size);

    for (size_t s = 0; s < table_size; ++s) {
        for (size_t i = 0; i < table_size; ++i) {
            int64_t parity_val = -(((util::popcount(i&s) & 1) << 1) - 1);
            int64_t f_val = -((static_cast<int64_t>(at(i)) << 1) - 1);
            ans[s] += parity_val * f_val;
        }
        ans[s] /= table_size;
    }
    return ans;
}

void BoolFun::print_fourier(const std::vector<double>* fourier_dist_poly) const {
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

std::vector<double> BoolFun::fourier_dist() const {
    const std::vector<double> poly = fourier();
    std::vector<double> poly_dist(input_size + 1);
    for (size_t i = 0; i < table_size; ++i) {
        poly_dist[util::popcount(i)] += poly[i] * poly[i];
    }
    return poly_dist;
}

double BoolFun::fourier_moment(double k,
        const std::vector<double>& fourier_dist_poly) const {
    if (fourier_dist_poly.empty()) {
        std::vector<double> poly = fourier_dist();
        if (poly.empty()) return -std::numeric_limits<double>::infinity();
        return fourier_moment(k, poly);
    }
    
    double ans = 0.0;
    for (size_t i = 0; i < fourier_dist_poly.size(); ++i) {
        ans += fourier_dist_poly[i] * pow(i, k);
    }
    return ans;
}

size_t BoolFun::adeg(double eps0, double eps1) const {
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

double BoolFun::lambda() const {
    std::vector<double> _garbage;
    return lambda(_garbage);
}

double BoolFun::lambda(std::vector<double>& x_out) const {
    x_out.resize(table_size);

    std::vector<double> tmp(table_size);
    double ans, prev_ans;
    const double EPS = 1e-15;
    const int MAX_ITER = 50000;

    std::vector<std::vector<size_t> > adj(table_size);
    for (size_t i = 0; i < table_size; ++i) {
        size_t cur_val = at(i);
        for (size_t j_idx = 1; j_idx < table_size; j_idx <<= 1) {
            size_t j = (i ^ j_idx);
            if (cur_val != at(j)) {
                adj[i].push_back(j);
            }
        }
    }

    // Power iteration
    int iter = 0;
    for (; iter < MAX_ITER; ++iter) {
        if (iter % 10000 == 0) {
            for (size_t i = 0; i < table_size; ++i) {
                x_out[i] = uniform(0.05, 1.0);
            }
            prev_ans = -1000;
        }
        // Normalize
        double norm = 0.0;
        for (size_t i = 0; i < table_size; ++i) {
            norm += x_out[i] * x_out[i];
        }
        norm = sqrt(norm);
        for (size_t i = 0; i < table_size; ++i) {
            x_out[i] /= norm;
        }

        // Multiply by adjacency matrix (implicitly)
        ans = 0.0;
        for (size_t i = 0; i < table_size; ++i) {
            tmp[i] = input_size * x_out[i];
            for (size_t j : adj[i]) {
                tmp[i] += x_out[j];
            }
            ans += tmp[i] * x_out[i];
        }

        x_out.swap(tmp);

        // Compute error
        double err = std::fabs(ans - prev_ans);
        if (err < EPS) break;
        prev_ans = ans;
    }
    if (iter == MAX_ITER) {
        std::cerr << "WARNING: power iteration did not converge, eigenvalue may be inaccurate\n";
    }

    // L1-Normalize
    double norm_l1 = std::accumulate(x_out.begin(), x_out.end(), 0.0);
    for (size_t i = 0; i < table_size; ++i) {
        x_out[i] /= norm_l1;
    }

    return ans - input_size;
}

double BoolFun::lambda_eigen3(std::vector<std::vector<double> > * x_out) const {
    Eigen::MatrixXd A(table_size, table_size);
    A.setZero();
    for (size_t i = 0; i < table_size; ++i) {
        size_t cur_val = at(i);
        for (size_t j_idx = 1; j_idx < table_size; j_idx <<= 1) {
            size_t j = (i ^ j_idx);
            if (cur_val != at(j)) {
                A(i, j) = A(j, i) = 1;
            }
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(table_size);
    es.compute(A);
    // std::cerr << "Eigen\n";
    // for (int i = 0; i < es.eigenvalues().rows(); ++i) {
    //     std::cerr << es.eigenvalues()(i) << ":\n" << es.eigenvectors().col(i).transpose() << "\n\n";
    // }
    // std::cerr << "\nEND\n\n";

    int n_eig = es.eigenvalues().rows();
    double lam = es.eigenvalues()(n_eig-1);
    if (x_out != nullptr) {
        for (int i = 0; i < n_eig && std::fabs(es.eigenvalues()(n_eig - i - 1) - lam) < 1e-12; ++i) {
            x_out->emplace_back(es.eigenvectors().rows());
            Eigen::Map<Eigen::VectorXd> x_out_vec(x_out->at(i).data(), x_out->at(i).size());
            x_out_vec = es.eigenvectors().col(n_eig - i - 1);
            x_out_vec /= x_out_vec.sum();
        }
    }

    return lam;
}

std::vector<double> BoolFun::sens_spectrum() const {
    Eigen::MatrixXd A(table_size, table_size);
    A.setZero();
    for (size_t i = 0; i < table_size; ++i) {
        size_t cur_val = at(i);
        for (size_t j_idx = 1; j_idx < table_size; j_idx <<= 1) {
            size_t j = (i ^ j_idx);
            if (cur_val != at(j)) {
                A(i, j) = A(j, i) = 1;
            }
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(table_size);
    es.compute(A);

    int n_eig = es.eigenvalues().rows();
    std::vector<double> lams;
    for (int i = 0; i < n_eig; ++i) {
        lams.push_back(es.eigenvalues()(i));
        // if (i >= n_eig-4) {
            std::cerr << es.eigenvalues()(i) << ": ";
            for (int j = 0; j < es.eigenvectors().rows(); ++j) {
                if (std::fabs(es.eigenvectors()(j, i)) < 1e-10) {
                    std::cerr << "0 ";
                } else {
                    std::cerr << es.eigenvectors()(j, i) << " ";
                }
            }
            std::cerr << "\n";
        // }
    }
    std::cerr << "\n";
    return lams;
}

size_t BoolFun::sens_connected_compon(int min_sz, std::vector<size_t>* sizes_out) const {
    std::vector<int> stk;
    std::vector<bool> vis(table_size);
    size_t ans = 0;
    for (size_t i = 0; i < table_size; ++i) {
        if (vis[i]) continue;
        stk.push_back(i);
        vis[i] = true;
        int sz = 0;
        while (stk.size()) {
            int v = stk.back();
            stk.pop_back();
            ++sz;
            size_t cur_val = at(v);
            for (size_t j_idx = 1; j_idx < table_size; j_idx <<= 1) {
                size_t j = (v ^ j_idx);
                if (vis[j]) continue;
                if (cur_val != at(j)) {
                    stk.push_back(j);
                    vis[j] = true;
                }
            }
        }
        if (sz >= min_sz) {
            ++ans;
            if (sizes_out != nullptr) {
                sizes_out->push_back(sz);
            }
        }
    }
    if (sizes_out != nullptr) {
        std::sort(sizes_out->begin(), sizes_out->end(), std::greater<size_t>());
    }
    return ans;
}

size_t BoolFun::sens_noniso_count() const {
    const int INF = 0x3f3f3f3f;
    Eigen::MatrixXi dist(table_size, table_size);
    dist.setConstant(INF);
    size_t ans = 0;
    for (size_t i = 0; i < table_size; ++i) {
        size_t cur_val = at(i);
        for (size_t j_idx = 1; j_idx < table_size; j_idx <<= 1) {
            size_t j = (i ^ j_idx);
            if (cur_val != at(j)) {
                ++ans;
                break;
            }
        }
    }
    return ans;
}

int BoolFun::sens_diam() const {
    const int INF = 0x3f3f3f3f;
    Eigen::MatrixXi dist(table_size, table_size);
    dist.setConstant(INF);
    for (size_t i = 0; i < table_size; ++i) {
        size_t cur_val = at(i);
        for (size_t j_idx = 1; j_idx < table_size; j_idx <<= 1) {
            size_t j = (i ^ j_idx);
            if (cur_val != at(j)) {
                dist(i, j) = dist(j, i) = 1;
            }
        }
    }
    for (size_t k = 0; k < table_size; ++k) {
        for (size_t i = 0; i < table_size; ++i) {
            for (size_t j = 0; j < table_size; ++j) {
                dist(i, j) = std::min(dist(i, j), dist(i, k) + dist(k, j));
            }
        }
    }
    int ans = 0;
    for (size_t i = 0; i < table_size; ++i) {
        for (size_t j = 0; j < table_size; ++j) {
            if (dist(i, j) >= INF) continue;
            ans = std::max(dist(i, j), ans);
        }
    }
    return ans;
}

void BoolFun::sens_write_adjlist(const std::string& path, bool write_zero_deg_vert) const {
    std::ofstream ofs(path);
    for (size_t i = 0; i < table_size; ++i) {
        size_t cur_val = at(i);
        std::string iname = input_to_bin(i); 
        bool start = false;
        if (write_zero_deg_vert) {
            start = true;
            ofs << iname << " ";
        }
        for (size_t j_idx = 1; j_idx < table_size; j_idx <<= 1) {
            size_t j = (i ^ j_idx);
            if (cur_val != at(j)) {
                if (!start) {
                    start = true;
                    ofs << iname << " ";
                }
                std::string jname = input_to_bin(j); 
                ofs << jname << " ";
            }
        }
        if (start) ofs << "\n";
    }
    ofs.close();
}

void BoolFun::sens_nx_draw(const std::string& python, bool draw_zero_deg_vert) const {
    std::string tmp_path = "tmp_graph.al";
    sens_write_adjlist(tmp_path, draw_zero_deg_vert);
    std::string path = "show_graph.py";
    int tries = 0;
    while (tries < 2) {
        std::ifstream test1(path);
        if (test1) {
            test1.close();
            break;
        }
        test1.close();
        path = "../" + path;
    }
    int v = std::system((python + " " + path + " " + tmp_path).c_str());
    std::remove(tmp_path.c_str());
    if (v != 0) {
        std::cerr << "Warning: Failed to show graph in Python, "
            "please make sure Python can be called and NetworkX "
            "is installed\n";
    }
}

void BoolFun::sens_print_adjmat() const {
    for (size_t i = 0; i < table_size; ++i) {
        size_t cur_val = at(i);
        for (size_t j = 0; j < table_size; ++j) {
            if (i>j) {
                std::cout << "  ";
                continue;
            }
            if (i==j) {
                std::cout << "* ";
                continue;
            }
            size_t diff = i^j;
            if (((diff - 1) & diff) != 0) {
                // Check that differ by 1 bit
                std::cout << ". ";
                continue;
            }
            if (cur_val != at(j) > 0.5)
                std::cout << "1 ";
            else
                std::cout << "0 ";
        }
        std::cout <<"\n";
    }
}

double BoolFun::lambda_ub() const {
    std::vector<size_t> degree(table_size);
    for (size_t pos = 0; pos < table_size; ++pos) {
        degree[pos] = s(pos);
    }

    double worst = 0.0;
    for (size_t pos = 0; pos < table_size; ++pos) {
        double bound = 0.0;
        size_t cur_val = at(pos);
        for (size_t i = 0; i < input_size; ++i) {
            size_t nei = pos ^ (1ULL << i);
            if (cur_val != at(nei)) {
                bound += sqrt(degree[nei]);
            }
        }
        bound /= sqrt(degree[pos]);
        worst = std::max(worst, bound);
    }
    return worst;
}

double BoolFun::lambda_lb() const {
    std::vector<size_t> degree(table_size);
    size_t m = 0;
    for (size_t pos = 0; pos < table_size; ++pos) {
        degree[pos] = s(pos);
        m += degree[pos];
    }
    if (m == 0) return 0.0;

    double ans = 0.0;
    for (size_t pos = 0; pos < table_size; ++pos) {
        double bound = 0.0;
        size_t cur_val = at(pos);
        for (size_t i = 0; i < input_size; ++i) {
            size_t nei = pos ^ (1ULL << i);
            if (cur_val != at(nei)) {
                bound += sqrt(degree[nei]);
            }
        }
        ans += bound * sqrt(degree[pos]);
    }
    return ans / m;
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
        data[i] = uniform(lo, hi);
    }
}

void RealBoolFun::gaussian_randomize(double mu, double sigma) {
    for (size_t i = 0; i < table_size; ++i) {
        data[i] = randn(mu, sigma);
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

std::vector<double> RealBoolFun::fourier() const {
    std::vector<double> ans(table_size);

    for (size_t s = 0; s < table_size; ++s) {
        for (size_t i = 0; i < table_size; ++i) {
            double parity_val = -(((util::popcount(i&s) & 1) << 1) - 1);
            double f_val = data[i];
            ans[s] += parity_val * f_val;
        }
        ans[s] /= table_size;
    }
    return ans;
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
