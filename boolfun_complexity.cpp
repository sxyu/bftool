#include "boolfun.hpp"

#include "util.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "lpsolver.hpp"
#include "boolfun_macros.hpp"

namespace {
inline int64_t neg1p(int64_t val01) {
    return -(((val01 & 1) << 1) - 1);
}
}  // namespace

namespace bftool {

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

size_t BoolFun::deg() const {
    const std::vector<double>& four = fourier(); 
    for (size_t i = four.size()-1; ~i; --i) {
        if (four[i] != 0.0) {
            return util::popcount(i);
        }
    }
    return 0;
}

const std::vector<double>& BoolFun::fourier() const {
    if (fourier_data.empty()) {
        fourier_data.resize(table_size);
        if (input_size < 4) {
            for (size_t s = 0; s < table_size; ++s) {
                for (size_t i = 0; i < table_size; ++i) {
                    int64_t parity_val = -(((util::popcount(i&s) & 1) << 1) - 1);
                    int64_t f_val = -((static_cast<int64_t>(at(i)) << 1) - 1);
                    fourier_data[s] += parity_val * f_val;
                }
                fourier_data[s] /= table_size;
            }
        }
        else {
            std::vector<int64_t> ans_i(table_size);
            for (size_t i = 0; i < table_size; ++i) {
                ans_i[i] = neg1p(at(i));
            }
            fft(ans_i);
            for (size_t i = 0; i < table_size; ++i) {
                fourier_data[i] = static_cast<double>(ans_i[i]) / table_size;
            }
        }
    }
    return fourier_data;
}

std::vector<int64_t> BoolFun::fourier_01(bool f2, size_t* deg_out) const {
    std::vector<int64_t> dp(restriction_mask_size);
    size_t best_deg_term = 0;
    std::vector<int64_t> poly(table_size);
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
        if (restricted_1_cnt == 0) {
            poly[non_restricted_mask] = dp[i] *
                -((((non_restricted_cnt) & 1) << 1) - 1);
        }
    }
    if (deg_out != nullptr)
        *deg_out = best_deg_term;
    return poly;
}

void BoolFun::print_fourier(const std::vector<double>* fourier_poly) const {
    if (fourier_poly == nullptr) {
        const std::vector<double>& four = fourier();
        return print_fourier(&four);
    }

    std::vector<std::vector<size_t> > popcnt_order(input_size + 1);
    for (size_t i = 0; i < table_size; ++i) {
        popcnt_order[util::popcount(i)].push_back(i);
    }
    bool first = true;
    for (size_t i = 0; i < popcnt_order.size(); ++i) {
        for (size_t j : popcnt_order[i]) {
            if (fourier_poly->at(j) == 0.0) continue;
            if (!first) {
                if (fourier_poly->at(j) > 0) std::cout << "+ ";
                else std::cout << "- ";
            } else {
                first = false;
                if (fourier_poly->at(j) < 0) {
                    std::cout << "- ";
                }
            }
            std::cout << std::fabs(fourier_poly->at(j));
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

void BoolFun::print_fourier_01(const std::vector<int64_t>* fourier_poly, bool f2) const {
    if (fourier_poly == nullptr) {
        const std::vector<int64_t>& four = fourier_01(f2);
        return print_fourier_01(&four);
    }

    std::vector<std::vector<size_t> > popcnt_order(input_size + 1);
    for (size_t i = 0; i < table_size; ++i) {
        popcnt_order[util::popcount(i)].push_back(i);
    }
    bool first = true;
    for (size_t i = 0; i < popcnt_order.size(); ++i) {
        for (size_t j : popcnt_order[i]) {
            if (fourier_poly->at(j) == 0.0) continue;
            if (!first) {
                if (fourier_poly->at(j) > 0) std::cout << "+ ";
                else std::cout << "- ";
            } else {
                first = false;
                if (fourier_poly->at(j) < 0) {
                    std::cout << "- ";
                }
            }
            int64_t absval = std::abs(fourier_poly->at(j));
            if (absval != 1) std::cout << absval << " ";
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
    const std::vector<double>& poly = fourier();
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
                x_out[i] = util::uniform(0.05, 1.0);
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

}
