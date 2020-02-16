#include <iostream>
#include <iomanip>
#include <cstdint>
#include <random>
#include <algorithm>
#include "lpsolver.hpp"
#include "boolfun.hpp"
#include "util.hpp"
using namespace bftool;

namespace {
    template <class T> std::ostream& operator<<(std::ostream& os, const std::vector<T> & v){ for(int i=0; i<(int)v.size();++i){ if(i) os << " "; os << v[i];} return os; }
    void print_eigenspace(double lam, std::vector<std::vector<double>>& vecs) {
        std::cout << lam << ", mult=" << vecs.size() << "\n";
        for (auto& v: vecs) {
            for (double t : v) {
                std::cout << t << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::vector<double> composed_eigenvec(const BoolFun& f,
            const BoolFun& g, const std::vector<double>& vf,
            const std::vector<double>& vg) {
        std::vector<double> vfg(1ULL << (f.input_size * g.input_size));
        for (size_t i = 0; i < vfg.size(); ++i) {
            size_t tmp = i;
            size_t cur_fun_input = 0;
            vfg[i] = 1.0;
            for (size_t j = 0; j < f.input_size; ++j) {
                size_t dig = tmp & (g.table_size - 1);
                vfg[i] *= vg[dig];
                cur_fun_input |= static_cast<size_t>(
                        g(dig)) << j;
                tmp >>= g.input_size;
            }
            vfg[i] *= vf[cur_fun_input];
        }
        return vfg;
    }
    std::vector<double> mul_adjmat_vec(const BoolFun& f,
            const std::vector<double>& v) {
        std::vector<double> ans(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            size_t i_val = f(i);
            for (size_t jp = 0; jp < f.input_size; ++jp) {
                size_t j = i ^ (1ULL << jp);
                size_t j_val = f(j);
                if (i_val != j_val) {
                    ans[i] += v[j];
                }
            }
        }
        return ans;
    }
    std::vector<double> mul_vecs(const std::vector<double>& v1,
            const std::vector<double>& v2) {
        std::vector<double> ans(v1.size());
        for (size_t i = 0; i < v1.size(); ++i) {
            ans[i] = v1[i] * v2[i];
        }
        return ans;
    }
    double dot_vecs(const std::vector<double>& v1,
            const std::vector<double>& v2) {
        double ans = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            ans += v1[i] * v2[i];
        }
        return ans;
    }
    void report_ce(BoolFun& f) {
        std::cout << "COUNTEREXAMPLE!\n";
        std::cout << f << "\n";
        std::cout << "deg " << f.deg() << "\n";
        std::cout << "lambda " << f.lambda() << "\n";
        std::cout << "lub " << f.lambda_ub() << "\n";
        std::cout << "llb " << f.lambda_lb() << "\n";
        std::cout << "su " << f.su() << "\n";
        std::cout << "s " << f.s() << "\n";
        std::vector<double> poly_fourier = f.fourier_dist();
        for (int i = 1; i < 20; ++i) {
            std::cout << "mmt " << pow(f.fourier_moment(i, poly_fourier), 1./i) << "\n";
        }
        std::cout << "~deg " << f.adeg() << "\n\n";
        // std::exit(1);
    }
    void check_vecs_div(BoolFun& f, const std::vector<double>& v1,
            const std::vector<double>& v2, double v1_div_v2) {
        for (size_t i = 0; i < v1.size(); ++i) {
            if (std::fabs(v1[i] - v2[i] * v1_div_v2) > 1e-9) {
                std::cout << "BAD\n";
                for (auto t : v1) std::cout << t << " ";
                std::cout << "\n";
                for (auto t : v2) std::cout << t << " ";
                std::cout << "\n";
                std::exit(1);
            }
        }
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(7) << std::boolalpha;
    BoolFun f = BoolFun::kushilevitz();
    // f.neg1p();
    f.print_fourier();
    f.print_fourier_01();
    std::cout << f.deg() << "\n";
    std::cout << f.adeg() << "\n";
    return 0;
    // for (int i = 0; i < 8; ++i) {
    //     BoolFun fr = f.restrict(5, i&1).restrict(4,(i>>1)&1).restrict(1,(i>>2)&1);
    //     fr.print_fourier();
    //     std::cout << fr << "\n";
    //     std::cout << "\n";
    // }
    // std::cout << f << "\n";
    // std::cout << f.restrict(0,0) << "\n";
    // std::cout << f.restrict(0,1).restrict(0,1).su() << "\n";
    // std::cout << f.restrict(0,1).restrict(0,1).restrict(1,1).su() << "\n";
    // BoolFun ff(5); //= BoolFun::random(3);//BoolFun::andn(3) * BoolFun::orn(3);
    // do {
    //     // std::cout << ff << "\n";
    //     size_t rid = util::randint<size_t>(0, ff.input_size-1);
    //     BoolFun fr = ff.restrict(rid, 1);
    //     // std::cout << fr << "\n";
    //     BoolFun fr2 = ff.restrict(rid, 0);
    //     // std::cout << "LAMBDAS\n";
    //     double l1 = ff.su(); //ff.lambda_eigen3();
    //     double l2 = fr.su(); //fr.lambda_eigen3();
    //     double l3 = fr2.su();//fr2.lambda_eigen3();
    //     double diff =l1 - std::max(l2, l3);
    //     std::cout << diff << "\n";
    //     if (diff > 1.0000000000001) {
    //         std::cout << ff << "\n" << fr << "\n" << fr2 << "\n"
    //             << l1 << "\n" << l2 << "\n" << l3 << "\n";
    //     }
    // }
    // while(ff.next());
    std::cout << "DONE\n";
    // std::cout << "FOURIER\n";
    // ff.print_fourier();
    // fr.print_fourier();
    // fr2.print_fourier();
    return 0;
}
