#include <iostream>
#include <iomanip>
#include <cstdint>
#include <random>
#include <algorithm>
#include "boolfun.hpp"
#include "util.hpp"
using namespace bftool;

namespace {
    template <class T> std::ostream& operator<<(std::ostream& os, const std::vector<T> & v){ for(int i=0; i<(int)v.size();++i){ if(i) os << " "; os << v[i];} return os; }
    template <class T> std::ostream& operator<<(std::ostream& os, const std::pair<T, T> & p){ os << p.first << " " << p.second; return os; }
}

int main() {
    std::cout << std::fixed << std::setprecision(7) << std::boolalpha;

    BoolFun f = BoolFun::kushilevitz();
    auto g = RealBoolFun(f, true) * f.sens_fun();
    g.print_fourier();
    // f.mul_parity();
    // f.print_fourier();

    return 0;
}
