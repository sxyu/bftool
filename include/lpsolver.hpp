#include <vector>
#include <utility>
#include <cmath>
namespace bftool {

template <class T>
// Simplex algorithm from KTH Royal Institute codebook (KACTL)
struct LPSolver {
    using Vector = std::vector<double>;
    using Matrix = std::vector<Vector>;

    struct ConstVector {
        T value;
        int length;
        ConstVector(T value, int length) : value(value), length(length) {}
        operator T() const {
            return value;
        }
        T operator [](int _idx) const {
            return value;
        }
        int size() const {
            return length;
        }
    };

    const T EPS = 1e-8, INF = 1/.0;

    int m, n;
    std::vector<int> N, B;
    Matrix D;

    template<class VectorLike1, class VectorLike2>
    /** Create LP solver, output x: LPSolver(A, b, c).solve(x);
     *  max c'x s.t. Ax <= b, x >= 0 */
    LPSolver(const Matrix& A, const VectorLike1& b, const VectorLike2& c) :
        m(static_cast<int>(b.size())), n(static_cast<int>(c.size())), N(n+1),
        B(m), D(m+2, Vector(n+2)) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                D[i][j] = A[i][j];
            }
        }
        N[n] = -1;
        D[m+1][n] = 1;

        for (int i = 0; i < m; ++i) {
            B[i] = n+i;
            D[i][n] = -1;
            D[i][n+1] = b[i];
        }
        for (int j = 0; j < n; ++j) {
            N[j] = j;
            D[m][j] = -c[j];
        }
    }

    /** Returns objective value, solution saved to x;
     *  returns -INF if infeasible, INF if unbounded */
    T solve(Vector & x) {
        int r = 0;
        for (int i = 1; i < m; ++i) {
            if (D[i][n+1] < D[r][n+1]) {
                r = i;
            }
        }
        if (D[r][n+1] < -EPS) {
            pivot(r, n);
            if (!simplex(2) || D[m+1][n+1] < -EPS) {
                return -INF;
            }
            for (int i = 0; i < m; ++i) {
                if (B[i] == -1) {
                    int s = 0;
                    for (int j = 1; j <= n; ++j) {
                        if (s == -1 ||
                                std::make_pair(D[i][j], N[j])
                                < std::make_pair(D[i][s], N[s])) {
                            s = j;
                        }
                    }
                    pivot(i, s);
                }
            } // for
        } // if
        bool ok = simplex(1);
        x = Vector(n);
        for (int i = 0; i < m; ++i) {
            if (B[i] < n) {
                x[B[i]] = D[i][n+1];
            }
        }
        return ok ? D[m][n+1] : INF;
    }

private:
    void pivot(int r, int s) {
        T *a = D[r].data(), inv = 1. / a[s];
        for (int i = 0; i < m+2; ++i) {
            if (i != r && std::fabs(D[i][s]) > EPS) {
                T *b = D[i].data(), inv2 = b[s] * inv;
                for (int j = 0; j < n+2; ++j) {
                    b[j] -= a[j] * inv2;
                }
                b[s] = a[s] * inv2;
            }
        }
        for (int j = 0; j < n+2; ++j) {
            if (j != s)
                D[r][j] *= inv;
        }
        for (int i = 0; i < m+2; ++i) {
            if (i != r)
                D[i][s] *= -inv;
        }
        D[r][s] = inv;
        std::swap(B[r], N[s]);
    }

    bool simplex(int phase) {
        int x = m + phase - 1;
        for (;;) {
            int s = -1;
            for (int j = 0; j < n+1; ++j) {
                if (N[j] != -phase) {
                    if (s == -1 || std::make_pair(D[x][j], N[j]) < std::make_pair(D[x][s] , N[s]))
                        s = j;
                }
            }
            if (D[x][s] >= -EPS)
                return true;
            int r = -1;
            for (int i = 0; i < m; ++i) {
                if (D[i][s] <= EPS)
                    continue;

                if (r == -1 || std::make_pair(D[i][n+1] / D[i][s], B[i])
                        < std::make_pair(D[r][n+1] / D[r][s], B[r])) {
                    r = i;
                }
            }
            if (r == -1) return false;
            pivot(r, s);
        }
    }
}; // struct
using LPSolverd = LPSolver<double>;

}  // namespace bftool  
