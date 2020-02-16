#include <cstdint>
#include <ostream>
#include <vector>

namespace bftool {
/** Simple Boolean function representation for {0,1}^n -> {0,1}
 *  (smaller and faster for some operations) */
class BoolFun {
public:
    /** Internal helper struct for setting values */
    struct ValueSetter {
        // Set value of bit
        template<int value> void set();
        int operator =(int val);

        // Flip the bit
        void invert();

        // Randomly set the bit
        void randomize();

        // Constructor (Internal)
        ValueSetter(uint8_t& data, uint8_t mask);

        // Casts (Interval)
        operator int() const;
        operator bool() const;

        // Internal identifying data
        uint8_t& data;
        uint8_t mask;
    };

    BoolFun(size_t size = 0, int init_val = 0);

    // Set all bits to 'val' (val should be 0 or 1)
    BoolFun& operator=(int val);

    // Copy
    BoolFun& operator=(const BoolFun& val) =default;

    // Get a ValueSetter for modifying a single bit. Inefficient, but convenient for setting. Please use () to get values.
    ValueSetter operator[](size_t pos);

    // Get a single bit (fast)
    bool operator()(size_t pos) const;
    bool operator[](size_t pos) const;
    bool at(size_t pos) const;

    // Boolean function composition
    BoolFun operator* (const BoolFun& other) const;

    // Invert all bits
    void invert();

    // Set to parity; provide subcube mask to make it parity on subset
    void set_parity(size_t mask = -1);

    // Randomly set all bits
    void randomize();

    // Randomly set all by choosing num_ones random inputs to have value 1, else 0
    void choose_randomize(size_t num_ones);

    // Randomly set k pairs of bits each pair with even/odd Hamming weights
    size_t balanced_randomize(size_t k = 100);

    // Built-ins
    // Get and function of size
    static BoolFun andn(size_t size);

    // Get or function of size
    static BoolFun orn(size_t size);

    // Get address function with addr_size address bits,
    // of total size addr_size + 2^addr_size
    // addr_size = 1 => MUX3
    // WARNING: table size is doubly expon in addr_size
    static BoolFun addr(size_t addr_size = 1);

    // Get threshold function of size and level
    // (ham weight >= level --> function is 1)
    static BoolFun threshold(size_t size, size_t level);

    // Get parity function of size, optionally with subcube mask
    static BoolFun parity(size_t size, size_t mask = -1);

    // Get 'threshold-parity' (is there an existing name) function of size
    // which is equal to parity (or -parity) at ham weight < 'level'
    // and equal to 0 >= 'level'
    static BoolFun thresh_parity(size_t size, size_t level);

    // Get Kushlevitz on 6 variables with deg(f) = 3 and s(f) = 6
    static BoolFun kushilevitz();

    // Get not-all-equal function on size variables (default size 3)
    static BoolFun nae(size_t size = 3);

    // Get uniformly random function of size
    static BoolFun random(size_t size);

    // Compute from integer polynomial, where variables take values in {0,1}
    // sets invalid flag if not valid
    static BoolFun from_poly(const std::vector<int64_t>& poly);

    /* Add one to truth table, considering entire truth
       table as an integer; gets next Boolean function,
       for purposes of enumeration.
       Returns false iff resulting truth table is all zero.
       Usage: BoolFun f(N); do{ // use f } while(f.next()); */
    bool next();

    /** Restrict variable i to val (in 0,1) to get
     *  function of size input_size-1 */
    BoolFun restrict(size_t i, int val);

    // ** General **
    // Expectation
    double expectation() const;

    // Variance
    double variance() const;

    // Expectation as {+- 1}^n -> {+-1} polynomial
    double expectation_pm1() const;

    // Variance as {+- 1}^n -> {+-1} polynomial
    double variance_pm1() const;

    // ** Complexity measures **

    // Sensitivity
    size_t s(size_t pos) const;
    size_t s() const;
    size_t s0() const;
    size_t s1() const;

    // Average sensitivity / total influence
    double su() const;

    // Influence of variable i
    double inf(size_t i) const;

    // Fractional bs (fbs)/fractional cert complexity (FC)
    double fbs(size_t pos) const;
    // fbs/FC, outputs an optimal certificate to 'x_out'
    double fbs(size_t pos, std::vector<double>& x_out) const;
    double fbs() const;
    double fbs0() const;
    double fbs1() const;

    // Minimum decision tree depth
    size_t D() const;

    // Degree as a real polynomial
    size_t deg() const;

    // Return Fourier polynomial coefficients
    const std::vector<double>& fourier() const;

    // Return polynomial (not really Fourier)
    // {0,1}^n->{0,1} (which can be shown to have only int coeffs)
    // f2: if true, outputs in F2
    // deg_out: optionally, output the polynomial degree here
    std::vector<int64_t> fourier_01(bool f2 = false, size_t* deg_out = nullptr) const;

    // Print Fourier polynomial prettily, optionally passing a pre-computed polynomial
    void print_fourier(const std::vector<double>* fourier_poly = nullptr) const;
    void print_fourier_01(const std::vector<int64_t>* fourier_poly = nullptr, bool f2 = false) const;

    // Return probability distribution derived from
    // Fourier polynomial where each entry is the total probability
    // for hamming weight i. Size is only input_size+1
    std::vector<double> fourier_dist() const;

    // Return kth moment of Fourier distribution; pass precomputed polynomial
    // from fourier_dist() as second argument to speed up repeat computation
    // limit k -> inf should be deg(f)^k
    double fourier_moment(double k,
            const std::vector<double>& fourier_dist_poly
            = std::vector<double>()) const;

    // Approximate degree (default 2-sided error) as a real polynomial;
    // eps0, eps1 to specify error bound (warning: very slow)
    size_t adeg(double eps0 = 1./3., double eps1 = 1./3.) const;

    // Estimate largest eigenvalue of sensitivity graph using power iteration (crappy own implementation)
    double lambda() const;

    // Estimate largest eigenvalue of sensitivity graph using power iteration (crappy own implementation);
    // outputs principal eigenvector to x_out
    double lambda(std::vector<double>& x_out) const;

    // Compute largest eigenvalue of sensitivity graph using Eigen3,
    // optionally get basis for eigenspace of this eigenvalue.
    double lambda_eigen3(std::vector<std::vector<double> > * x_out = nullptr) const;

    // Compute spectrum of sensitivity graph using Eigen3
    std::vector<double> sens_spectrum() const;

    // Compute diameter of sensitivity graph
    int sens_diam() const;

    // Compute number of connected components of sensitivity graph of
    // size at least min_sz (default 1)
    // Optionally, output sizes to sizes_out
    size_t sens_connected_compon(int min_sz = 1, std::vector<size_t>* sizes_out = nullptr) const;

    // Compute number of non-isolated vertices of sensitivity graph
    size_t sens_noniso_count() const;

    // Write adjacency list of sensitivity graph to file
    void sens_write_adjlist(const std::string& path, bool write_zero_deg_vert = true) const;

    // Try to magically display draw the sensitivity graph using Python/networkx
    void sens_nx_draw(const std::string& python = "python3", bool draw_zero_deg_vert = true) const;

    // Print a styled version of the adjacency matrix of the sensitivity graph (upper triangular)
    void sens_print_adjmat() const;

    // Basic lambda bounds based on sensitivity and Collatz-Wielandt
    double lambda_ub() const;
    double lambda_lb() const;

    // Returns true iff function is balanced
    bool is_balanced() const;

    // Returns true iff function is parity-balanced (same number of ones of each parity)
    bool is_parity_balanced() const;

    // Convert number to bineary with input_size bits
    std::string input_to_bin(size_t x) const;

    // Size data
    size_t input_size, table_size, table_size_bytes, restriction_mask_size;

    // Invalid flag
    bool invalid = false;

    // Internal truth table data (advanced)
    std::vector<uint8_t> data;
private:
    mutable std::vector<double> fourier_data;
};

/** General Boolean function representation for {0,1}^n -> R.
 *  Currently only has a subset of the features. */
class RealBoolFun {
public:
    RealBoolFun(size_t size = 0, double init_val = 0);
    RealBoolFun(const BoolFun& f);
    RealBoolFun(const std::vector<double>& tab);

    // Set all bits to 'val'
    RealBoolFun& operator=(double val);

    // Copy
    RealBoolFun& operator=(const RealBoolFun& val) =default;
    RealBoolFun& operator=(const BoolFun& f);

    // Set from value table
    RealBoolFun& operator=(const std::vector<double>& tab);

    // Get a single value
    double& operator[](size_t pos);

    // Get a single value
    double operator()(size_t pos) const;
    double operator[](size_t pos) const;
    double at(size_t pos) const;

    // Randomly set function values to u.a.r. [lo, hi)
    void randomize(double lo = 0.0, double hi = 1.0);

    // Invert all values (1 - value)
    void invert();

    // Negate all values
    void negate();

    // Map 1->-1 and 0->1
    void neg1p();

    // Take sign of all values (0 will be mapped to 1)
    void sgn();

    // Round all values
    void round();

    // Convert to a BoolFun
    BoolFun sgn_to_boolfun() const;
    BoolFun round_to_boolfun() const;

    // Scale all values by a real
    RealBoolFun& operator *=(double t);
    RealBoolFun& operator /=(double t);
    // Shift all values by a real
    RealBoolFun& operator +=(double t);
    RealBoolFun& operator -=(double t);

    // Randomly set function values to Gaussian (mean, stddev)
    void gaussian_randomize(double mu = 0.0, double sigma = 1.0);

    // Compute from real polynomial, where variables take values in {0, 1}
    static RealBoolFun from_poly(const std::vector<double>& poly);

    // Compute from real polynomial, where variables to take values in {+-1}
    static RealBoolFun from_poly_pm1(const std::vector<double>& poly);

    // Get a random function with values u.a.r. in [lo, hi) 
    static RealBoolFun random(size_t size, double lo = 0.0, double hi = 1.0);

    // ** General **
    // Expectation
    double expectation() const;

    // Variance
    double variance() const;

    // Return Fourier polynomial coefficients (parity basis)
    const std::vector<double>& fourier() const;

    // Print Fourier polynomial prettily, optionally passing a pre-computed polynomail
    void print_fourier(const std::vector<double>* fourier_dist_poly = nullptr) const;

    // ** Complexity measures **

    // Sensitivity
    double s(size_t pos) const;
    double s() const;
    double s0() const;
    double s1() const;

    // Degree as a real polynomial
    size_t deg() const;

    // Approximate degree as a real polynomial
    size_t adeg(double eps0 = 1./3., double eps1 = 1./3.) const;

    // Size data
    size_t input_size, table_size, restriction_mask_size;

    // Invalid flag, currently always false
    bool invalid = false;

    // Internal table data (advanced)
    std::vector<double> data;
private:
    mutable std::vector<double> fourier_data;
};

// Ostream operators, to output nice text when used with cout <<, etc.
std::ostream& operator << (std::ostream& os, const BoolFun& bf);
std::ostream& operator << (std::ostream& os, const BoolFun::ValueSetter& vs);
std::ostream& operator << (std::ostream& os, const RealBoolFun& bf);
}  // namespace bftool
