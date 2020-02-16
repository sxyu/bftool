#include <cstddef>
#include <cstdint>

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
namespace {
template<class Vector>
void fft(Vector& in_out) {
    size_t half = 1;
    size_t sz = in_out.size();
    for (size_t tsz = 2; tsz <= sz; tsz <<= 1ULL) {
        for (size_t l = 0; l < sz; l += tsz) {
            for (size_t i = 0; i < half; ++i) {
                auto a = in_out[l+i], b = in_out[l+half+i];
                in_out[l+i] = a+b;
                in_out[l+half+i] = a-b;
            }
        }
        half = tsz;
    }
}
}  // namespace
