#ifndef MATH_ENGINE_H
#define MATH_ENGINE_H

#include <vector>
#include <cmath>
#include <algorithm>

/**
 * Constant scale factor for Bipower Variation calculation (pi / 2).
 */
constexpr double PI_FACTOR = 1.5707963267948966; 

/**
 * @struct RollingMetrics
 * @brief Stores the computed financial metrics over a rolling time-series window.
 * 
 * Each vector index corresponds to the result of a specific window slice.
 */
struct RollingMetrics {
    std::vector<double> realized_variance;     // Sum of squared log returns
    std::vector<double> bipower_variation;     // Product of adjacent absolute log returns
    std::vector<double> jump_component;        // Non-negative difference between RV and BPV
    std::vector<double> order_flow_imbalance;  // Net buying/selling pressure based on volume
};

/**
 * @brief Computes RV, BPV, Jumps, and OFI over a sliding window.
 * 
 * Iterates over raw price and volume arrays, applying a sliding window to compute 
 * microstructural features. It pre-allocates memory to maximize execution speed 
 * and avoid dynamic resizing overhead.
 * 
 * @param prices Pointer to the flat array of execution prices.
 * @param qtys Pointer to the flat array of traded quantities.
 * @param is_buyer_maker Pointer to the boolean array indicating sell pressure.
 * @param total_size The total number of elements in the input arrays.
 * @param window_size The number of ticks to include in each sliding window calculation.
 * @return RollingMetrics A struct containing fully populated vectors of the calculated features.
 */
inline RollingMetrics calculate_rolling_window(
    const double* prices, 
    const double* qtys, 
    const bool* is_buyer_maker, 
    size_t total_size, 
    size_t window_size) 
{
    RollingMetrics result;
    
    // If the dataset is smaller than the window, return empty vectors
    if (total_size < window_size || window_size < 2) {
        return result;
    }

    size_t num_windows = total_size - window_size + 1;
    
    // Pre-allocate memory for extreme speed (prevents dynamic reallocation during the loop)
    result.realized_variance.reserve(num_windows);
    result.bipower_variation.reserve(num_windows);
    result.jump_component.reserve(num_windows);
    result.order_flow_imbalance.reserve(num_windows);

    // Slide the window across the entire flat array
    for (size_t start_idx = 0; start_idx < num_windows; ++start_idx) {
        double rv_sum = 0.0;
        double bpv_sum = 0.0;
        double ofi_sum = 0.0;
        
        // Calculate metrics strictly inside the current window
        for (size_t i = 0; i < window_size; ++i) {
            size_t current_idx = start_idx + i;
            
            // 1. Order Flow Imbalance (OFI)
            if (is_buyer_maker[current_idx]) {
                ofi_sum -= qtys[current_idx]; // Sell pressure decreases OFI
            } else {
                ofi_sum += qtys[current_idx]; // Buy pressure increases OFI
            }

            // 2. Variance Metrics (requires at least 1 previous tick in the window)
            if (i > 0) {
                double r_t = std::log(prices[current_idx]) - std::log(prices[current_idx - 1]);
                rv_sum += r_t * r_t;

                if (i > 1) {
                    double r_t_minus_1 = std::log(prices[current_idx - 1]) - std::log(prices[current_idx - 2]);
                    bpv_sum += std::abs(r_t) * std::abs(r_t_minus_1);
                }
            }
        }
        
        double bpv = PI_FACTOR * bpv_sum;
        double jumps = std::max(rv_sum - bpv, 0.0);

        // Push calculated metrics for this specific window into the output arrays
        result.realized_variance.push_back(rv_sum);
        result.bipower_variation.push_back(bpv);
        result.jump_component.push_back(jumps);
        result.order_flow_imbalance.push_back(ofi_sum);
    }

    return result;
}

#endif // MATH_ENGINE_H