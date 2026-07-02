#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // Enables automatic std::vector to Python conversions
#include "math_engine.hpp"

namespace py = pybind11;

/**
 * @brief Bridges Python NumPy arrays with the C++ sliding window math engine.
 * 
 * This function extracts raw memory pointers from Python NumPy arrays so the C++ engine 
 * can process them instantly without copying the data. Once the calculations are done, 
 * it packs the C++ vectors back into an easy-to-use Python dictionary.
 * 
 * @param prices_array NumPy array of execution prices.
 * @param qtys_array NumPy array of traded quantities.
 * @param maker_array NumPy array of boolean flags (true if sell pressure, false if buy).
 * @param window_size The number of ticks to include in each sliding window calculation.
 * @return py::dict A standard Python dictionary containing the calculated time-series arrays.
 */
py::dict calculate_rolling_metrics(py::array_t<double> prices_array, 
                                   py::array_t<double> qtys_array, 
                                   py::array_t<bool> maker_array,
                                   size_t window_size) {
    
    // Request raw buffers from NumPy (grants direct, fast memory access)
    py::buffer_info prices_buf = prices_array.request();
    py::buffer_info qtys_buf = qtys_array.request();
    py::buffer_info maker_buf = maker_array.request();

    // Get the total number of items in the array
    size_t total_size = prices_buf.shape[0];

    // Cast the buffers into raw C++ pointers for the math engine
    const double* prices = static_cast<const double*>(prices_buf.ptr);
    const double* qtys = static_cast<const double*>(qtys_buf.ptr);
    const bool* is_buyer_maker = static_cast<const bool*>(maker_buf.ptr);

    // Execute the sliding window loop from math_engine.hpp
    RollingMetrics res = calculate_rolling_window(prices, qtys, is_buyer_maker, total_size, window_size);

    // Map the C++ outputs directly back to a Python dictionary
    py::dict output;
    output["realized_variance"] = res.realized_variance;
    output["bipower_variation"] = res.bipower_variation;
    output["jump_component"] = res.jump_component;
    output["order_flow_imbalance"] = res.order_flow_imbalance;

    return output;
}

/**
 * @brief Defines the native Python module and exposes the C++ functions.
 * 
 * This macro creates the actual 'bipower_core' module that you import in Python.
 * It ties the C++ function 'calculate_rolling_metrics' to Python and names its arguments.
 */
PYBIND11_MODULE(bipower_core, m) {
    m.doc() = "High-frequency jump-diffusion and microstructure engine";
    m.def("calculate_rolling_metrics", &calculate_rolling_metrics, 
          "Computes sliding window RV, BPV, Jumps, and OFI over a vector slice",
          py::arg("prices_array"), py::arg("qtys_array"), py::arg("maker_array"), py::arg("window_size"));
}