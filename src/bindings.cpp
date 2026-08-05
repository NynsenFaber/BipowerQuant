#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // Enables automatic std::vector to Python conversions
#include "math_engine.hpp"

#include <string>

namespace py = pybind11;

/**
 * Arrays are declared with `c_style | forcecast` so pybind11 delivers a
 * contiguous buffer of the right dtype, converting if necessary. Without it a
 * strided view (`prices[::2]`) or an int array would hand the engine a raw
 * pointer whose memory layout does not match how the loop walks it — reading
 * neighbouring elements instead of the ones the caller selected, silently and
 * with plausible-looking results.
 */
using PriceArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using MakerArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;

namespace {

/** Length of a strictly 1-D array, or a ValueError naming the offending input. */
size_t require_1d(const py::buffer_info &buf, const char *name) {
    if (buf.ndim != 1) {
        throw std::invalid_argument(
            std::string(name) + " must be a 1-D array, got " +
            std::to_string(buf.ndim) + " dimensions");
    }
    return static_cast<size_t>(buf.shape[0]);
}

} // namespace

/**
 * @brief Bridges Python NumPy arrays with the C++ sliding window math engine.
 *
 * Extracts raw memory pointers from the NumPy arrays so the engine can process
 * them without copying, then packs the results back into a Python dictionary.
 *
 * The three arrays index the same ticks, so a length mismatch is a caller error
 * rather than something to truncate around: the engine sizes its loop from
 * `prices` alone, and a shorter `qtys` would be read past its end.
 *
 * @param prices_array NumPy array of execution prices.
 * @param qtys_array NumPy array of traded quantities.
 * @param maker_array NumPy array of boolean flags (true if sell pressure, false if buy).
 * @param window_size The number of ticks to include in each sliding window calculation.
 * @return py::dict Python dictionary of the calculated time-series arrays.
 * @throws std::invalid_argument if the inputs are not 1-D arrays of equal length.
 */
py::dict calculate_rolling_metrics(PriceArray prices_array,
                                   PriceArray qtys_array,
                                   MakerArray maker_array,
                                   size_t window_size) {

    // Request raw buffers from NumPy (grants direct, fast memory access)
    py::buffer_info prices_buf = prices_array.request();
    py::buffer_info qtys_buf = qtys_array.request();
    py::buffer_info maker_buf = maker_array.request();

    const size_t total_size = require_1d(prices_buf, "prices_array");
    const size_t qtys_size = require_1d(qtys_buf, "qtys_array");
    const size_t maker_size = require_1d(maker_buf, "maker_array");

    if (qtys_size != total_size || maker_size != total_size) {
        throw std::invalid_argument(
            "prices_array, qtys_array and maker_array must have the same length; got " +
            std::to_string(total_size) + ", " + std::to_string(qtys_size) + " and " +
            std::to_string(maker_size));
    }

    // Cast the buffers into raw C++ pointers for the math engine
    const double* prices = static_cast<const double*>(prices_buf.ptr);
    const double* qtys = static_cast<const double*>(qtys_buf.ptr);
    const bool* is_buyer_maker = static_cast<const bool*>(maker_buf.ptr);

    // The loop is pure arithmetic over buffers already owned by the caller, so
    // it neither touches Python objects nor allocates through the interpreter.
    // Releasing the GIL lets a caller run several windows in parallel threads.
    RollingMetrics res;
    {
        py::gil_scoped_release unlock;
        res = calculate_rolling_window(prices, qtys, is_buyer_maker, total_size, window_size);
    }

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

    // `std::invalid_argument` maps to Python's ValueError, which is what a caller
    // passing mismatched arrays should have to catch.
    m.def("calculate_rolling_metrics", &calculate_rolling_metrics,
          "Computes sliding window RV, BPV, Jumps, and OFI over a vector slice.\n\n"
          "All three arrays must be 1-D and the same length. Windows shorter than\n"
          "2 ticks, or longer than the series, yield empty results.",
          py::arg("prices_array"), py::arg("qtys_array"), py::arg("maker_array"),
          py::arg("window_size"));
}
