/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. */

/* Helper methods for fast index mapping builds */

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;
using namespace std;




void build_dataset_item_indices(py::array_t<uint8_t>& dataset_index,
			    const py::array_t<double>& weights,
			    const int64_t size, const bool verbose) {
  /* Given multiple datasets and a weighting array, build samples
   such that it follows those wieghts.*/

  if (verbose) {
    std::cout << "> building indices for multi slice datasets ..." << std::endl;
  }

  // Get the pointer access without the checks.
  auto dataset_index_ptr = dataset_index.mutable_unchecked<1>();
  auto weights_ptr = weights.unchecked<1>();

  // Initialize buffer for number of samples used for each dataset.
  int32_t num_datasets = sizeof(weights)
  int64_t num_samples[num_datasets];
  for(int64_t i = 0; i < num_datasets; ++i) {
    num_samples[i] = 0;
  }

  // For each sample:
  for(int64_t sample_idx = 0; sample_idx < size; ++sample_idx) {

    // Determine where the max error in sampling is happening.
    auto sample_idx_double = std::max(static_cast<double>(sample_idx), 1.0);
    int64_t max_error_index = 0;
    double max_error = static_cast<double>(num_samples[0]) / sample_idx_double - weights_ptr[0];
    for (int64_t dataset_idx = 1; dataset_idx < num_datasets; ++dataset_idx) {
      double error = static_cast<double>(num_samples[dataset_idx]) / sample_idx_double - weights_ptr[dataset_idx];
      if (error < max_error) {
        max_error = error;
        max_error_index = dataset_idx;
      }
    }

    // Populate the indices.
    dataset_index_ptr[sample_idx] = static_cast<uint8_t>(max_error_index);

    // Update the total samples.
    num_samples[max_error_index] += 1;

  }

  // print info
  if (verbose) {
    std::cout << " > sample ratios:" << std::endl;
    for (int64_t dataset_idx = 0; dataset_idx < num_datasets; ++dataset_idx) {
      auto ratio = static_cast<double>(num_samples[dataset_idx]) /
	static_cast<double>(size);
      std::cout << "   dataset " << dataset_idx << ", input: " <<
	weights_ptr[dataset_idx] << ", achieved: " << ratio << std::endl;
    }
  }

}
