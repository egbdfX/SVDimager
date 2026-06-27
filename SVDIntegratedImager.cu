#include <cstddef>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "fitsio.h"
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/BasicSL/Complex.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableRecord.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#define M_PI 3.14159265358979323846
#define CHEB_MAX_MOMENTS 32
#define CHEB_TARGET_ERROR 1.0e-2f

struct HostMeasurementSetData {
    std::size_t num_rows = 0;
    std::size_t num_channels = 0;
    std::size_t num_samples = 0;
    std::vector<float> uvw;
    std::vector<float> frequencies_hz;
    std::vector<float> vis0_real;
    std::vector<float> vis0_imag;
    std::vector<float> vis3_real;
    std::vector<float> vis3_imag;
    std::vector<std::uint8_t> flag0;
    std::vector<std::uint8_t> flag3;
    std::vector<float> weight0;
    std::vector<float> weight3;
};

struct DevicePreprocessBuffers {
    float* d_bin = nullptr;
    float* d_vin = nullptr;
    float* d_vis_real = nullptr;
    float* d_vis_imag = nullptr;
    std::size_t num_samples = 0;
};

HostMeasurementSetData read_measurement_set(const std::string& ms_path);
HostMeasurementSetData read_measurement_set_rows(
    const std::string& ms_path,
    const std::vector<std::size_t>& selected_rows
);
void preprocess_measurement_set_gpu(
    const HostMeasurementSetData& host_data,
    DevicePreprocessBuffers& device_buffers
);
void free_device_preprocess_buffers(DevicePreprocessBuffers& device_buffers);
int FIpipeDevice(
    const DevicePreprocessBuffers& preprocess_buffers,
    float* dirty_image,
    std::size_t image_size,
    float cell_size
);

namespace {

std::size_t flattened_index(std::size_t row, std::size_t chan, std::size_t num_rows) {
    return row + chan * num_rows;
}

struct RowAxisLayout {
    int channel_axis = 0;
    int pol_axis = 1;
    std::size_t num_channels = 0;
    std::size_t num_pols = 0;
};

RowAxisLayout detect_row_axis_layout(const casacore::IPosition& shape) {
    if (shape.nelements() != 2) {
        throw std::runtime_error("Expected a 2D row shape");
    }

    RowAxisLayout layout;
    const std::size_t dim0 = static_cast<std::size_t>(shape[0]);
    const std::size_t dim1 = static_cast<std::size_t>(shape[1]);

    if (dim1 == 4 || dim1 == 2 || dim1 == 1) {
        layout.channel_axis = 0;
        layout.pol_axis = 1;
        layout.num_channels = dim0;
        layout.num_pols = dim1;
        return layout;
    }
    if (dim0 == 4 || dim0 == 2 || dim0 == 1) {
        layout.channel_axis = 1;
        layout.pol_axis = 0;
        layout.num_channels = dim1;
        layout.num_pols = dim0;
        return layout;
    }

    if (dim0 >= dim1) {
        layout.channel_axis = 0;
        layout.pol_axis = 1;
        layout.num_channels = dim0;
        layout.num_pols = dim1;
    } else {
        layout.channel_axis = 1;
        layout.pol_axis = 0;
        layout.num_channels = dim1;
        layout.num_pols = dim0;
    }
    return layout;
}

casacore::IPosition make_row_index(const RowAxisLayout& layout, std::size_t chan, std::size_t pol) {
    casacore::IPosition index(2, 0, 0);
    index[layout.channel_axis] = static_cast<int>(chan);
    index[layout.pol_axis] = static_cast<int>(pol);
    return index;
}

void check_fits_status(int status, const std::string& context) {
    if (status != 0) {
        fits_report_error(stderr, status);
        throw std::runtime_error(context);
    }
}

void write_fits_2d(
    const std::string& filename,
    const std::vector<float>& values,
    long axis0,
    long axis1
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    long naxes[2] = {axis0, axis1};
    const std::string output_name = "!" + filename;

    fits_create_file(&fptr, output_name.c_str(), &status);
    check_fits_status(status, "Failed to create FITS file: " + filename);

    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    check_fits_status(status, "Failed to create FITS image: " + filename);

    const long total = axis0 * axis1;
    fits_write_img(fptr, TFLOAT, 1, total, const_cast<float*>(values.data()), &status);
    check_fits_status(status, "Failed to write FITS image: " + filename);

    fits_close_file(fptr, &status);
    check_fits_status(status, "Failed to close FITS file: " + filename);
}

}

HostMeasurementSetData read_measurement_set_rows(
    const std::string& ms_path,
    const std::vector<std::size_t>& selected_rows
) {
    using casacore::Array;
    using casacore::Complex;
    using casacore::ROArrayColumn;
    using casacore::Table;
    using casacore::Vector;

    Table vis(ms_path, Table::Old);
    const std::size_t table_rows = vis.nrow();
    const std::size_t num_rows = selected_rows.size();

    if (num_rows == 0) {
        throw std::runtime_error("No rows selected from " + ms_path);
    }

    ROArrayColumn<double> uvw_col(vis, "UVW");
    ROArrayColumn<Complex> data_col(vis, "DATA");
    ROArrayColumn<bool> flag_col(vis, "FLAG");

    const bool has_weight_spectrum = vis.tableDesc().isColumn("WEIGHT_SPECTRUM");
    std::unique_ptr<ROArrayColumn<float>> weight_col;
    if (has_weight_spectrum) {
        weight_col.reset(new ROArrayColumn<float>(vis, "WEIGHT_SPECTRUM"));
    }

    Table spw = vis.keywordSet().asTable("SPECTRAL_WINDOW");
    ROArrayColumn<double> chan_freq_col(spw, "CHAN_FREQ");

    std::vector<float> frequencies_hz;
    for (std::size_t row = 0; row < spw.nrow(); ++row) {
        Vector<double> row_freq;
        chan_freq_col.get(static_cast<casacore::rownr_t>(row), row_freq);
        for (casacore::uInt chan = 0; chan < row_freq.size(); ++chan) {
            frequencies_hz.push_back(static_cast<float>(row_freq[chan]));
        }
    }

    if (frequencies_hz.empty()) {
        throw std::runtime_error("No frequencies found in SPECTRAL_WINDOW for " + ms_path);
    }

    HostMeasurementSetData result;
    result.num_rows = num_rows;
    result.num_channels = frequencies_hz.size();
    result.num_samples = result.num_rows * result.num_channels;
    result.frequencies_hz = std::move(frequencies_hz);
    result.uvw.resize(result.num_rows * 3);
    result.vis0_real.resize(result.num_samples);
    result.vis0_imag.resize(result.num_samples);
    result.vis3_real.resize(result.num_samples);
    result.vis3_imag.resize(result.num_samples);
    result.flag0.resize(result.num_samples);
    result.flag3.resize(result.num_samples);
    result.weight0.resize(result.num_samples, 1.0f);
    result.weight3.resize(result.num_samples, 1.0f);

    for (std::size_t local_row = 0; local_row < result.num_rows; ++local_row) {
        const std::size_t row = selected_rows[local_row];
        if (row >= table_rows) {
            throw std::runtime_error("Selected row is outside " + ms_path);
        }

        Vector<double> uvw_row;
        Array<Complex> data_row;
        Array<bool> flag_row;

        uvw_col.get(static_cast<casacore::rownr_t>(row), uvw_row);
        data_col.get(static_cast<casacore::rownr_t>(row), data_row);
        flag_col.get(static_cast<casacore::rownr_t>(row), flag_row);

        if (uvw_row.size() != 3) {
            throw std::runtime_error("UVW does not have 3 entries in " + ms_path);
        }

        const auto data_shape = data_row.shape();
        if (data_shape.nelements() != 2) {
            throw std::runtime_error("DATA does not have 2 dimensions in " + ms_path);
        }

        const RowAxisLayout row_layout = detect_row_axis_layout(data_shape);
        if (row_layout.num_pols < 4) {
            throw std::runtime_error("DATA has fewer than 4 polarisations in " + ms_path);
        }

        Array<float> weight_row;
        if (has_weight_spectrum) {
            weight_col->get(static_cast<casacore::rownr_t>(row), weight_row);
        }

        result.uvw[local_row * 3 + 0] = static_cast<float>(uvw_row[0]);
        result.uvw[local_row * 3 + 1] = static_cast<float>(uvw_row[1]);
        result.uvw[local_row * 3 + 2] = static_cast<float>(uvw_row[2]);

        const std::size_t vis_channels = row_layout.num_channels;
        if (vis_channels == 0) {
            throw std::runtime_error("DATA has zero channels in " + ms_path);
        }

        for (std::size_t chan = 0; chan < result.num_channels; ++chan) {
            const std::size_t dst_idx = flattened_index(local_row, chan, result.num_rows);
            const std::size_t chan_in_row = chan % vis_channels;
            const casacore::IPosition pol0_index = make_row_index(row_layout, chan_in_row, 0);
            const casacore::IPosition pol3_index = make_row_index(row_layout, chan_in_row, 3);
            const Complex vis0_value = data_row(pol0_index);
            const Complex vis3_value = data_row(pol3_index);

            result.vis0_real[dst_idx] = vis0_value.real();
            result.vis0_imag[dst_idx] = vis0_value.imag();
            result.vis3_real[dst_idx] = vis3_value.real();
            result.vis3_imag[dst_idx] = vis3_value.imag();
            result.flag0[dst_idx] = flag_row(pol0_index) ? 1U : 0U;
            result.flag3[dst_idx] = flag_row(pol3_index) ? 1U : 0U;

            if (has_weight_spectrum) {
                result.weight0[dst_idx] = weight_row(pol0_index);
                result.weight3[dst_idx] = weight_row(pol3_index);
            }
        }
    }

    return result;
}

HostMeasurementSetData read_measurement_set(const std::string& ms_path) {
    using casacore::Table;

    Table vis(ms_path, Table::Old);
    std::vector<std::size_t> all_rows(vis.nrow());
    for (std::size_t row = 0; row < all_rows.size(); ++row) {
        all_rows[row] = row;
    }
    return read_measurement_set_rows(ms_path, all_rows);
}

namespace {

constexpr float kSpeedOfLight = 299792458.0f;

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        const cudaError_t status__ = (call);                                     \
        if (status__ != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(status__));              \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        const cublasStatus_t status__ = (call);                                  \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error("cuBLAS call failed");                      \
        }                                                                        \
    } while (0)

#define CHECK_CUSOLVER(call)                                                     \
    do {                                                                         \
        const cusolverStatus_t status__ = (call);                                \
        if (status__ != CUSOLVER_STATUS_SUCCESS) {                               \
            throw std::runtime_error("cuSOLVER call failed");                    \
        }                                                                        \
    } while (0)

__global__ void expand_baselines_kernel(
    const float* uvw,
    const float* frequencies_hz,
    float* baselines_col_major,
    std::size_t num_rows,
    std::size_t num_channels
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t num_samples = num_rows * num_channels;
    if (idx >= num_samples) {
        return;
    }

    const std::size_t row = idx % num_rows;
    const std::size_t chan = idx / num_rows;
    const float scale = frequencies_hz[chan] / kSpeedOfLight;

    baselines_col_major[0 * num_samples + idx] = uvw[row * 3 + 0] * scale;
    baselines_col_major[1 * num_samples + idx] = uvw[row * 3 + 1] * scale;
    baselines_col_major[2 * num_samples + idx] = uvw[row * 3 + 2] * scale;
}

__global__ void accumulate_sums_kernel(
    const float* baselines_col_major,
    float* sums,
    std::size_t num_samples
) {
    __shared__ float local_sums[3 * 256];
    const int tid = threadIdx.x;
    const std::size_t idx = blockIdx.x * blockDim.x + tid;

    local_sums[tid + 0 * blockDim.x] = 0.0f;
    local_sums[tid + 1 * blockDim.x] = 0.0f;
    local_sums[tid + 2 * blockDim.x] = 0.0f;

    if (idx < num_samples) {
        local_sums[tid + 0 * blockDim.x] = baselines_col_major[0 * num_samples + idx];
        local_sums[tid + 1 * blockDim.x] = baselines_col_major[1 * num_samples + idx];
        local_sums[tid + 2 * blockDim.x] = baselines_col_major[2 * num_samples + idx];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_sums[tid + 0 * blockDim.x] += local_sums[tid + stride + 0 * blockDim.x];
            local_sums[tid + 1 * blockDim.x] += local_sums[tid + stride + 1 * blockDim.x];
            local_sums[tid + 2 * blockDim.x] += local_sums[tid + stride + 2 * blockDim.x];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&sums[0], local_sums[0 * blockDim.x]);
        atomicAdd(&sums[1], local_sums[1 * blockDim.x]);
        atomicAdd(&sums[2], local_sums[2 * blockDim.x]);
    }
}

__global__ void center_baselines_kernel(
    const float* baselines_col_major,
    const float* means,
    float* centered_col_major,
    std::size_t num_samples
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) {
        return;
    }

    centered_col_major[0 * num_samples + idx] = baselines_col_major[0 * num_samples + idx] - means[0];
    centered_col_major[1 * num_samples + idx] = baselines_col_major[1 * num_samples + idx] - means[1];
    centered_col_major[2 * num_samples + idx] = baselines_col_major[2 * num_samples + idx] - means[2];
}

__global__ void normalize_sums_kernel(float* sums, std::size_t num_samples) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 3) {
        sums[idx] /= static_cast<float>(num_samples);
    }
}

__global__ void build_basis_kernel(
    const float* eigenvectors_col_major,
    float* basis_row_major,
    float* vin_row_major
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int axis = 0; axis < 3; ++axis) {
        basis_row_major[0 * 3 + axis] = eigenvectors_col_major[axis + 2 * 3];
        basis_row_major[1 * 3 + axis] = eigenvectors_col_major[axis + 1 * 3];
        basis_row_major[2 * 3 + axis] = eigenvectors_col_major[axis + 0 * 3];
        vin_row_major[0 * 3 + axis] = basis_row_major[0 * 3 + axis];
        vin_row_major[1 * 3 + axis] = basis_row_major[1 * 3 + axis];
        vin_row_major[2 * 3 + axis] = basis_row_major[2 * 3 + axis];
    }
}

__global__ void project_bin_kernel(
    const float* baselines_col_major,
    const float* basis_row_major,
    float* bin_row_major,
    std::size_t num_samples
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) {
        return;
    }

    const float x0 = baselines_col_major[0 * num_samples + idx];
    const float x1 = baselines_col_major[1 * num_samples + idx];
    const float x2 = baselines_col_major[2 * num_samples + idx];

    bin_row_major[idx * 3 + 0] =
        x0 * basis_row_major[0 * 3 + 0] +
        x1 * basis_row_major[0 * 3 + 1] +
        x2 * basis_row_major[0 * 3 + 2];
    bin_row_major[idx * 3 + 1] =
        x0 * basis_row_major[1 * 3 + 0] +
        x1 * basis_row_major[1 * 3 + 1] +
        x2 * basis_row_major[1 * 3 + 2];
    bin_row_major[idx * 3 + 2] =
        x0 * basis_row_major[2 * 3 + 0] +
        x1 * basis_row_major[2 * 3 + 1] +
        x2 * basis_row_major[2 * 3 + 2];
}

__global__ void align_pca_signs_kernel(
      float* basis_row_major,
      float* vin_row_major,
      float* bin_row_major,
      std::size_t num_samples
) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
          for (int component = 0; component < 3; ++component) {
              std::size_t max_sample = 0;
              float max_value = fabsf(bin_row_major[component]);

              for (std::size_t sample = 1; sample < num_samples; ++sample) {
                  const float candidate = fabsf(bin_row_major[sample * 3 + component]);
                  if (candidate > max_value) {
                      max_value = candidate;
                      max_sample = sample;
                  }
              }

              if (bin_row_major[max_sample * 3 + component] < 0.0f) {
                  for (int axis = 0; axis < 3; ++axis) {
                      basis_row_major[component * 3 + axis] =
                          -basis_row_major[component * 3 + axis];
                      vin_row_major[component * 3 + axis] =
                          -vin_row_major[component * 3 + axis];
                  }

                  for (std::size_t sample = 0; sample < num_samples; ++sample) {
                      bin_row_major[sample * 3 + component] =
                          -bin_row_major[sample * 3 + component];
                  }
              }
          }
      }
  }

__global__ void collapse_visibility_kernel(
    const float* vis0_real,
    const float* vis0_imag,
    const float* vis3_real,
    const float* vis3_imag,
    const std::uint8_t* flag0,
    const std::uint8_t* flag3,
    const float* weight0,
    const float* weight3,
    float* vis_real,
    float* vis_imag,
    std::size_t num_samples
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) {
        return;
    }

    const float w0 = flag0[idx] ? 0.0f : weight0[idx];
    const float w3 = flag3[idx] ? 0.0f : weight3[idx];
    const float w_sum = w0 + w3;

    if (w_sum > 0.0f) {
        vis_real[idx] = (vis0_real[idx] * w0 + vis3_real[idx] * w3) / w_sum;
        vis_imag[idx] = (vis0_imag[idx] * w0 + vis3_imag[idx] * w3) / w_sum;
    } else {
        vis_real[idx] = 0.0f;
        vis_imag[idx] = 0.0f;
    }
}

template <typename T>
T* cuda_alloc_and_copy(const std::vector<T>& host_values, cudaStream_t stream) {
    T* device_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&device_ptr, host_values.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpyAsync(
        device_ptr,
        host_values.data(),
        host_values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream
    ));
    return device_ptr;
}

}

void preprocess_measurement_set_gpu(
    const HostMeasurementSetData& host_data,
    DevicePreprocessBuffers& device_buffers
) {
    if (host_data.num_samples == 0) {
        throw std::runtime_error("Measurement set is empty");
    }

    cublasHandle_t cublas_handle = nullptr;
    cusolverDnHandle_t cusolver_handle = nullptr;
    cudaStream_t stream = nullptr;

    float* d_uvw = nullptr;
    float* d_freq = nullptr;
    float* d_vis0_real = nullptr;
    float* d_vis0_imag = nullptr;
    float* d_vis3_real = nullptr;
    float* d_vis3_imag = nullptr;
    std::uint8_t* d_flag0 = nullptr;
    std::uint8_t* d_flag3 = nullptr;
    float* d_weight0 = nullptr;
    float* d_weight3 = nullptr;
    float* d_baselines = nullptr;
    float* d_centered = nullptr;
    float* d_sums = nullptr;
    float* d_covariance = nullptr;
    float* d_eigenvalues = nullptr;
    float* d_basis = nullptr;
    int* d_info = nullptr;
    float* d_workspace = nullptr;

    try {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));
        CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle, stream));

        d_uvw = cuda_alloc_and_copy(host_data.uvw, stream);
        d_freq = cuda_alloc_and_copy(host_data.frequencies_hz, stream);
        d_vis0_real = cuda_alloc_and_copy(host_data.vis0_real, stream);
        d_vis0_imag = cuda_alloc_and_copy(host_data.vis0_imag, stream);
        d_vis3_real = cuda_alloc_and_copy(host_data.vis3_real, stream);
        d_vis3_imag = cuda_alloc_and_copy(host_data.vis3_imag, stream);
        d_flag0 = cuda_alloc_and_copy(host_data.flag0, stream);
        d_flag3 = cuda_alloc_and_copy(host_data.flag3, stream);
        d_weight0 = cuda_alloc_and_copy(host_data.weight0, stream);
        d_weight3 = cuda_alloc_and_copy(host_data.weight3, stream);

        const std::size_t num_samples = host_data.num_samples;
        const int threads = 256;
        const int blocks = static_cast<int>((num_samples + threads - 1) / threads);

        CHECK_CUDA(cudaMalloc(&d_baselines, 3 * num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_centered, 3 * num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sums, 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_covariance, 9 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_eigenvalues, 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_basis, 9 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_vin, 9 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_bin, 3 * num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_vis_real, num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_vis_imag, num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        expand_baselines_kernel<<<blocks, threads, 0, stream>>>(
            d_uvw,
            d_freq,
            d_baselines,
            host_data.num_rows,
            host_data.num_channels
        );
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemsetAsync(d_sums, 0, 3 * sizeof(float), stream));
        accumulate_sums_kernel<<<blocks, threads, 0, stream>>>(d_baselines, d_sums, num_samples);
        CHECK_CUDA(cudaGetLastError());
        normalize_sums_kernel<<<1, 32, 0, stream>>>(d_sums, num_samples);
        CHECK_CUDA(cudaGetLastError());
        center_baselines_kernel<<<blocks, threads, 0, stream>>>(d_baselines, d_sums, d_centered, num_samples);
        CHECK_CUDA(cudaGetLastError());

        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            3,
            3,
            static_cast<int>(num_samples),
            &alpha,
            d_centered,
            static_cast<int>(num_samples),
            d_centered,
            static_cast<int>(num_samples),
            &beta,
            d_covariance,
            3
        ));

        int workspace_size = 0;
        CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_UPPER,
            3,
            d_covariance,
            3,
            d_eigenvalues,
            &workspace_size
        ));
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size * sizeof(float)));

        CHECK_CUSOLVER(cusolverDnSsyevd(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_UPPER,
            3,
            d_covariance,
            3,
            d_eigenvalues,
            d_workspace,
            workspace_size,
            d_info
        ));

        build_basis_kernel<<<1, 1, 0, stream>>>(d_covariance, d_basis, device_buffers.d_vin);
        CHECK_CUDA(cudaGetLastError());

        project_bin_kernel<<<blocks, threads, 0, stream>>>(
            d_baselines,
            d_basis,
            device_buffers.d_bin,
            num_samples
        );
        CHECK_CUDA(cudaGetLastError());

        align_pca_signs_kernel<<<1, 1, 0, stream>>>(
            d_basis,
            device_buffers.d_vin,
            device_buffers.d_bin,
            num_samples
        );
        CHECK_CUDA(cudaGetLastError());

        collapse_visibility_kernel<<<blocks, threads, 0, stream>>>(
            d_vis0_real,
            d_vis0_imag,
            d_vis3_real,
            d_vis3_imag,
            d_flag0,
            d_flag3,
            d_weight0,
            d_weight3,
            device_buffers.d_vis_real,
            device_buffers.d_vis_imag,
            num_samples
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));

        device_buffers.num_samples = num_samples;
    } catch (...) {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        free_device_preprocess_buffers(device_buffers);
        cudaFree(d_uvw);
        cudaFree(d_freq);
        cudaFree(d_vis0_real);
        cudaFree(d_vis0_imag);
        cudaFree(d_vis3_real);
        cudaFree(d_vis3_imag);
        cudaFree(d_flag0);
        cudaFree(d_flag3);
        cudaFree(d_weight0);
        cudaFree(d_weight3);
        cudaFree(d_baselines);
        cudaFree(d_centered);
        cudaFree(d_sums);
        cudaFree(d_covariance);
        cudaFree(d_eigenvalues);
        cudaFree(d_basis);
        cudaFree(d_info);
        cudaFree(d_workspace);
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
        }
        if (cusolver_handle != nullptr) {
            cusolverDnDestroy(cusolver_handle);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        throw;
    }

    cudaFree(d_uvw);
    cudaFree(d_freq);
    cudaFree(d_vis0_real);
    cudaFree(d_vis0_imag);
    cudaFree(d_vis3_real);
    cudaFree(d_vis3_imag);
    cudaFree(d_flag0);
    cudaFree(d_flag3);
    cudaFree(d_weight0);
    cudaFree(d_weight3);
    cudaFree(d_baselines);
    cudaFree(d_centered);
    cudaFree(d_sums);
    cudaFree(d_covariance);
    cudaFree(d_eigenvalues);
    cudaFree(d_basis);
    cudaFree(d_info);
    cudaFree(d_workspace);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolver_handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void free_device_preprocess_buffers(DevicePreprocessBuffers& device_buffers) {
    cudaFree(device_buffers.d_bin);
    cudaFree(device_buffers.d_vin);
    cudaFree(device_buffers.d_vis_real);
    cudaFree(device_buffers.d_vis_imag);
    device_buffers.d_bin = nullptr;
    device_buffers.d_vin = nullptr;
    device_buffers.d_vis_real = nullptr;
    device_buffers.d_vis_imag = nullptr;
    device_buffers.num_samples = 0;
}

__constant__ float quadrature_nodes[14] = {
	0.9964425,0.98130317,0.95425928,0.91563303,0.86589252,
	0.80564137,0.73561088,0.65665109,0.56972047,0.47587422,
	0.37625152,0.27206163,0.16456928,0.05507929
};
__constant__ float quadrature_weights[14] = {
	0.00912428,0.02113211,0.03290143,0.04427293,0.05510735,
	0.06527292,0.07464621,0.08311342,0.09057174,0.09693066,
	0.10211297,0.10605577,0.10871119,0.11004701
};
__constant__ float quadrature_kernel[14] = {
	7.71381676e-07,4.06901586e-06,2.09164257e-05,1.01923695e-04,
	4.61199576e-04,1.90183990e-03,7.02391280e-03,2.28652529e-02,
	6.46725327e-02,1.56933676e-01,3.23208771e-01,5.60024174e-01,
	8.10934691e-01,9.76937533e-01
};

__device__ float exp_semicircle(const float beta, float x){
	const float xx = x*x;
    
	return ((xx > float(1.0)) ? float(0.0) : exp(beta*(sqrt(float(1.0) - xx) - float(1.0))));
}

__device__ void chebyshev_sequence(float s, int num_terms, float* values) {
    values[0] = 1.0f;
    if (num_terms == 1) {
        return;
    }
    values[1] = s;
    for (int n = 2; n < num_terms; ++n) {
        values[n] = 2.0f * s * values[n - 1] - values[n - 2];
    }
}

__device__ void bessel_sequence(float alpha, int max_order, float* jvals) {
    const float half_alpha = 0.5f * alpha;
    const float quarter_alpha_sq = half_alpha * half_alpha;
    for (int n = 0; n <= max_order; ++n) {
        float term = 1.0f;
        for (int m = 1; m <= n; ++m) {
            term *= half_alpha / static_cast<float>(m);
        }

        float sum = term;
        for (int k = 1; k < 32; ++k) {
            term *= -quarter_alpha_sq / (static_cast<float>(k) * static_cast<float>(n + k));
            sum += term;
            if (fabsf(term) < 1.0e-7f * (1.0f + fabsf(sum))) {
                break;
            }
        }
        jvals[n] = sum;
    }
}

__device__ float real_complex_product(float ar, float ai, cufftComplex z) {
	return ar * z.x - ai * z.y;
}

struct HostChebyshevSelection {
	float r3_min = 0.0f;
	float r3_max = 0.0f;
	float r3_centre = 0.0f;
	float r3_half_range = 1.0e-6f;
	float inv_r3_half_range = 1.0e6f;
	float max_abs_t3 = 0.0f;
	size_t valid_pixels = 0;
	float alpha_max = 0.0f;
};

HostChebyshevSelection finish_host_chebyshev_selection(
        float r3_min,
        float r3_max,
        float max_abs_t3,
        size_t valid_pixels
) {
    HostChebyshevSelection result;
    result.r3_min = r3_min;
    result.r3_max = r3_max;
    result.r3_centre = 0.5f * (result.r3_max + result.r3_min);
    result.r3_half_range = fmaxf(0.5f * (result.r3_max - result.r3_min), 1.0e-6f);
    result.inv_r3_half_range = 1.0f / result.r3_half_range;
    result.max_abs_t3 = max_abs_t3;
    result.valid_pixels = valid_pixels;
    result.alpha_max = 2.0f * M_PI * result.max_abs_t3 * result.r3_half_range;
    return result;
}

__global__ void r3_range_blocks_kernel(
        const float* Bin,
        float* block_min,
        float* block_max,
        size_t num_baselines
) {
    __shared__ float local_min[256];
    __shared__ float local_max[256];
    const int tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + tid;

    float r3_min = CUDART_INF_F;
    float r3_max = -CUDART_INF_F;
    if (idx < num_baselines) {
        const float r3 = Bin[idx * 3 + 2];
        r3_min = r3;
        r3_max = r3;
    }
    local_min[tid] = r3_min;
    local_max[tid] = r3_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_min[tid] = fminf(local_min[tid], local_min[tid + stride]);
            local_max[tid] = fmaxf(local_max[tid], local_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min[blockIdx.x] = local_min[0];
        block_max[blockIdx.x] = local_max[0];
    }
}

__global__ void t3_stats_blocks_kernel(
        const float Vin[3][3],
        float* block_max_abs_t3,
        unsigned int* block_valid_pixels,
        size_t image_size,
        float cell_size
) {
    __shared__ float local_max[256];
    __shared__ unsigned int local_valid[256];
    const int tid = threadIdx.x;
    const size_t pixel = blockIdx.x * blockDim.x + tid;
    const size_t pixel_count = image_size * image_size;

    float max_abs_t3 = 0.0f;
    unsigned int valid = 0;
    if (pixel < pixel_count) {
        const size_t y = pixel / image_size;
        const size_t x = pixel - y * image_size;
        const float half_image_size = static_cast<float>(image_size) / 2.0f;
        const float t1 = cell_size * (static_cast<float>(y) - half_image_size);
        const float t2 = cell_size * (static_cast<float>(x) - half_image_size);
        const float v13 = Vin[0][2];
        const float v23 = Vin[1][2];
        const float v33 = Vin[2][2];
        const float root_arg = 1.0f - (t1 + v13) * (t1 + v13) - (t2 + v23) * (t2 + v23);
        if (root_arg >= 0.0f) {
            const float branch_sign = (v33 >= 0.0f) ? 1.0f : -1.0f;
            const float t3 = -v33 + branch_sign * sqrtf(root_arg);
            max_abs_t3 = fabsf(t3);
            valid = 1;
        }
    }
    local_max[tid] = max_abs_t3;
    local_valid[tid] = valid;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_max[tid] = fmaxf(local_max[tid], local_max[tid + stride]);
            local_valid[tid] += local_valid[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max_abs_t3[blockIdx.x] = local_max[0];
        block_valid_pixels[blockIdx.x] = local_valid[0];
    }
}

__global__ void finalize_cheb_stats_kernel(
        const float* r3_block_min,
        const float* r3_block_max,
        const float* t3_block_max,
        const unsigned int* t3_block_valid,
        float* stats_out,
        size_t r3_blocks,
        size_t t3_blocks
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float r3_min = r3_block_min[0];
    float r3_max = r3_block_max[0];
    for (size_t i = 1; i < r3_blocks; ++i) {
        r3_min = fminf(r3_min, r3_block_min[i]);
        r3_max = fmaxf(r3_max, r3_block_max[i]);
    }

    float max_abs_t3 = 0.0f;
    unsigned int valid_pixels = 0;
    for (size_t i = 0; i < t3_blocks; ++i) {
        max_abs_t3 = fmaxf(max_abs_t3, t3_block_max[i]);
        valid_pixels += t3_block_valid[i];
    }

    stats_out[0] = r3_min;
    stats_out[1] = r3_max;
    stats_out[2] = max_abs_t3;
    stats_out[3] = static_cast<float>(valid_pixels);
}

__device__ void atomic_max_float(float* address, float value) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) {
            break;
        }
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__global__ void cheb_bucket_error_kernel(
        float alpha_max,
        float* bucket_errors,
        int alpha_samples,
        int s_samples
) {
    const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_samples = alpha_samples * s_samples;
    if (sample_idx >= total_samples) {
        return;
    }

    const int alpha_idx = sample_idx / s_samples;
    const int s_idx = sample_idx - alpha_idx * s_samples;
    const float alpha = (alpha_samples == 1)
        ? alpha_max
        : alpha_max * static_cast<float>(alpha_idx) / static_cast<float>(alpha_samples - 1);
    const float s = -1.0f + 2.0f * static_cast<float>(s_idx) / static_cast<float>(s_samples - 1);

    float jvals[CHEB_MAX_MOMENTS];
    bessel_sequence(alpha, CHEB_MAX_MOMENTS - 1, jvals);

    const float exact_r = cosf(alpha * s);
    const float exact_i = sinf(alpha * s);
    float approx_r = 0.0f;
    float approx_i = 0.0f;
    float tnm2 = 1.0f;
    float tnm1 = s;

    for (int n = 0; n < CHEB_MAX_MOMENTS; ++n) {
        float tn = 1.0f;
        if (n == 1) {
            tn = tnm1;
        } else if (n > 1) {
            tn = 2.0f * s * tnm1 - tnm2;
            tnm2 = tnm1;
            tnm1 = tn;
        }

        if (n == 0) {
            approx_r += jvals[n] * tn;
        } else {
            const float amplitude = 2.0f * jvals[n] * tn;
            switch (n & 3) {
                case 0:
                    approx_r += amplitude;
                    break;
                case 1:
                    approx_i += amplitude;
                    break;
                case 2:
                    approx_r -= amplitude;
                    break;
                default:
                    approx_i -= amplitude;
                    break;
            }
        }

        int bucket = -1;
        if (n == 7) {
            bucket = 0;
        } else if (n == 15) {
            bucket = 1;
        } else if (n == 23) {
            bucket = 2;
        } else if (n == 31) {
            bucket = 3;
        }

        if (bucket >= 0) {
            const float dr = exact_r - approx_r;
            const float di = exact_i - approx_i;
            const float err = sqrtf(dr * dr + di * di);
            atomic_max_float(&bucket_errors[bucket], err);
        }
    }
}

__global__ void convolveKernel(float *conv_corr_kernel, size_t image_size, size_t grid_size, float conv_corr_norm_factor) {
    const int support = 8;
    size_t t1_t2 = blockIdx.x*blockDim.x + threadIdx.x;
    if(t1_t2 < image_size / 2 + 1){
        float t1_t2_norm = (float)t1_t2 / grid_size;
        float correction = 0.0;
        for(int i=0; i < sizeof(quadrature_nodes)/sizeof(*quadrature_nodes); i++){
            float angle = t1_t2_norm * support * quadrature_nodes[i];
            correction += quadrature_kernel[i] * quadrature_weights[i] * cospif(angle);
        }
        conv_corr_kernel[t1_t2] = correction * support / conv_corr_norm_factor;
    }
}

__global__ void fused_gridding(cufftComplex* r_grid,
                               const float*  B_in,
                               const float*  Vis_real,
                               const float*  Vis_imag,
                               const float   Vin[3][3],
                               const float   r1r2_scale,
                               const size_t  grid_size,
                               const size_t  num_baselines,
                               const float   r3_centre,
                               const float   inv_r3_half_range,
                               const int     num_terms){
    const int   KERNEL_SUPPORT_BOUND = 16;
    const int   support              = 8;
    const float beta                 = 15.3704324328;
    const int   half_support         = support / 2;
    const float inv_half_support     = 1.0f / half_support;
    const long  grid_min_r1r2        = -(long)grid_size      / 2;
    const long  grid_max_r1r2        = ((long)grid_size - 1) / 2;
    const long  origin_offset_r1r2   =  (long)grid_size      / 2;
    const float weight               = fabsf(Vin[0][0]*Vin[1][1] - Vin[0][1]*Vin[1][0]);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_baselines){
        float pos_r1      = B_in[idx*3+0] * r1r2_scale;
        float pos_r2      = B_in[idx*3+1] * r1r2_scale;
        float s           = (B_in[idx*3+2] - r3_centre) * inv_r3_half_range;
        s = fminf(1.0f, fmaxf(-1.0f, s));
        float cheb_values[CHEB_MAX_MOMENTS];
        chebyshev_sequence(s, num_terms, cheb_values);
        long  grid_r1_min = max((long)ceilf (pos_r1 - half_support), grid_min_r1r2);
        long  grid_r1_max = min((long)floorf(pos_r1 + half_support), grid_max_r1r2);
        long  grid_r2_min = max((long)ceilf (pos_r2 - half_support), grid_min_r1r2);
        long  grid_r2_max = min((long)floorf(pos_r2 + half_support), grid_max_r1r2);
        if (grid_r1_min > grid_r1_max || grid_r2_min > grid_r2_max) {
            return;
        }
        float kernel_r1[KERNEL_SUPPORT_BOUND],
              kernel_r2[KERNEL_SUPPORT_BOUND];
        for(long grid_r1 = grid_r1_min; grid_r1 <= grid_r1_max; grid_r1++){
            kernel_r1[grid_r1 - grid_r1_min] = exp_semicircle(beta,(grid_r1 - pos_r1) * inv_half_support);
        }
        for(long grid_r2 = grid_r2_min; grid_r2 <= grid_r2_max; grid_r2++){
            kernel_r2[grid_r2 - grid_r2_min] = exp_semicircle(beta,(grid_r2 - pos_r2) * inv_half_support);
        }
        for(long grid_r1 = grid_r1_min; grid_r1 <= grid_r1_max; grid_r1++){
            for(long grid_r2 = grid_r2_min; grid_r2 <= grid_r2_max; grid_r2++){
                float kernel_value = kernel_r1[grid_r1 - grid_r1_min] * kernel_r2[grid_r2 - grid_r2_min];
                if(((grid_r1 + grid_r2) & 1) != 0){
                    kernel_value = -kernel_value;
                }
                const long grid_offset_r1r2r3 = (grid_r1 + origin_offset_r1r2) * (long)grid_size + grid_r2 + origin_offset_r1r2;
                for (int term = 0; term < num_terms; ++term) {
                    const size_t slice_offset = static_cast<size_t>(term) * grid_size * grid_size;
                    const float moment_weight = (cheb_values[term] / weight) * kernel_value;
                    atomicAdd(&r_grid[slice_offset + grid_offset_r1r2r3].x, Vis_real[idx] * moment_weight);
                    atomicAdd(&r_grid[slice_offset + grid_offset_r1r2r3].y, Vis_imag[idx] * moment_weight);
                }
            }
        }
    }
}

__global__ void fused_interpolation(float*       dirty,
                                    const cufftComplex* r_grid_stack,
                                    const float  Vin[3][3],
                                    const float  dc_rad,
                                    const size_t di,
                                    const size_t gi,
                                    const float* conv_corr_kernel,
                                    const float  conv_corr_norm_factor,
                                    const float  inv_num_baselines,
                                    const float  r3_centre,
                                    const float  r3_half_range,
                                    const int    num_terms){
    const size_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy  = blockIdx.y * blockDim.y + threadIdx.y;
    const long   half_image_size = di/2;
    const float  di2  = di*0.5f;
    const float  idxf =       idx - di2;
    const float  idyf =       idy - di2;
    const long   idxr = (long)idx - half_image_size;
    const long   idyr = (long)idy - half_image_size;
    const float  r0   = 180.0f / M_PI;
    const float  dc   = dc_rad / M_PI * 180;
    const float  V00  = Vin[0][0];
    const float  V01  = Vin[0][1];
    const float  V10  = Vin[1][0];
    const float  V11  = Vin[1][1];
    const float  V20  = Vin[2][0];
    const float  V21  = Vin[2][1];
    const float  V22  = Vin[2][2];
    const float  xi   = V20/V22;
    const float  eta  = V21/V22;


    float        oi0, oi1;
    float        pixel_sum;

    dirty   +=   half_image_size*di + half_image_size;

    if(idx<di && idy<di){

        const float p1 = (-V00*idxf + V10*idyf) / fabsf(V22) + di2;
        const float p2 = (-V01*idxf + V11*idyf) / fabsf(V22) + di2;

        float x   = -dc * (p1 - (di2 + 1.0f));
        float y   =  dc * (p2 - (di2 + 1.0f));
        float h   = hypotf(x, y);
        float hr0 = h/r0;

        float r, w, z;
        if(h != 0.0f){
            x /=  h;
            y /= -h;
        }else{
            x  = 0.0f;
            y  = 1.0f;
        }

        if(h <= r0){

            r = h;
            if(h < r0*sqrtf(0.5f)){
                z =            1.0f - sqrtf(1.0f - hr0*hr0);
            }else{
                z = hr0*hr0 / (1.0f + sqrtf(1.0f - hr0*hr0));
            }

            w = xi*xi + eta*eta;
            if(w == 0.0f){
                x =  r*x;
                y = -r*y;
            }else{
                x =  r*x + z*r0*xi;
                y = -r*y + z*r0*eta;
            }

            oi0 = -x/dc + di2 + 1.0f;
            oi1 =  y/dc + di2 + 1.0f;
        }else{

            oi0 = p1;
            oi1 = p2;
        }

        const float t1 = dc_rad * static_cast<float>(idyr);
        const float t2 = dc_rad * static_cast<float>(idxr);
        const float v13 = Vin[0][2];
        const float v23 = Vin[1][2];
        const float v33 = Vin[2][2];
        const float root_arg = 1.0f - (t1 + v13) * (t1 + v13) - (t2 + v23) * (t2 + v23);
        if (root_arg < 0.0f) {
            return;
        }
        const float branch_sign = (v33 >= 0.0f) ? 1.0f : -1.0f;
        const float t3 = -v33 + branch_sign * sqrtf(root_arg);
        const float alpha = 2.0f * M_PI * t3 * r3_half_range;
        const float beta = 2.0f * M_PI * t3 * r3_centre;
        const float cb = cosf(beta);
        const float sb = sinf(beta);
        float jvals[CHEB_MAX_MOMENTS];
        bessel_sequence(alpha, num_terms - 1, jvals);

        pixel_sum = 0.0f;
        const long grid_index = gi*gi/2 + gi/2 + idyr*(long)gi + idxr;
        for (int term = 0; term < num_terms; ++term) {
            const size_t slice_offset = static_cast<size_t>(term) * gi * gi;
            const cufftComplex moment = r_grid_stack[slice_offset + grid_index];
            float coeff_r = 0.0f;
            float coeff_i = 0.0f;
            if (term == 0) {
                coeff_r = cb * jvals[term];
                coeff_i = sb * jvals[term];
            } else {
                const float amplitude = 2.0f * jvals[term];
                switch (term & 3) {
                    case 0:
                        coeff_r =  amplitude * cb;
                        coeff_i =  amplitude * sb;
                        break;
                    case 1:
                        coeff_r = -amplitude * sb;
                        coeff_i =  amplitude * cb;
                        break;
                    case 2:
                        coeff_r = -amplitude * cb;
                        coeff_i = -amplitude * sb;
                        break;
                    default:
                        coeff_r =  amplitude * sb;
                        coeff_i = -amplitude * cb;
                        break;
                }
            }
            pixel_sum += real_complex_product(coeff_r, coeff_i, moment);
        }
        if(idxr+idyr & 1){
            pixel_sum = - pixel_sum;
        }

        pixel_sum *= 1 / (conv_corr_kernel[abs(idxr)] *
                          conv_corr_kernel[abs(idyr)] *
                          conv_corr_norm_factor       *
                          conv_corr_norm_factor);
        pixel_sum  = fabs(pixel_sum);

        const float LL    = oi0 - half_image_size;
        const float MM    = oi1 - half_image_size;
        const float value = pixel_sum * inv_num_baselines;

        if(fabs(LL) < half_image_size-1 && fabs(MM)<half_image_size-1){
            const float LLf = floorf(LL);
            const float MMf = floorf(MM);
            const float LLc = ceilf (LL);
            const float MMc = ceilf (MM);

            atomicAdd(&dirty[(long)MMf * (long)di + (long)LLf],  (1-LL+LLf) * (1-MM+MMf) * value);
            atomicAdd(&dirty[(long)MMc * (long)di + (long)LLf],  (1-LL+LLf) * (0+MM-MMf) * value);
            atomicAdd(&dirty[(long)MMf * (long)di + (long)LLc],  (0+LL-LLf) * (1-MM+MMf) * value);
            atomicAdd(&dirty[(long)MMc * (long)di + (long)LLc],  (0+LL-LLf) * (0+MM-MMf) * value);
        }
    }
}

size_t ceiling_divide(size_t a, size_t b) {
    size_t q =  a/b;
    return q + (a > q*b);
}

int FIpipeDevice(const DevicePreprocessBuffers& preprocess_buffers, float* dirty_image, size_t image_size, float cell_size){
	float* dirty;
	float* conv_corr_kernel;
	cudaError_t cudaStatus;
	cufftComplex* r_grid_stack;
	
	cudaEvent_t start, stop, eventstream;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&eventstream);
	
	const size_t num_baselines = preprocess_buffers.num_samples;
	size_t grid_size = ceiling_divide(image_size*3, 2); // * 1.5, rounding up
	float r1r2_scale = cell_size*grid_size;
	float conv_corr_norm_factor = 2.4937047051153827;
	const int stats_threads = 256;
    const size_t r3_stats_blocks = ceiling_divide(num_baselines, stats_threads);
    const size_t pixel_count = image_size * image_size;
    const size_t t3_stats_blocks = ceiling_divide(pixel_count, stats_threads);
    float* d_r3_block_min = nullptr;
    float* d_r3_block_max = nullptr;
    float* d_t3_block_max = nullptr;
    unsigned int* d_t3_block_valid = nullptr;
    float* d_cheb_stats = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_r3_block_min, r3_stats_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_r3_block_max, r3_stats_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_t3_block_max, t3_stats_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_t3_block_valid, t3_stats_blocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc((void**)&d_cheb_stats, 4 * sizeof(float)));

    r3_range_blocks_kernel<<<r3_stats_blocks, stats_threads>>>(
        preprocess_buffers.d_bin,
        d_r3_block_min,
        d_r3_block_max,
        num_baselines
    );
    CHECK_CUDA(cudaGetLastError());
    t3_stats_blocks_kernel<<<t3_stats_blocks, stats_threads>>>(
        (const float (*)[3])preprocess_buffers.d_vin,
        d_t3_block_max,
        d_t3_block_valid,
        image_size,
        cell_size
    );
    CHECK_CUDA(cudaGetLastError());

    finalize_cheb_stats_kernel<<<1, 1>>>(
        d_r3_block_min,
        d_r3_block_max,
        d_t3_block_max,
        d_t3_block_valid,
        d_cheb_stats,
        r3_stats_blocks,
        t3_stats_blocks
    );
    CHECK_CUDA(cudaGetLastError());

    float cheb_stats[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    CHECK_CUDA(cudaMemcpy(
        cheb_stats,
        d_cheb_stats,
        sizeof(cheb_stats),
        cudaMemcpyDeviceToHost
    ));
    cudaFree(d_r3_block_min);
    cudaFree(d_r3_block_max);
    cudaFree(d_t3_block_max);
    cudaFree(d_t3_block_valid);
    cudaFree(d_cheb_stats);

    const HostChebyshevSelection cheb_selection = finish_host_chebyshev_selection(
        cheb_stats[0],
        cheb_stats[1],
        cheb_stats[2],
        static_cast<size_t>(cheb_stats[3])
    );

    const int bucket_terms[4] = {8, 16, 24, 32};
    const int selector_alpha_samples = 257;
    const int selector_s_samples = 1024;
    const int selector_total_samples = selector_alpha_samples * selector_s_samples;
    float* d_bucket_errors = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_bucket_errors, 4 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_bucket_errors, 0, 4 * sizeof(float)));
    const int selector_threads = 256;
    const int selector_blocks = static_cast<int>(ceiling_divide(selector_total_samples, selector_threads));
    cheb_bucket_error_kernel<<<selector_blocks, selector_threads>>>(
        cheb_selection.alpha_max,
        d_bucket_errors,
        selector_alpha_samples,
        selector_s_samples
    );
    CHECK_CUDA(cudaGetLastError());
    float bucket_errors[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    CHECK_CUDA(cudaMemcpy(bucket_errors, d_bucket_errors, sizeof(bucket_errors), cudaMemcpyDeviceToHost));
    cudaFree(d_bucket_errors);

    int active_cheb_terms = CHEB_MAX_MOMENTS;
    for (int i = 0; i < 4; ++i) {
        if (bucket_errors[i] <= CHEB_TARGET_ERROR) {
            active_cheb_terms = bucket_terms[i];
            break;
        }
    }
    const size_t moment_plane_size = grid_size * grid_size;
    const size_t total_moment_size = static_cast<size_t>(active_cheb_terms) * moment_plane_size;
    
	cudaMalloc((void**)&dirty, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&conv_corr_kernel, (image_size/2+1)*sizeof(float));
	cudaMalloc((void**)&r_grid_stack, total_moment_size * sizeof(cufftComplex));
	
	cudaMemset(dirty, 0, image_size * image_size * sizeof(float));
	cudaMemset(conv_corr_kernel, 0, (image_size/2+1) * sizeof(float));
	cudaMemset(r_grid_stack, 0, total_moment_size * sizeof(cufftComplex));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error sync 1 : %s\n", cudaGetErrorString(cudaStatus));
	}
	
	dim3 Ts = {32, 32}, Bs = {
        (unsigned)ceiling_divide(image_size, Ts.x),
        (unsigned)ceiling_divide(image_size, Ts.y),
    };
    dim3 Tk = {1024},   Bk = {
        (unsigned)ceiling_divide(image_size/2+1, Tk.x)
    };
    dim3 Tg = {1024},   Bg = {
        (unsigned)ceiling_divide(num_baselines,  Tg.x)
    };
	
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	
	cudaEventRecord(start);
	cufftHandle plan;
	int fft_dims[2] = {
        static_cast<int>(grid_size),
        static_cast<int>(grid_size)
    };
    cufftPlanMany(
        &plan,
        2,
        fft_dims,
        nullptr,
        1,
        static_cast<int>(moment_plane_size),
        nullptr,
        1,
        static_cast<int>(moment_plane_size),
        CUFFT_C2C,
        active_cheb_terms
    );
	cufftSetStream(plan, stream1);
	
	convolveKernel<<<Bk, Tk, 0, stream2>>>(conv_corr_kernel, image_size, grid_size, conv_corr_norm_factor);
	fused_gridding<<<Bg, Tg, 0, stream1>>>(r_grid_stack,
		                                   preprocess_buffers.d_bin,
		                                   preprocess_buffers.d_vis_real,
		                                   preprocess_buffers.d_vis_imag,
		                                   (const float (*)[3])preprocess_buffers.d_vin,
		                                   r1r2_scale,
		                                   grid_size,
		                                   num_baselines,
                                           cheb_selection.r3_centre,
                                           cheb_selection.inv_r3_half_range,
                                           active_cheb_terms);
	cufftExecC2C(plan, r_grid_stack, r_grid_stack, CUFFT_INVERSE);
	
	cudaEventRecord(eventstream,stream2);
	cudaStreamWaitEvent(stream1,eventstream,0);
	
    fused_interpolation<<<Bs, Ts, 0, stream1>>>(dirty,
		                                        r_grid_stack,
		                                        (const float (*)[3])preprocess_buffers.d_vin,
		                                        cell_size,
		                                        image_size,
		                                        grid_size,
		                                        conv_corr_kernel,
		                                        conv_corr_norm_factor,
		                                        1.0f/static_cast<float>(num_baselines),
                                                cheb_selection.r3_centre,
                                                cheb_selection.r3_half_range,
                                                active_cheb_terms);
	
	cudaStreamSynchronize(stream1);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cufftDestroy(plan);
	cudaEventDestroy(eventstream);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
    
	cudaMemcpy(dirty_image, dirty, image_size * image_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error sync 2 : %s\n", cudaGetErrorString(cudaStatus));
	}
	
	cudaFree(dirty);
	cudaFree(conv_corr_kernel);
	cudaFree(r_grid_stack);
	
	return 0;
}

namespace {

void print_integrated_usage(const char* program_name) {
    std::cerr
        << "Usage: " << program_name
        << " MeasurementSet.ms Image_Size Cell_Size Output_Name.fits\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        print_integrated_usage(argv[0]);
        return 2;
    }

    try {
        const std::string ms_path(argv[1]);
        const std::size_t image_size = static_cast<std::size_t>(std::stoul(argv[2]));
        const float cell_size = std::stof(argv[3]);
        const std::string output_name(argv[4]);

        if (image_size == 0) {
            throw std::runtime_error("Image_Size must be greater than zero");
        }
        if (cell_size <= 0.0f) {
            throw std::runtime_error("Cell_Size must be greater than zero");
        }

        const auto total_start = std::chrono::high_resolution_clock::now();

        std::cerr << "Reading Measurement Set: " << ms_path << '\n';
        HostMeasurementSetData host_data = read_measurement_set(ms_path);

        std::cerr << "Preprocessing " << host_data.num_samples
                  << " visibility sample(s) on GPU\n";
        DevicePreprocessBuffers device_buffers;
        try {
            preprocess_measurement_set_gpu(host_data, device_buffers);

            std::vector<float> dirty_image(image_size * image_size, 0.0f);
            std::cerr << "Running imager on GPU preprocess buffers\n";
            const int image_status = FIpipeDevice(
                device_buffers,
                dirty_image.data(),
                image_size,
                cell_size
            );
            if (image_status != 0) {
                throw std::runtime_error("FIpipeDevice failed");
            }

            write_fits_2d(
                output_name,
                dirty_image,
                static_cast<long>(image_size),
                static_cast<long>(image_size)
            );
            const auto total_stop = std::chrono::high_resolution_clock::now();
            const double total_seconds =
                std::chrono::duration<double>(total_stop - total_start).count();
            std::cerr << "Wrote image FITS: " << output_name << '\n';
            std::cout << "Total pipeline time: " << total_seconds * 1000.0 << " ms" << std::endl;
            free_device_preprocess_buffers(device_buffers);
        } catch (...) {
            free_device_preprocess_buffers(device_buffers);
            throw;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Integrated SVD imager failed: " << ex.what() << '\n';
        return 1;
    }
}
	
