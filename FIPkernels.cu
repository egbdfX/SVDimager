#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <future>
#include <vector>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#define M_PI 3.14159265358979323846
#define WCSTRIG_TOL 1e-10
#define CHEB_MAX_MOMENTS 32
#define CHEB_TARGET_ERROR 1.0e-2f

/* The gridding kernels are developed based on SKA SDP (https://gitlab.com/ska-telescope/sdp/ska-sdp-func). */

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

long int computeCeil(float num) {
	if (num<0) {
		return -floorf(-num);
	} else {
		return ceilf(num);
	}
}

long int computeFloor(float num) {
	if (num<0) {
		return -ceilf(-num);
	} else {
		return floorf(num);
	}
}

__device__ long int ceil_device(float num) {
	if (num<0) {
		return -floorf(-num);
	} else {
		return ceilf(num);
	}
}

__device__ long int floor_device(float num) {
	if (num<0) {
		return -ceilf(-num);
	} else {
		return floorf(num);
	}
}

__device__ float fmod_device(float x, float y) {
	return fmod(x, y);
}

__device__ float exp_semicircle(const float beta, float x){
	const float xx = x*x;
    
	return ((xx > float(1.0)) ? float(0.0) : exp(beta*(sqrt(float(1.0) - xx) - float(1.0))));
}

__device__ void __sincosd(float angle, float &s, float &c) {
	// angle in degrees
	if (fmod(angle, 90.0f) == 0) {
		int i = static_cast<int>(fabsf(floor_device(angle / 90.0f + 0.5f))) % 4;
		switch (i) {
			case 0:
				s = 0.0f;
				c = 1.0f;
				break;
			case 1:
				s = (angle > 0.0f) ? 1.0f : -1.0f;
				c = 0.0f;
				break;
			case 2:
				s = 0.0f;
				c = -1.0f;
				break;
			case 3:
				s = (angle > 0.0f) ? -1.0f : 1.0f;
				c = 0.0f;
				break;
		}
	} else {
		s = sinf(angle * M_PI / 180.0f);
		c = cosf(angle * M_PI / 180.0f);
	}
}

__device__ float __sind(float angle) {
	// angle in degrees
	if (fmod(angle, 90.0f) == 0) {
		int i = static_cast<int>(fabsf(floor_device(angle / 90.0f - 0.5f))) % 4;
		switch (i) {
			case 0:
				return 1.0f;
			case 1:
				return 0.0f;
			case 2:
				return -1.0f;
			case 3:
				return 0.0f;
		}
	} else {
		return sinf(angle * M_PI / 180.0f);
	}
}

__device__ float __cosd(float angle) {
	// angle in degrees
	if (fmod(angle, 90.0f) == 0) {
		int i = static_cast<int>(fabsf(floor_device(angle / 90.0f + 0.5f))) % 4;
		switch (i) {
			case 0:
				return 1.0f;
			case 1:
				return 0.0f;
			case 2:
				return -1.0f;
			case 3:
				return 0.0f;
		}
	} else {
		return cosf(angle * M_PI / 180.0f);
	}
}

__device__ float __atan2d(float y, float x) {
	if (y == 0.0f) {
		return (x >= 0.0f) ? 0.0f : 180.0f;
	} else if (x == 0.0f) {
		return (y > 0.0f) ? 90.0f : -90.0f;
	} else {
		return atan2f(y, x) * 180.0f / M_PI;
	}
}

__device__ float __acosd(float v) {
	if (v >= 1.0f && v - 1.0f < WCSTRIG_TOL) {
		return 0.0f;
	} else if (v == 0.0f) {
		return 90.0f;
	} else if (v <= -1.0f && v + 1.0f > -WCSTRIG_TOL) {
		return 180.0f;
	} else {
		return acosf(v) * 180.0f / M_PI;
	}
}

__device__ float __asind(float v) {
	if (v <= -1.0f && v + 1.0f > -WCSTRIG_TOL) {
		return -90.0f;
	} else if (v == 0.0f) {
		return 0.0f;
	} else if (v >= 1.0f && v - 1.0f < WCSTRIG_TOL) {
		return 90.0f;
	} else {
		return asinf(v) * 180.0f / M_PI;
	}
}

__device__ void chebyshev_sequence(float s, int num_terms, float* values) {
	if (num_terms <= 0) {
		return;
	}
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
	if (max_order < 0) {
		return;
	}

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

double host_chebyshev_t(int n, double s) {
	if (n == 0) {
		return 1.0;
	}
	if (n == 1) {
		return s;
	}

	double tnm2 = 1.0;
	double tnm1 = s;
	for (int k = 2; k <= n; ++k) {
		const double tn = 2.0 * s * tnm1 - tnm2;
		tnm2 = tnm1;
		tnm1 = tn;
	}
	return tnm1;
}

void host_bessel_sequence(double alpha, int max_order, std::vector<double>& jvals) {
	if (max_order < 0) {
		jvals.clear();
		return;
	}

	jvals.assign(max_order + 1, 0.0);
	for (int n = 0; n <= max_order; ++n) {
		jvals[n] = std::cyl_bessel_j(static_cast<double>(n), alpha);
	}
}

std::complex<double> host_chebyshev_phase_approx(double alpha, double s, int kept_terms) {
	std::vector<double> jvals;
	host_bessel_sequence(alpha, kept_terms - 1, jvals);

	std::complex<double> sum(0.0, 0.0);
	for (int n = 0; n < kept_terms; ++n) {
		const double tn = host_chebyshev_t(n, s);
		if (n == 0) {
			sum += std::complex<double>(jvals[n] * tn, 0.0);
			continue;
		}

		std::complex<double> phase(1.0, 0.0);
		switch (n & 3) {
			case 0:
				phase = std::complex<double>(1.0, 0.0);
				break;
			case 1:
				phase = std::complex<double>(0.0, 1.0);
				break;
			case 2:
				phase = std::complex<double>(-1.0, 0.0);
				break;
			default:
				phase = std::complex<double>(0.0, -1.0);
				break;
		}
		sum += 2.0 * phase * jvals[n] * tn;
	}
	return sum;
}

float host_chebyshev_error_sample(float alpha, int kept_terms) {
	const int sample_count = 2048;
	double max_err = 0.0;

	for (int i = 0; i < sample_count; ++i) {
		const double s = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(sample_count - 1);
		const std::complex<double> exact = std::exp(std::complex<double>(0.0, static_cast<double>(alpha) * s));
		const std::complex<double> approx = host_chebyshev_phase_approx(static_cast<double>(alpha), s, kept_terms);

		max_err = std::max(max_err, std::abs(exact - approx));
	}

	return static_cast<float>(max_err);
}

float host_chebyshev_error_sample_range(float alpha_max, int kept_terms, int alpha_samples = 513) {
	float max_err = 0.0f;
	for (int i = 0; i < alpha_samples; ++i) {
		const float alpha = (alpha_samples == 1)
			? alpha_max
			: alpha_max * static_cast<float>(i) / static_cast<float>(alpha_samples - 1);
		max_err = std::max(max_err, host_chebyshev_error_sample(alpha, kept_terms));
	}
	return max_err;
}

int host_find_min_chebyshev_terms(float alpha_max, float target_error, int max_kept_terms = 32) {
	for (int kept_terms = 1; kept_terms <= max_kept_terms; ++kept_terms) {
		if (host_chebyshev_error_sample_range(alpha_max, kept_terms) <= target_error) {
			return kept_terms;
		}
	}
	return -1;
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
	int selected_cheb_terms = CHEB_MAX_MOMENTS;
	float selected_cheb_sample = 0.0f;
};

HostChebyshevSelection compute_host_chebyshev_selection(
		const float* Bin,
		const float* Vin,
		size_t num_baselines,
		size_t image_size,
		float cell_size
) {
	HostChebyshevSelection result;
	result.r3_min = Bin[2];
	result.r3_max = Bin[2];
	for (size_t i = 1; i < num_baselines; ++i) {
		const float r3 = Bin[i * 3 + 2];
		result.r3_min = fminf(result.r3_min, r3);
		result.r3_max = fmaxf(result.r3_max, r3);
	}

	result.r3_centre = 0.5f * (result.r3_max + result.r3_min);
	result.r3_half_range = fmaxf(0.5f * (result.r3_max - result.r3_min), 1.0e-6f);
	result.inv_r3_half_range = 1.0f / result.r3_half_range;

	const float v13 = Vin[0 * 3 + 2];
	const float v23 = Vin[1 * 3 + 2];
	const float v33 = Vin[2 * 3 + 2];
	const float branch_sign = (v33 >= 0.0f) ? 1.0f : -1.0f;
	const float half_image_size = static_cast<float>(image_size) / 2.0f;
	for (size_t y = 0; y < image_size; ++y) {
		const float t1 = cell_size * (static_cast<float>(y) - half_image_size);
		for (size_t x = 0; x < image_size; ++x) {
			const float t2 = cell_size * (static_cast<float>(x) - half_image_size);
			const float root_arg = 1.0f - (t1 + v13) * (t1 + v13) - (t2 + v23) * (t2 + v23);
			if (root_arg < 0.0f) {
				continue;
			}
			const float t3 = -v33 + branch_sign * sqrtf(root_arg);
			result.max_abs_t3 = fmaxf(result.max_abs_t3, fabsf(t3));
			++result.valid_pixels;
		}
	}

	result.alpha_max = 2.0f * M_PI * result.max_abs_t3 * result.r3_half_range;
	const int selected = host_find_min_chebyshev_terms(
			result.alpha_max, CHEB_TARGET_ERROR, CHEB_MAX_MOMENTS);
	result.selected_cheb_terms = (selected > 0) ? selected : CHEB_MAX_MOMENTS;
	result.selected_cheb_sample = host_chebyshev_error_sample_range(
			result.alpha_max, result.selected_cheb_terms);
	return result;
}

__global__ void convolveKernel(float *conv_corr_kernel, size_t image_size, size_t grid_size, float conv_corr_norm_factor) {
	const int support = 8;
	size_t t1_t2 = blockIdx.x * blockDim.x + threadIdx.x;
	if (t1_t2 < image_size / 2 + 1) {
		float t1_t2_norm = static_cast<float>(t1_t2) / grid_size;
		float correction = 0.0;
		float angle;
		for (int i = 0; i < 14; ++i) {
			angle = M_PI * t1_t2_norm * support * quadrature_nodes[i];
			correction += quadrature_kernel[i] * quadrature_weights[i] * cosf(angle);
		}
		conv_corr_kernel[t1_t2] = correction * support / conv_corr_norm_factor;
	}
}

__global__ void computeVisWeighted(float *Vis_real, float *Vis_imag, size_t num_baselines, float inten_scale) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_baselines) {
		Vis_real[idx] = Vis_real[idx]/inten_scale;
		Vis_imag[idx] = Vis_imag[idx]/inten_scale;
	}
}

__global__ void gridding(float* B_in,
		float* moment_grids_real, float* moment_grids_imag,
		float* Vis_real, float* Vis_imag,
		float r1r2_scale, size_t grid_size, size_t num_baselines,
		float r3_centre, float inv_r3_half_range, int num_terms) {
	
    const int support = 8;
	int half_support = support / 2;
	float inv_half_support = 1 / static_cast<float>(half_support);
	long int grid_min_r1r2 = -static_cast<long int>(grid_size) / 2;
	long int grid_max_r1r2 = (static_cast<long int>(grid_size) - 1) / 2;
	long int origin_offset_r1r2 = static_cast<long int>(grid_size) / 2;
	const int KERNEL_SUPPORT_BOUND = 16;
	const float beta = 15.3704324328;
	float kernel_value;
	
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < num_baselines) {
		float pos_r1 = B_in[idx*3+0] * r1r2_scale;
		float pos_r2 = B_in[idx*3+1] * r1r2_scale;
        float r3 = B_in[idx*3+2];
		float s = (r3 - r3_centre) * inv_r3_half_range;
		s = fminf(1.0f, fmaxf(-1.0f, s));
		float cheb_values[CHEB_MAX_MOMENTS];
		chebyshev_sequence(s, num_terms, cheb_values);
		long int grid_r1_min = max(ceil_device(pos_r1 - half_support), grid_min_r1r2);
		long int grid_r1_max = min(floor_device(pos_r1 + half_support), grid_max_r1r2);
		long int grid_r2_min = max(ceil_device(pos_r2 - half_support), grid_min_r1r2);
		long int grid_r2_max = min(floor_device(pos_r2 + half_support), grid_max_r1r2);
		if (grid_r1_min > grid_r1_max || grid_r2_min > grid_r2_max) {
			return;
		}
		float kernel_r1[KERNEL_SUPPORT_BOUND], kernel_r2[KERNEL_SUPPORT_BOUND];
		for (long int grid_r1 = grid_r1_min; grid_r1 <= grid_r1_max; grid_r1++)
		{
			kernel_r1[grid_r1 - grid_r1_min] = exp_semicircle(beta,(static_cast<float>(grid_r1) - pos_r1) * inv_half_support);
		}
		for (long int grid_r2 = grid_r2_min; grid_r2 <= grid_r2_max; grid_r2++)
		{
			kernel_r2[grid_r2 - grid_r2_min] = exp_semicircle(beta,(static_cast<float>(grid_r2) - pos_r2) * inv_half_support);
		}
		
		for (long int grid_r1 = grid_r1_min; grid_r1 <= grid_r1_max; grid_r1++)
		{
			for (long int grid_r2 = grid_r2_min; grid_r2 <= grid_r2_max; grid_r2++)
			{
				kernel_value = kernel_r1[grid_r1 - grid_r1_min] * kernel_r2[grid_r2 - grid_r2_min];
				if (((grid_r1 + grid_r2) & 1) != 0) {
					kernel_value = -kernel_value;
				}
				const long int grid_offset_r1r2r3 = (grid_r1 + origin_offset_r1r2) * static_cast<long int>(grid_size) + (grid_r2 + origin_offset_r1r2);
                for (int term = 0; term < num_terms; ++term) {
					const size_t slice_offset = static_cast<size_t>(term) * grid_size * grid_size;
					const float weight = cheb_values[term] * kernel_value;
					atomicAdd(&moment_grids_real[slice_offset + grid_offset_r1r2r3], Vis_real[idx] * weight);
					atomicAdd(&moment_grids_imag[slice_offset + grid_offset_r1r2r3], Vis_imag[idx] * weight);
				}
			}
		}
	}
}

__global__ void combineToComplex(float* data_real, float* data_imag, cufftComplex* complex_data, size_t grid_size) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = grid_size * grid_size;
	if (idx < size) {
		complex_data[idx].x = data_real[idx];
		complex_data[idx].y = data_imag[idx];
	}
}

__global__ void accumulation(float* dirty_pre,
		const cufftComplex* moment0_shifted,
		const cufftComplex* moment1_shifted,
		const cufftComplex* moment2_shifted,
		const float* V_in,
		size_t image_size, size_t grid_size, float cell_size) {
	size_t half_image_size = image_size / 2;
	size_t grid_index_offset_image_centre = grid_size*grid_size/2 + grid_size/2;
	size_t image_index_offset_image_centre = half_image_size*image_size + half_image_size;
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
	if (idx < image_size && idy < image_size) { 
		idx = idx - half_image_size;
		idy = idy - half_image_size;
		float t1 = cell_size * static_cast<float>(idy);
		float t2 = cell_size * static_cast<float>(idx);

        const float v13 = V_in[0 * 3 + 2];
		const float v23 = V_in[1 * 3 + 2];
		const float v33 = V_in[2 * 3 + 2];
        const float root_arg = 1.0f - (t1 + v13) * (t1 + v13) - (t2 + v23) * (t2 + v23);
        if (root_arg < 0.0f) {
			return;
		}
        const float branch_sign = (v33 >= 0.0f) ? 1.0f : -1.0f;
        const float t3 = -v33 + branch_sign * sqrtf(root_arg);
        
        float phase = 2.0f * M_PI * t3;
		float phase_sq = phase * phase;
		cufftComplex m0 = moment0_shifted[grid_index_offset_image_centre + idy * grid_size + idx];
		cufftComplex m1 = moment1_shifted[grid_index_offset_image_centre + idy * grid_size + idx];
		cufftComplex m2 = moment2_shifted[grid_index_offset_image_centre + idy * grid_size + idx];
        
        float pixel_sum = m0.x - phase * m1.y - 0.5f * phase_sq * m2.x;
		if (((abs(idx)+abs(idy)) & 1) != 0) {
			pixel_sum = - pixel_sum;
		}
		dirty_pre[image_index_offset_image_centre + idy*image_size + idx] = pixel_sum;
	}
}

__global__ void scaling(float* dirty_pre, float* conv_corr_kernel, size_t image_size, float conv_corr_norm_factor) {
	size_t half_image_size = image_size / 2;
	size_t image_index_offset_image_centre = half_image_size*image_size + half_image_size;
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (idx < image_size && idy < image_size) { 
		idx = idx - half_image_size;
		idy = idy - half_image_size;
        
		dirty_pre[image_index_offset_image_centre + idy * image_size + idx] *= 1/(conv_corr_kernel[abs(idx)]*conv_corr_kernel[abs(idy)]*conv_corr_norm_factor*conv_corr_norm_factor);
		dirty_pre[image_index_offset_image_centre + idy * image_size + idx] = fabs(dirty_pre[image_index_offset_image_centre + idy * image_size + idx]);
	}
}

__global__ void coordschange(float* output_index, float* V_in, size_t image_size) {
	size_t half_image_size = image_size / 2;
	
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    
	if (idx < image_size && idy < image_size) {
		output_index[(idx*image_size+idy)*2+0] = (-V_in[0*3+0]*(static_cast<float>(idx) - static_cast<float>(half_image_size))+V_in[1*3+0]*(static_cast<float>(idy) - static_cast<float>(half_image_size)))/fabs(V_in[2*3+2]) + static_cast<float>(half_image_size);
		output_index[(idx*image_size+idy)*2+1] = (-V_in[0*3+1]*(static_cast<float>(idx) - static_cast<float>(half_image_size))+V_in[1*3+1]*(static_cast<float>(idy) - static_cast<float>(half_image_size)))/fabs(V_in[2*3+2]) + static_cast<float>(half_image_size);	
	}
}

__global__ void p2p(float* output_index, float* V_in, float dc, size_t di) {
	/* According to paper: M. R.  Calabretta, E. W.  Greisen, 'Representations of celestial coordinates in FITS,' A&A,395(3),1077-1122,2002.*/
	
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
	float xi = V_in[6]/V_in[8];
	float eta = V_in[7]/V_in[8];
	dc = dc / M_PI * 180;
    
	if (idx < di && idy < di) {
		
		float p1 = output_index[(idx*di+idy)*2+0]; // p2
		float p2 = output_index[(idx*di+idy)*2+1]; // p1
		
		float x = -static_cast<float>(dc) * (p1 - (static_cast<float>(di) / 2.0f + 1.0f));
		float y = static_cast<float>(dc) * (p2 - (static_cast<float>(di) / 2.0f + 1.0f));

		float r0 = 180.0f / M_PI;
		float x0 = x / r0;
		float y0 = y / r0;
		float r2 = x0 * x0 + y0 * y0;
		
		float phi;
		if (r2 != 0.0f) {
			phi = __atan2d(x0, -y0);
		} else {
			phi = 0.0f;
		}

		float theta;
		if (r2 < 0.5f) {
			theta = __acosd(sqrtf(r2));
		} else if (r2 <= 1.0f) {
			theta = __asind(sqrtf(1.0f - r2));
		} else {
			return;
		}

		float sinphi, cosphi;
		__sincosd(phi, sinphi, cosphi);
		x = sinphi;
		y = cosphi;

		float t = (90.0f - fabsf(theta)) / 180.0f * M_PI;
		float z, costhe;
		if (t < 1.0e-5f) {
			if (theta > 0.0f) {
				z = t * t / 2.0f;
			} else {
				z = 2.0f - t * t / 2.0f;
			}
			costhe = t;
		} else {
			z = 1.0f - __sind(theta);
			costhe = __cosd(theta);
		}

		r0 = 180.0f / M_PI;
		float r = r0 * costhe;
		float w = xi * xi + eta * eta;

		if (w == 0.0f) {
			x = r * x;
			y = -r * y;
		} else {
			z = z * r0;
			float z1 = xi * z;
			float z2 = eta * z;
			x = r * x + z1;
			y = -r * y + z2;
		}

		output_index[(idx*di+idy)*2+0] = -1.0f / static_cast<float>(dc) * x + static_cast<float>(di) / 2.0f + 1.0f;
		output_index[(idx*di+idy)*2+1] = 1.0f / static_cast<float>(dc) * y + static_cast<float>(di) / 2.0f + 1.0f;
	}
}

__global__ void finalinterp(float* output_index, float* dirty_pre, float* dirty, size_t image_size, size_t num_baselines) {
	long int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long int idy = blockIdx.y * blockDim.y + threadIdx.y;
	size_t half_image_size = image_size / 2;
	size_t image_index_offset_image_centre = static_cast<long int>(half_image_size*image_size + half_image_size);
    
	if (idx < image_size && idy < image_size) {
		float LL = output_index[(static_cast<size_t>(idx)*image_size+static_cast<size_t>(idy))*2+0] - static_cast<float>(half_image_size);
		float MM = output_index[(static_cast<size_t>(idx)*image_size+static_cast<size_t>(idy))*2+1] - static_cast<float>(half_image_size);
		
		idx = idx - static_cast<long int>(half_image_size);
		idy = idy - static_cast<long int>(half_image_size);
		
		const float inv_num_baselines = 1.0f / static_cast<float>(num_baselines);
        const float value = dirty_pre[image_index_offset_image_centre+idy*static_cast<long int>(image_size)+idx] * inv_num_baselines;
		
		if (fabs(LL) < half_image_size-1 && fabs(MM)<half_image_size-1) {
			atomicAdd(
				&dirty[image_index_offset_image_centre+floor_device(MM)*static_cast<long int>(image_size)+floor_device(LL)],
				(1-LL+floor_device(LL))*(1-MM+floor_device(MM))*value
			);
			atomicAdd(
				&dirty[image_index_offset_image_centre+ceil_device(MM)*static_cast<long int>(image_size)+floor_device(LL)],
				(1-LL+floor_device(LL))*(MM-floor_device(MM))*value
			);
			atomicAdd(
				&dirty[image_index_offset_image_centre+floor_device(MM)*static_cast<long int>(image_size)+ceil_device(LL)],
				(LL-floor_device(LL))*(1-MM+floor_device(MM))*value
			);
			atomicAdd(
				&dirty[image_index_offset_image_centre+ceil_device(MM)*static_cast<long int>(image_size)+ceil_device(LL)],
				(LL-floor_device(LL))*(MM-floor_device(MM))*value
			);
		}
	}
}

int FIpipe(float* Visreal, float* Visimag, float* Bin, float* Vin, float* dirty_image, size_t num_baselines, size_t image_size, float cell_size){
	float* Vis_real;
	float* Vis_imag;
	float* B_in;
	float* V_in;
	float* dirty;
	float* dirty_pre;
	float* conv_corr_kernel;
	float* rr0_grid_real;
	float* rr0_grid_imag;
	float* rr1_grid_real;
	float* rr1_grid_imag;
	float* rr2_grid_real;
	float* rr2_grid_imag;
	cudaError_t cudaStatus;
	cufftComplex* rr0_grid_stack;
	cufftComplex* rr1_grid_stack;
	cufftComplex* rr2_grid_stack;
	float* output_index;
	cudaError_t cudaError;
	
	cudaEvent_t start, stop, eventstream;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&eventstream);
	
	size_t grid_size = computeCeil(1.5*static_cast<float>(image_size));
	float uv_scale = cell_size*grid_size;
	float conv_corr_norm_factor = 2.4937047051153827;
	
	cudaMalloc((void**)&Vis_real, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&Vis_imag, num_baselines * 1 * sizeof(float));
	cudaMalloc((void**)&B_in, num_baselines * 3 * sizeof(float));
	cudaMalloc((void**)&V_in, 3 * 3 * sizeof(float));
	cudaMalloc((void**)&dirty, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&dirty_pre, image_size * image_size * sizeof(float));
	cudaMalloc((void**)&conv_corr_kernel, (image_size/2+1)*sizeof(float));
	cudaMalloc((void**)&rr0_grid_real, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&rr0_grid_imag, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&rr1_grid_real, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&rr1_grid_imag, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&rr2_grid_real, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&rr2_grid_imag, grid_size * grid_size * sizeof(float));
	cudaMalloc((void**)&rr0_grid_stack, grid_size * grid_size * sizeof(cufftComplex));
	cudaMalloc((void**)&rr1_grid_stack, grid_size * grid_size * sizeof(cufftComplex));
	cudaMalloc((void**)&rr2_grid_stack, grid_size * grid_size * sizeof(cufftComplex));
	cudaMalloc((void**)&output_index, image_size * image_size * 2 * sizeof(float));
	
	cudaMemcpy(Vis_real, Visreal, num_baselines * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Vis_imag, Visimag, num_baselines * 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_in, Bin, num_baselines * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(V_in, Vin, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice); // cross term included
	cudaMemset(dirty, 0, image_size * image_size * sizeof(float));
	cudaMemset(dirty_pre, 0, image_size * image_size * sizeof(float));
	cudaMemset(conv_corr_kernel, 0, (image_size/2+1) * sizeof(float));
	cudaMemset(rr0_grid_real, 0, grid_size * grid_size * sizeof(float));
	cudaMemset(rr0_grid_imag, 0, grid_size * grid_size * sizeof(float));
	cudaMemset(rr1_grid_real, 0, grid_size * grid_size * sizeof(float));
	cudaMemset(rr1_grid_imag, 0, grid_size * grid_size * sizeof(float));
	cudaMemset(rr2_grid_real, 0, grid_size * grid_size * sizeof(float));
	cudaMemset(rr2_grid_imag, 0, grid_size * grid_size * sizeof(float));
	cudaMemset(output_index, 0, image_size * image_size * 2 * sizeof(float));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error sync 1 : %s\n", cudaGetErrorString(cudaStatus));
	}
	
	cudaStream_t stream1, stream2, stream_fft0, stream_fft1, stream_fft2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream_fft0);
	cudaStreamCreate(&stream_fft1);
	cudaStreamCreate(&stream_fft2);
	
	cudaEventRecord(start);
	/* ****************************************************** */
	size_t num_threads = 1024;
	size_t num_blocks = computeCeil(static_cast<float>(image_size/2+1)/num_threads);
	convolveKernel<<<num_blocks,num_threads,0,stream2>>>(conv_corr_kernel, image_size, grid_size, conv_corr_norm_factor);
	
	float inten_scale = std::abs(Vin[0*3+0]*Vin[1*3+1]-Vin[0*3+1]*Vin[1*3+0]);
	
	/* ****************************************************** */
	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(num_baselines)/num_threads);
	computeVisWeighted<<<num_blocks,num_threads,0,stream1>>>(Vis_real,Vis_imag,num_baselines,inten_scale);
	
	/* ****************************************************** */
	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(num_baselines)/num_threads);
	gridding<<<num_blocks,num_threads,0,stream1>>>(
            B_in,
			rr0_grid_real, rr0_grid_imag,
			rr1_grid_real, rr1_grid_imag,
			rr2_grid_real, rr2_grid_imag,
			Vis_real, Vis_imag, uv_scale, grid_size, num_baselines);
	
    /* ****************************************************** */
	cufftHandle plan0, plan1, plan2;
	cufftCreate(&plan0);
    cufftCreate(&plan1);
    cufftCreate(&plan2);
	cufftSetStream(plan0, stream_fft0);
    cufftSetStream(plan1, stream_fft1);
    cufftSetStream(plan2, stream_fft2);
	cufftPlan2d(&plan0, grid_size, grid_size, CUFFT_C2C);
    cufftPlan2d(&plan1, grid_size, grid_size, CUFFT_C2C);
    cufftPlan2d(&plan2, grid_size, grid_size, CUFFT_C2C);

    cudaEvent_t fft_ready, fft_done0, fft_done1, fft_done2;
	cudaEventCreate(&fft_ready);
	cudaEventCreate(&fft_done0);
	cudaEventCreate(&fft_done1);
	cudaEventCreate(&fft_done2);

	cudaEventRecord(fft_ready, stream1);
	cudaStreamWaitEvent(stream_fft0, fft_ready, 0);
	cudaStreamWaitEvent(stream_fft1, fft_ready, 0);
	cudaStreamWaitEvent(stream_fft2, fft_ready, 0);

	num_threads = 1024;
	num_blocks = computeCeil(static_cast<float>(grid_size * grid_size)/num_threads);
	combineToComplex<<<num_blocks,num_threads,0,stream_fft0>>>(rr0_grid_real, rr0_grid_imag, rr0_grid_stack, grid_size);
    combineToComplex<<<num_blocks,num_threads,0,stream_fft1>>>(rr1_grid_real, rr1_grid_imag, rr1_grid_stack, grid_size);
    combineToComplex<<<num_blocks,num_threads,0,stream_fft2>>>(rr2_grid_real, rr2_grid_imag, rr2_grid_stack, grid_size);
    
	/* ****************************************************** */
	cufftExecC2C(plan0, rr0_grid_stack, rr0_grid_stack, CUFFT_INVERSE);
    cudaEventRecord(fft_done0, stream_fft0);
    cufftExecC2C(plan1, rr1_grid_stack, rr1_grid_stack, CUFFT_INVERSE);
    cudaEventRecord(fft_done1, stream_fft1);
    cufftExecC2C(plan2, rr2_grid_stack, rr2_grid_stack, CUFFT_INVERSE);
    cudaEventRecord(fft_done2, stream_fft2);

	cudaStreamWaitEvent(stream1, fft_done0, 0);
	cudaStreamWaitEvent(stream1, fft_done1, 0);
	cudaStreamWaitEvent(stream1, fft_done2, 0);

	/* ****************************************************** */
	num_threads = 32;
	dim3 numThreads(num_threads, num_threads);
	dim3 numBlocks(computeCeil(static_cast<float>(image_size)/num_threads), computeCeil(static_cast<float>(image_size)/num_threads));
	accumulation<<<numBlocks,numThreads,0,stream1>>>(
            dirty_pre, rr0_grid_stack, rr1_grid_stack, rr2_grid_stack, V_in,
			image_size, grid_size, cell_size);
    
	/* ****************************************************** */
	numThreads.x = num_threads;
	numThreads.y = num_threads;
	numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
	numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
	scaling<<<numBlocks,numThreads,0,stream1>>>(dirty_pre, conv_corr_kernel, image_size, conv_corr_norm_factor);
	
	/* ****************************************************** */
	numThreads.x = num_threads;
	numThreads.y = num_threads;
	numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
	numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
	coordschange<<<numBlocks,numThreads,0,stream2>>>(output_index, V_in, image_size);
	
	/* ****************************************************** */
	numThreads.x = num_threads;
	numThreads.y = num_threads;
	numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
	numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
	p2p<<<numBlocks,numThreads,0,stream2>>>(output_index, V_in, cell_size, image_size);
	
	cudaEventRecord(eventstream,stream2);
	
	cudaStreamWaitEvent(stream1,eventstream,0);
	
	/* ****************************************************** */
	numThreads.x = num_threads;
	numThreads.y = num_threads;
	numBlocks.x = computeCeil(static_cast<float>(image_size)/num_threads);
	numBlocks.y = computeCeil(static_cast<float>(image_size)/num_threads);
	finalinterp<<<numBlocks,numThreads,0,stream1>>>(output_index, dirty_pre, dirty, image_size, num_baselines);
	
	cudaStreamSynchronize(stream1);

	cufftDestroy(plan0);
	cufftDestroy(plan1);
	cufftDestroy(plan2);
	cudaEventDestroy(fft_ready);
	cudaEventDestroy(fft_done0);
	cudaEventDestroy(fft_done1);
	cudaEventDestroy(fft_done2);
	cudaEventDestroy(eventstream);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream_fft0);
	cudaStreamDestroy(stream_fft1);
	cudaStreamDestroy(stream_fft2);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    
	cudaMemcpy(dirty_image, dirty, image_size * image_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error sync 2 : %s\n", cudaGetErrorString(cudaStatus));
	}
	
	cudaFree(Vis_real);
	cudaFree(Vis_imag);
	cudaFree(B_in);
	cudaFree(V_in);
	cudaFree(dirty);
	cudaFree(dirty_pre);
	cudaFree(conv_corr_kernel);
	cudaFree(rr0_grid_real);
	cudaFree(rr0_grid_imag);
	cudaFree(rr1_grid_real);
	cudaFree(rr1_grid_imag);
	cudaFree(rr2_grid_real);
	cudaFree(rr2_grid_imag);
	cudaFree(rr0_grid_stack);
	cudaFree(rr1_grid_stack);
	cudaFree(rr2_grid_stack);
	cudaFree(output_index);
	
	return 0;
}
	
