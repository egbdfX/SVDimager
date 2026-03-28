#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include <vector>
#include <iostream>
#include <chrono>

int FIpipe(float* Visreal, float* Visimag, float* Bin, float* Vin, float* dirty_image, size_t num_baselines, size_t image_size, float cell_size);

using namespace std;

float* read_fits_image(const char* filename, long* naxes) {
    fitsfile *fptr;
    int status = 0;

    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(stderr, status);
        return NULL;
    }

    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }
    
    printf("DEBUG: Reading %s, naxis=%d\n", filename, naxis);
    
    // CRITICAL FIX: Handle 1D and 2D arrays differently
    long total_elements;
    if (naxis == 1) {
        // For 1D arrays, only get one dimension
        long temp_naxes[1];
        fits_get_img_size(fptr, 1, temp_naxes, &status);
        if (status) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return NULL;
        }
        naxes[0] = temp_naxes[0];
        naxes[1] = 1;  // Force second dimension to 1 for consistency
        total_elements = naxes[0];
        printf("DEBUG: 1D array, naxes[0]=%ld, total_elements=%ld\n", naxes[0], total_elements);
    } else if (naxis == 2) {
        // For 2D arrays, get both dimensions
        fits_get_img_size(fptr, 2, naxes, &status);
        if (status) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return NULL;
        }
        total_elements = naxes[0] * naxes[1];
        printf("DEBUG: 2D array, naxes[0]=%ld, naxes[1]=%ld, total_elements=%ld\n", 
               naxes[0], naxes[1], total_elements);
    } else {
        fprintf(stderr, "ERROR: Unsupported naxis=%d for file %s\n", naxis, filename);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Allocate based on actual total elements
    float *image_data = (float *)malloc(total_elements * sizeof(float));
    if (image_data == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed for %ld elements (%.2f MB)\n", 
                total_elements, (total_elements * sizeof(float)) / (1024.0 * 1024.0));
        fits_close_file(fptr, &status);
        return NULL;
    }

    printf("DEBUG: Allocated %.2f MB for %ld elements\n", 
           (total_elements * sizeof(float)) / (1024.0 * 1024.0), total_elements);

    // Read the actual number of elements
    fits_read_img(fptr, TFLOAT, 1, total_elements, NULL, image_data, NULL, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        fits_close_file(fptr, &status);
        return NULL;
    }

    printf("DEBUG: Successfully read %ld elements, first value=%.6e, last value=%.6e\n", 
           total_elements, image_data[0], image_data[total_elements-1]);

    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        return NULL;
    }

    return image_data;
}

int write_fits_image(const char* filename, float *image_data, long* naxes) {
    fitsfile *fptr;
    int status = 0;
    
    fitsfile *tmp_fptr;
    fits_open_file(&tmp_fptr, filename, READWRITE, &status);
    if (!status) {
        fits_delete_file(tmp_fptr, &status);
    }
    status = 0;

    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    long naxis = 2;
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    fits_write_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], image_data, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    char input_Visreal[1000];
    char input_Visimag[1000];
    char input_Bin[1000];
    char input_Vin[1000];
    char output_name[1000];
    
    long imasize[2];
    sprintf(input_Visreal, "%s", argv[1]);
    float *Visreal = read_fits_image(input_Visreal, imasize);
    if (Visreal == NULL) {
        return 1;
    }
    
    sprintf(input_Visimag, "%s", argv[2]);
    float *Visimag = read_fits_image(input_Visimag, imasize);
    if (Visimag == NULL) {
        return 1;
    }
    
    sprintf(input_Bin, "%s", argv[3]);
    float *Bin = read_fits_image(input_Bin, imasize);
    if (Bin == NULL) {
        return 1;
    }
    
    sprintf(input_Vin, "%s", argv[4]);
    float *Vin = read_fits_image(input_Vin, imasize);
    if (Vin == NULL) {
        return 1;
    }
    
    size_t image_size = std::stoul(argv[5]);
    float* dirty_image = (float*)malloc(image_size*image_size*sizeof(float));

    size_t num_baselines = std::stoul(argv[6]);
    float cell_size = std::stof(argv[7]);
    sprintf(output_name, "%s", argv[8]);
	
    FIpipe(Visreal, Visimag, Bin, Vin, dirty_image, num_baselines, image_size, cell_size);
	
    long naxes[2] = {long(image_size), long(image_size)};
    int status = write_fits_image(output_name, dirty_image, naxes);
    if (status) {
        fprintf(stderr, "Error writing FITS image\n");
        return 1;
    }
}
