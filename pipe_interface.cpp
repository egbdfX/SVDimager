#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include <vector>
#include <iostream>
#include <chrono>

int FIpipe(float* Visreal, float* Visimag, float* Bin, float* Vin, float* dirty_image, size_t num_baselines, size_t image_size, float freq_hz, float cell_size);

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
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    float *image_data = (float *)malloc(naxes[0] * naxes[1] * sizeof(float));
    if (image_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    fits_read_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], NULL, image_data, NULL, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        fits_close_file(fptr, &status);
        return NULL;
    }

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
    float freq_hz = std::stof(argv[7]);
    float cell_size = std::stof(argv[8]);
	sprintf(output_name, "%s", argv[9]);
	
	FIpipe(Visreal, Visimag, Bin, Vin, dirty_image, num_baselines, image_size, freq_hz, cell_size);
	
	long naxes[2] = {long(image_size), long(image_size)};
    int status = write_fits_image(output_name, dirty_image, naxes);
    if (status) {
        fprintf(stderr, "Error writing FITS image\n");
        return 1;
    }
}
