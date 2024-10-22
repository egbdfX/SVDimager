# SVD Imager

We have developed a GPU-accelerated SVD imager for Fast Imaging in radio astronomy. While it is a core component of the Fast Imaging Pipeline for transient detection, it can also function as a standalone imager, independent of the full pipeline. Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingPipe/tree/main#reference) for more information.

## User guidance

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are avaiable. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
If you do not need reprojection in the imaging, please download the files in 'NoReproj'; otherwise, please download the files in 'WithReproj'.

**Step 3:**
Run the Makefile by ```make```. Note that this Makefile is written for NVIDIA H100. If you are using other GPUs, you would need to make sure the CUDA arch is matching.

**Step 4:**
Run the code by executing the following command:

```./sharedlibrary_gpu Visreal_input.fits Visimag_input.fits B_input.fits V_input.fits Image_Size Number_of_Baselines Frequency Cell_Size Output_Name.fits```.

Here, ```Visreal_input.fits```, ```Visimag_input.fits```, ```B_input.fits```, and ```V_input.fits``` are the input files (in FITS format) corresponding to the real components of visibilities, the imagery components of visibilities, the (centred) SVDed baseline matrix, and the V matrix in the SVD, respectively. The remaining arguments are as their names suggest, where ```Image_Size``` is an integer (e.g., if you input 100, it means the image size is $100 \times 100$ pixels), ```Number_of_Baselines``` is an integer, ```Frequency``` is in units of Hz, ```Cell_Size``` is in units of radians, and the last argument is the name of the output file which should end with '.fits'.

**Step 5:**
The code will output a FITS file named ```Output_Name.fits``` (as user defined), which is the output snapshot.

## Test
If you want to test the code, please download the files from 'ExampleInput' of the corresponding reprojection method. Run the code by ```./sharedlibrary_gpu Visreal0.fits Visimag0.fits Bin0.fits Vin0.fits 4096 2080 50000000 0.0000213 dirty0.fits```. You should obtain a FITS file named dirty0.fits. If you open it (by SAOImageDS9, Fv or MATLAB etc), you will see a simulated sky brightness distribution of regular distributed sources. 

## Contact
If you have any questions or need further assistance, please feel free to contact at [egbdfmusic1@gmail.com](mailto:egbdfmusic1@gmail.com).

## Reference

**When referencing this code, please cite our related paper:**

X. Li, K. Adámek, M. Giles, W. Armour, "Fast imaging pipeline for transient detection with GPU acceleration," 2024.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
