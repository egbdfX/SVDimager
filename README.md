# Transient-Oriented Imager (TOI)

We develop a GPU-accelerated Transient-Oriented Imager (TOI) for Fast Imaging in radio astronomy. While it is a core component of the Fast Imaging Pipeline for transient detection, it can also function as a standalone imager, independent of the full pipeline. Please see our paper in Section [Reference](https://github.com/egbdfX/SVDimager/tree/main#reference) for more information.

## User guidance

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are available. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
Run the Makefile by ```make```. Note that this Makefile is written for NVIDIA H100. If you are using other GPUs, you would need to make sure the CUDA arch is matching.

**Step 3:**
Run the code by executing the following command:

```./sharedlibrary_gpu Visreal_input.fits Visimag_input.fits B_input.fits V_input.fits Image_Size Number_of_Baselines Frequency Cell_Size Output_Name.fits```.

Here, ```Visreal_input.fits```, ```Visimag_input.fits```, ```B_input.fits```, and ```V_input.fits``` are the input files (in FITS format) corresponding to the real components of visibilities, the imaginary components of visibilities, the (centred) SVDed baseline matrix, and the V matrix in the SVD, respectively. The remaining arguments are as their names suggest, where ```Image_Size``` is an integer (e.g., if you input 100, it means the image size is $100 \times 100$ pixels), ```Number_of_Baselines``` is an integer, ```Frequency``` is in units of Hz, ```Cell_Size``` is in units of radians, and the last argument is the name of the output file which should end with '.fits'.

**Step 4:**
The code will output a FITS file named ```Output_Name.fits``` (as user defined), which is the output snapshot.

## Test
If you want to test the code, please download the files from 'ExampleInput'. Run the code by ```./sharedlibrary_gpu Visreal0.fits Visimag0.fits Bin0.fits Vin0.fits 4096 2080 50000000 0.0000213 dirty0.fits```. You should obtain a FITS file named ```dirty0.fits```. If you open it (by SAOImageDS9, Fv or MATLAB etc), you will see a simulated sky brightness distribution of regular distributed sources. 

## Contact
If you have any questions or need further assistance, please feel free to contact at [egbdfmusic1@gmail.com](mailto:egbdfmusic1@gmail.com).

## Reference

**When referencing this code, please cite our related paper:**

X. Li, K. Adámek, O. Bilaniuk, V. Stolyarov, W. Armour, "[FIP-TOI: Fast Imaging Pipeline for Pulsar Localisation with a Transient-Oriented Radio Astronomical Imager](https://arxiv.org/abs/2512.06254)," 2026.

## License

Shield: [![BSD 3-Clause][bsd-3-shield]][bsd-3]

This work is licensed under a
[BSD 3-Clause License][bsd-3].

[bsd-3]: https://opensource.org/licenses/BSD-3-Clause
[bsd-3-shield]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
