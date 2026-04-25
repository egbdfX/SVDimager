# Transient-Oriented Imager (TOI)

We develop a GPU-accelerated Transient-Oriented Imager (TOI) for Fast Imaging in radio astronomy. Fast Imaging is a technique to detect and localise celestial transients (e.g., pulsars, fast radio bursts) in image-frequency domain. While TOI is a core component of the Fast Imaging Pipeline for transient detection, it can also function as a standalone imager, independent of the full pipeline. This imager supports flag-based data correction, frequency-dependent weighting (via the WEIGHT_SPECTRUM column), and multi-frequency synthesis (MFS). Please see our paper in Section [Reference](https://github.com/egbdfX/SVDimager/tree/main#reference) for more information.

## User guidance

### Input Generation

Run ```python SVD_MFS.py``` to generate inputs from per-time-slot Measurement Sets. Adjust the number of loops to read multiple Measurement Sets.

### Running the Imager

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are avaiable. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
Run the Makefile by ```make```. Ensure the CUDA architecture flags (```arch=compute_xx,code=sm_xx```) match your GPU hardware.

**Step 3:**
Run the code by executing the following command:

```./sharedlibrary_gpu Visreal_input.fits Visimag_input.fits B_input.fits V_input.fits Image_Size Number_of_Rows Cell_Size Output_Name.fits```.

Here, **Visreal_input.fits**, **Visimag_input.fits**, **B_input.fits**, and **V_input.fits** are the input files (in FITS format) corresponding to the real components of visibilities, the imagery components of visibilities, the (centred) SVDed baseline matrix, and the V matrix in the SVD, respectively. The remaining arguments are as their names suggest, where **Image_Size** is an integer (e.g., if you input 128, it means the image size is $128 \times 128$ pixels), **Number_of_Rows = Number_of_Baselines * Number_of_Frequency_Channels** is an integer, **Cell_Size** is in units of radians, and the last argument is the name of the output file which should end with '.fits'.

**Step 4:**
The code will output a FITS file named **Output_Name.fits** (as user defined), which is the output snapshot. It can be opened by SAOImageDS9, Fv or MATLAB etc.

## Branches in this repo
**main** -- 3D imaging (more accurate);

**coplanar-demo** -- 2D approximation (applicable for most use cases);

**single-channel-demo** -- single-channel solution (for test).

## Contact
If you have any questions or need further assistance, please feel free to contact at [egbdfmusic1@gmail.com](mailto:egbdfmusic1@gmail.com).

## Reference

**When referencing this code, please cite our related paper:**

X. Li, K. Adámek, M. Giles, W. Armour, "[FIP-TOI: Fast Imaging Pipeline for Pulsar Localisation with a Transient-Oriented Radio Astronomical Imager](https://arxiv.org/abs/2512.06254)," 2025.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
