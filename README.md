# Medical-Image-Processing__ADF_IDF_DiffusionFilter_AMF_LowRankMatrix


# Overview
This repository contains a comprehensive Jupyter Notebook demonstrating various image denoising techniques. Each section covers a different algorithm, providing both theoretical background and practical implementation. The notebook includes step-by-step explanations, mathematical formulations, and visualizations to aid understanding.

## Table of Contents
1. [Introduction](#introduction)
2. [Adaptive Median Filter (AMF)](#adaptive-median-filter-amf)
3. [Adaptive Weighted Mean Filter (AWMF)](#adaptive-weighted-mean-filter-awmf)
4. [Low Rank Matrix Approximation (LRMA)](#low-rank-matrix-approximation-lrma)
5. [Weighted Nuclear Norm Minimization (WNNM)](#weighted-nuclear-norm-minimization-wnnm)
6. [Anisotropic Diffusion Filter (ADF)](#anisotropic-diffusion-filter-adf)
7. [Isotropic Diffusion Filter (IDF)](#isotropic-diffusion-filter-idf)
8. [Additional Functions and Visualizations](#additional-functions-and-visualizations)

## Introduction
The notebook begins with an introduction to image noise and the importance of denoising in image processing. Different types of noise are discussed, with a focus on impulse noise and its impact on image quality.

## Adaptive Median Filter (AMF)
The AMF section covers:
- **Theory:** Explanation of the adaptive median filter and its advantages over the standard median filter.
- **Mathematical Formulation:** Detailed description of the algorithm's steps, including window size adjustment based on noise characteristics.
- **Implementation:** Python code for applying the AMF to noisy images.
- **Results:** Visual comparison between original, noisy, and denoised images using AMF.

## Adaptive Weighted Mean Filter (AWMF)
The AWMF section includes:
- **Theory:** Introduction to the weighted mean filter and its adaptation to handle impulse noise effectively.
- **Mathematical Formulation:** Step-by-step explanation of the AWMF algorithm, including weight calculation and window size adjustments.
- **Implementation:** Python code for the AWMF algorithm.
- **Results:** Visualization of the filtering process and comparison of results with other methods.

## Low Rank Matrix Approximation (LRMA)
The LRMA section explains:
- **Theory:** Concept of low rank matrix approximation for noise reduction.
- **Mathematical Formulation:** Detailed description of the LRMA algorithm, including matrix decomposition techniques.
- **Implementation:** Python code for LRMA.
- **Results:** Visual results of the LRMA filter applied to noisy images.

## Weighted Nuclear Norm Minimization (WNNM)
The WNNM section discusses:
- **Theory:** Introduction to nuclear norm minimization and its application in image denoising.
- **Mathematical Formulation:** Detailed steps of the WNNM algorithm, including matrix decomposition and weight adjustments.
- **Implementation:** Python code for WNNM.
- **Results:** Comparison of denoising results with other algorithms, highlighting the effectiveness of WNNM.

## Anisotropic Diffusion Filter (ADF)
The ADF section covers:
- **Theory:** Explanation of anisotropic diffusion and its benefits for edge-preserving smoothing.
- **Mathematical Formulation:** Detailed description of the diffusion process, including the calculation of diffusion coefficients.
- **Implementation:** Python code for applying ADF to noisy images.
- **Results:** Visual comparison between original, noisy, and denoised images using ADF.

## Isotropic Diffusion Filter (IDF)
The IDF section includes:
- **Theory:** Introduction to isotropic diffusion filtering and its nature for uniform diffusion across the image.
- **Mathematical Formulation:** Step-by-step explanation of the IDF algorithm, including iteration procedures and diffusion updates.
- **Implementation:** Python code for the IDF algorithm.
- **Results:** Visualization of the isotropic filtering process and comparison of results with other methods.

## Additional Functions and Visualizations
This section includes utility functions and additional visualizations to support the main algorithms:
- **Theory Section:** Detailed explanation of theoretical concepts used in various algorithms, including optimization problems and matrix decompositions.
- **Derivative Calculation:** Functions to compute image derivatives.
- **Visualization Tools:** Code to generate plots and compare images at different stages of processing.

## Conclusion
The notebook concludes with a summary of the findings and suggestions for further improvements or alternative approaches for image denoising.

## How to Run
1. Clone the repository: `git clone https://github.com/AryaKoureshi/Medical-Image-Processing__ADF_IDF_AMF_LowRankMatrix_WNNM.git`
2. Navigate to the repository directory: `cd Medical-Image-Processing__ADF_IDF_AMF_LowRankMatrix_WNNM`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Launch Jupyter Notebook: `jupyter notebook ADF_IDF_AMF_LRMA_WNNM.ipynb`

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- OpenCV
- Scikit-Image
- PyTorch
- pyiqa
- CuPy
- SciPy

## Contributions
Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
