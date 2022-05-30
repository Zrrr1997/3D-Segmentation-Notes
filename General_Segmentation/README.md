# General 3D Segmentation Models

# PHNN (2017)

## Related Work
-	Leading PLS approaches rely on prior 3D shape or anatomical landmarks
	-	Top-down approaches
	-	Pro: helps delineate regions hard to discriminate solely based on intensity features
	-	Con: large abnormalities are a severe challenge

# nnU-Net (2018)

## Motivation
-	Exact architecture, pre-processing, training, inference cause U-Net to underperform
-	Hyperparameter search, archtecture etc. must be designed for each dataset
-	Comparing to a baseline is not really fair
	-	Baseline might not be tuned correctly and easy to beat 


## Method
-	Fixed parameters
	-	Learning rate
	-	Loss function
	-	leaky ReLU instead of ReLU
	-	Instance normalization instead of batch normalization
	-	Optimizer
	-	Data augmentation
-	Rule-Based parameters
	-	Data fingerprint used to infer these parameters
		-	Median shape of images
		-	Intensity distribution
		-	Image Modality
	-	Resampling, Re-shape, Patch size, Batch size, GPU-memory limit
	-	Implemented via **heuristic**
		-	Follow certain rules until all parameters are fixed
		-	Makes it possible to prioritize different parameters
			-	E.g. patch size vs. batch size
-	Empirical Parameters
	-	Postprocessing
		-	Suppressiion of non-maximum components (exclude small structures)
	-	Ensembling
-	Cropp to nonzero regions
-	CT- HU units are clipped to [0.5,99.5] percentiles and then z-score normalized
-	Automatically chooses the best model architecture 
	-	2D UNet, 3D UNet, Cascade UNet based on 5-fold cross-validation




