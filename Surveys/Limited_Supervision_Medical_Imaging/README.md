# Medical Image Segmentation With Limited Supervision: A Review of Deep Network Models

## Motivation
-	Labeling in medical settings is expensive and models need to adapt to limited training data by either a specific model design or a specific learning strategy.
-	Paper lists a lot of papers with the application domains (with references)
-	Deep and wide models require a large annotated training dataset 
	-	In order to generalize to other domains
-	Why is it hard to collect medical segmentation data?
	-	Requires expert knowledge and is time-consuming
	-	Data differences in patient populations
	-	Acquisition parameters and protocols
	-	Strict legal and ethical requirements for patient privacy
## Related Work
-	[16], [23], [24] also review segmentation methods for medical images
-	![](../../images/taxonomy-limited-supervision.png)

## Overview of medical image segmentation
-	Problems:
	-	Medical images are heterogenous with imbalanced classes [17]
	-	Multiple modalities with sparse annotations
-	CNN Models (typical)
	-	U-Net, PSP-Net, DeepLab, Attention U-Net etc.

## Semi-Supervised learning
-	Small subset of labeled training data
	-	Semi-supervised learning 
	-	However, abundance of **unlabeled images**
	-	Semi-supervised methods include
		-	Transfer learning
			-	Re-use a model developed on a task as a starting point for a model for a second task
			-	Model "pre-training and fine-tuning strategy"	
				-	This tends to work if the features are general and suitable for both the source and target tasks
				-	2D pre-trained models are hard to apply to 3D medical models 
					-	If you just reduce it to slice-by-slice 2D medical segmentation, you lose spatial 3D context information
				-	Large domain gap between general and medical images
				-	There is not large-scale medical image dataset like ImageNet
		-	Data augmentation
			-	Artificially enhance the size, diversity and quality of the training data
			-	Generating new labeled data with GANs 
		-	Prior knowledge learning
			-	Incorporate prior knowledge (domain knowledge) about location, shape, anatomy, and context 
			-	Anatomical-prior guided methods assume the solution space can be expressed in the form of a prior distribution, enforcing the network to generate more anatomically plausible segmentations
			-	Atlas-based segmentation 
				-	Segmentation problem treated as a registration problem
					-	Atlas might be an image/mesh with known mask ---> task is to "warp/transform" it to an unknown image (and then use the same transform for the mask)
				-	Correspondences between atlas and the unlabelled images
					-	Very computationally expensive
			-	Shape-prior based segmentation
				-	Active Shape Model (warp prior anatomical model to match your image/volume)
		-	Cirruculum learning
			-	Easy-to-Hard strategy
				-	Starts with easier aspects of the task or easier subtasks and then gradually increases the difficulty level.
				-	First learn the "easy samples" and then the "hard samples" (DATA curriculum learning)
					-	These change over the training. Dynamically adapting the ranking is called **self-paced** learning
				-	First learn the "easy tasks"
					-	Similar to multi-stage learning (first coarse segmentation / then pixelwise-segmentation)
		-	Self-training (pseudo-labeling)
			-	Iterative process that alternatively generates pseudo-labels on the unlabeled data and retrains the learner on the combined labeled and pseudo-labeled data.
			-	Very similar to **active learning for interactive segmentation**, but model generates labels, instead of the **human-in-the-loop**
			-	Straightforward, but effective way of leveraging unlabelled data
			-	Student-Teacher model
				-	Teacher is trained on GT-data and used to generate pseudo-labels for the student
				-	Teacher should also generate uncertainty maps of its predictions
					-	Bayesian networks
						-	Learn a posterior distribution over parameters
					-	MC-Dropout
						-	Averaging the results of multiple stochastic forward passes under random dropout.
							-	Uncertainty = normalized entropy of the predictions, variance of the samples or mutual information
					-	MC-BatchNorm
					-	Deep Ensembles	
		-	Co-training
			-	Train multiple base-classifiers on different subsets of data (views)
			-	Generate pseudo-labels on unseen subsets, if the base-classifiers have a high agreement
		-	Consistency regularization
			-	Assumes a good model with generate consistent predictions for similar inputs
				-	Equivalent predictions for perturbed inputs (unlabelled)
				-	
		-	Self-supervised learning
			-	Type of unsupervised learning
			-	Leverages the unlabeled data with automatically generate supervisory signal (e.g. jigsaw task).
			-	Aims to obtain a good representation of the training data without using any manual label.
				-	Learn generic concepts which can be transferred to downstream tasks.
		-	Adversarial learning
			-	Can be used so **synthesize** additional data
		-	Few-shot learning
			-	Learn a model on base classes and only k-labeled images of unseen classes (can be zero ---> zero-shot learning)
				-	Without re-training the model
				-	k-images are referred to as the support set
-	![](../../images/semi-supervised.png)
## Partial annotations


## Inaccurate annotations
	-	Noisy labels, Bounding Boxes, Scribbles
-	
