# Multimodal Medical Imaging Fusion (2019)

## Related Work
-	PET/CT - “A combined PET/CT scanner for clinical oncology"
-	MRI/CT - Joint segmentation of anatomical and functional images:
Applications in quantification of lesions from PET, PET-CT, MRIPET, and MRI-PET-CT images
	-	MedIA (2013)
-	fMRI/EEG



## Method
-	Fusion could be at
	-	Feature level
		-	Modality concatenation
	-	Classifier level
		-	Each modality as separate input
		-	Single modality features are used to learn a multimodal classifier
	-	Decision-making level
	-	They do not mention the Mirror U-Net as an idea for multimodal fusion
-	Combination of MR (T1 and T2), PET, and CTs
	-	All in the same image size for each subject
-	The task is tumor classification on 2D patches
	-	The images are divided into 28x28 patches
	-	If the center of patch is labelled as tumor then the whole patch is considered as a tumor
	-	The multimodal models are trained for binary classification
-	Adding Gaussian Noise to each modality shows that
	-	CTs are very sensitive to noise (boundary becomes blurred and ambiguous)
	-	MRI and PET preserve the tumor's boundary
	-	Multimodal models are far less susceptible to noise
		-	Apart from the decision level model
-	Results show that early fusion outperforms late fusion
	-	Early fusion is more simple to implement
	-	Incrementally adding more modalities improves the performance
	-	Middle fusion is competitive to early fusion models
		-	Contribution quantification with with-without ablation studies
