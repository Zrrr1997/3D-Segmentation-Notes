# Evaluation
Usually, evaluation metrics could encompass the scope of pixel quality, region quality and surface distance quality.

## Metrics
-	**Pixel quality**
	-	Pixel Accuracy (PA)
		-	Ratio of pixels properly classified, divided by the total number of pixels. 
-	**Region quality**
	-	Dice score
		-	Twice the overlap area of predicted and ground-truth maps, divided by the total number of pixels
		-	Indicates the volume agreement between the generated segmentation result and the ground truth segmentation mask
	-	Volume Overlap Error (VOE)
		-	Complement of the Jaccard index (1 - (overlap / union))
	-	Relative Volume Difference (RVD)
		-	(|B| - |A|)/ |A|
-	**Surface Distance quality**
	-	Defined over the surface voxels of a region.
	-	Average symmetric surface distance (ASD)
		-	The average difference between the surface of the segmented object and the ground truth segmentation mask in 3D. After the border voxels of the segmentation output and the ground truth segmentation mask in 3D.
	-	Maximum symmetric surface distance (MSD)
		-	Symmetric Hausdorff Distance

-	**Intraclass correlation coefficient (ICC)**
	-	Measure the performance consistency of different runs/annotator
