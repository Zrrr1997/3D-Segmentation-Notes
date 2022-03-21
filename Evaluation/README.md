# Evaluation
Usually, evaluation metrics could encompass the scope of pixel quality, region quality and surface distance quality.

## Metrics
-	**Pixel quality**
	-	Pixel Accuracy (PA)
		-	Ratio of pixels properly classified, divided by the total number of pixels. 
-	**Region quality**
	-	Dice score
		-	Twice the overlap area of predicted and ground-truth maps, divided by the total number of pixels
	-	Volume Overlap Error (VOE)
		-	Complement of the Jaccard index (1 - (overlap / union))
	-	Relative Volume Difference (RVD)
		-	(|B| - |A|)/ |A|
-	**Surface Distance quality**
	-	Defined over the surface voxels of a region.
	-	Average symmetric surface distance (ASD)
	-	Maximum symmetric surface distance (MSD)
		-	Symmetric Hausdorff Distance
