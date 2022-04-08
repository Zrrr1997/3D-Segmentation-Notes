# Comparative Evaluation of Interactive Segmentation Approaches (2016)

## Aim
-	Comparison of segmentation quality of:
	-	Region Growing
	-	Watershed
		-	Requires to pre-compute the gradient
		-	It is sensitive to unclear edges
			-	If too many edges are detected the watershed tends to produce oversegmentation
	-	GrowCut
-	Metrics
	-	Rand Index
	-	Mutual Information
	-	Dice Coefficient
-	GrowCut is the best one

