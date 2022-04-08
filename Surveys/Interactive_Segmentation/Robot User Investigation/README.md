# Interactive Neural Network Robot User Investigation for Medical Image Segmentation (2019)
-	DeepIGeoS and UI-Net have proposed robot users during training

## Robot users
-	Random sampling over whole image
-	Random sampling from the ground truth mask
-	Kohli (2012) 
	-	Place one seed point in the center of the largest, wrongly labeled image area
-	DIOS (2016)
	-	3 Strategies (see notes in Interactive Segmentation)
-	DeepIGeoS (2017)
	-	Place seed point on random locations on the wrongly segmented areas]


## Segmentation Model
-	GrowCut
	-	Dice Score measured after each iteration
-	DeepIGeoS > DIOS > Random GT > Random > Kohli
	-	However, using DeepIGeoS's pattern works only if the user behaves like the robot user during training!
	-	Random users are more generalizable


