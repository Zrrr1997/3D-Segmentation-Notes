# 3D-Segmentation

## Goals
-	Computer aided diagnosis
	-	Efficiency and accuracy

## Issues
-	Medical images suffer from blur, noise, and low contrast (problem when using classical approaches)
	-	Almost impossible to obtain accurate boundaries from low-level features alone (as they are inherently noisy)
	-	Also almost impossible from deeper semantic features only as they exclude the image detail information
	-	Mitigated by CNNs which utilize both (e.g. skip-connections)

## Which parts are segmented?
-	Liver-tumor
-	Brain and brain-tumor
-	Optic disk
-	Cell 
-	Lung
-	Pulmonary nodules
-	Cardiac



## Supervised Learning
-	FCN, **U-Net** (encoder-decoder model)
	-	Output restores the original image size, but with segmentations
	-	Pragmatic, but reduces the interpretability
-	3D CNNs (3D U-Net, **V-Net**)
	-	Directly apply to volumetric data (CT/MRI)
	-	Typical issue: large complexity and too many parameters

-	RNN (ResUNet + RNN, **uncommon**)
	-	Captures the local and global spatial features
	-	Works well with good medical image quality data
		-	Smaller slice thickness
-	Skip Connections
	-	Fuses low and high-level features
	-	Problem: semantic gap between low- and high-level features leads to a blurred feature map
		-	Usually solved by addiding additional convolutional operations to the skip connection to reduce the semantic gap (MultiResUNet, FED-Net, mUNet)

-	Cascade Models
	-	Train **two or more models** to improve segmentation
	-	**Coarse-fine segmentation**
		-	One CNN for coarse segmentation (e.g. livers), and another for fine which takes the course result as input (e.g. liver-tumors)
	-	**Detection segmentation**
		-	First an object detector (R-CNN, YOLO)
		-	Then a model for detailed segmentation on the detection
	-	**Mixed cascade (hybrid**
		-	H-DenseUNet - Use 2D DenseUNet to extract 2D features and 3D DenseUNet for 3D features and a **hybrid feature fusion layer** to jointly optimize them




