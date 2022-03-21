# 3D-Segmentation

## Goals
-	Computer aided diagnosis
	-	Efficiency and accuracy

## Issues
-	Medical images suffer from blur, noise, and low contrast (problem when using classical approaches)
	-	Almost impossible to obtain accurate boundaries from low-level features alone (as they are inherently noisy)
	-	Also almost impossible from deeper semantic features only as they exclude the image detail information
	-	Mitigated by CNNs which utilize both (e.g. skip-connections)
-	Objects come in large range of scales
	-	E.g., a tumor could be a different size, depending on the stage
	-	Leads to a large class imbalance 

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
		-	LW-HCN - Lightweight version of the top one (depthwise and 3D depth separable convolutions)
	-	**Other**
		-	KiUNet - Improvement of UNet with a cascade of another CNN
-	GAN-Based
	-	Conditional GAN (cGAN)
		-	Generator is trained to output segmentation results
		-	Discriminator guesses whether the output is from the generator or the ground-truth label
		-	cGAN works fine when training data is limited
-	Deeply Supervised Networks (DSN)
	-	Add some auxiliary branching classifiers to some layers
	-	3D-DSN for volume-to-volume learning

-	Strategies
	-	Dense Connections 
		- 	Propagate previous layers to all of the late layers
	-	Inception modules (**CE-Net** - GoogleNet for medical imaging)
		-	Mitigate vanishing gradient, network convergence, large memory	
		-	Different receptive fields are combined - learn features with different scales
	-	Depth Separability
		-	MobileNet - Depthwise separable + Pointwise convolution
			-	Reduce memory usage
		-	**LV-Net** - depth separable convolutions for 3D conv layers
			-	Cons: It may result on loss of accuracy in medical image segmentation

-	Non-local attention
	-	Non-local U-Net (Transformer architecture)
	-	Spatial attention looks for interesting target regions
	-	Channel attention looks for interesting features

-	Multi-scale features
	-	Atrous Spatial Pyramid Pooling (ASPP)
		-	Dilate the filter size to enlarge the receptive field
		-	Cons: Loss of detail due to the gridding effect

## Loss Function
-	Cross Entropy
	-	Simple, popular
	-	Cons: Ignores Class imbalance, shows low performance for small target segmentation
-	Weighted Cross Entropy
	-	Reduces the class imbalance problem by weighting the contribution of each class to the loss
-	Dice Loss
	-	Very similar to IoU - measures the degree of overlap between segmentation and GT
	-	Intersection can be implemented as a dot product (in VNet)
	-	Cons: Gradients of Dice Loss can explode for small objects
-	Tversky Loss
	-	Regularized Dice Loss
-	Generalized Dice Loss
	-	Solves the class imbalance problem	
	-	More stable and robust to Dice Loss
-	Boundary Loss
	-	Minimize the distance between segmented boundaries and GT boundaries
-	Exponential Logarithmic Loss
	-	Combine Focal Loss and Dice Loss

## Weakly-Supervised Learning
-	Motivation: Data acquisition and labeling often incur high costs, especially in the field of medical image analysis.
-	**Data Augmentation** - Basically Always Done
	-	Brightness, saturation, contrast, rotation, distortion, scaling
	-	Gaussian Blur impoves colon tissue segmentation
-	**Conditional GAN (cGAN)**
	-	cGAN adds a **condition** to the GAN to guide the generation of images	
-	**Transfer Learning**
	-	Utilize trained parameters of a model to initialize new model -> fast model training with limited labels
	-	Train on ImageNet, fine-tune on medical images
		-	Cons: Often this is unstable and test dataset-dependent
-	**Domain Adaptation**
	-	Use CycleGAN to generate MRI images from CT images and **labels**
		-	Train segmentation model on generated + real MRI images
		-	15% better than just training on real MRI images
-	**Interactive Segmentation**
	-	Manually drawing medical image segmentation labels is usually tedious and time-consuming, especially for the drawing of 3D volume data.
	-	Clinicials can correct the initially segmented image generated by a model.
	-	Method: 
		-	Use simply user interactions for labeling
			-	Mouse clicks, outline boxes
		-	Then model updates its parameters and generates new segmentation images to obtain new feedback from the clinicials
	-	Related work:
		-	DeepIGeoS
			-	Cascade of two CNNs for 2D and 3D images
			-	Reduces needed user interaction time
		-	BIFSeg (similar to GrabCut)
			-	User draws BBox and this region is given as an input to a segmentation model
		-	GM Interacting
			-	Updates the model w.r.t. user **text** input
## Semi-supervised 
-	Segmentation + Evaluation Network
	-	Output of segmentation network and original image are given to an evaluation network
		-	Goal of segmentation model is to maximize the evaluation network's score
		-	Evaluation network should also obtain high scores for unlabeled images
		-	Evaluation network is updated to assign low scores to unmarked images but high scores to marked images

## Miscelanious Research Directions
-	Network Architecture Search (NAS)
	-	Search space: candidate collection of network structures to be searched
	-	Search strategy: aims to find the optimal network structure as fas as possible
		-	RL, evolutionary algorithms, and gradients
	-	Performance estimation: how well the network performs on target datasets
	-	NASU-Net outperforms U-Net
-	Multimodality
	-	CNNs can be trained on MRI + CT data jointly 






















