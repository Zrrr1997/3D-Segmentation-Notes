# Interactive Segmentation

# GraphCut (2001) - seminal paper

## Related Work
-	Snakes, deformable templates, shortest path, ratio regions only work on 2D images
	-	Intelligent scissors and Live wire need very precise user interactions
-	Region growing, split and merge methods do not have cost functions
	-	Since the properties of what a "boundary" is are not optimized - leaking in blurry boundaries can be observed
-	Only an approximate solution - region competition, level set methods, normalized cuts
-	

## Method
-	Binary segmentation (Foreground-Background)
-	Hard constraints added by user
	-	Indicated Pixels = Seeds
-	Soft constraints regarding boundary and region properties
	-	Solution is find by min-cut/max-flow algorithm
-	Method can be applied to 3D images as well
-	Assume some pixels are marked as internal, and some as external to the object
-	![](../images/graphcut.png)
-	The boundary term defines the neighbouring edge weights
-	The regional terms defines the weight of the edges to the sink/source terminals
-	Regional terms (unary) is the negative log-likelihood, given the histogram of scribbles
-	Boundary terms (binary) are the pixel-intensity + spatial distance Gaussian Kernel
	-	Penalty when pixel intensity of neighbours with different labels is smaller than sigma


# GrabCut (2004) - Extension of GraphCut (2001)
-	Graph-Cut = Texture (colour) + Edge (contrast) information for foreground-background segmentation
-	Graph Cut
	-	Formulated as an energy minimization, which should correspond to a good segmentation
	-	Unary term: 
		-	Entropy of fore- and background greyscale values (histograms)
		-	![](../images/unary_grabcut.png)
		-	Encourages non-uniformity of the gray scale values assigned to the alphas {0, 1}
	-	Binary term (smoothness term):
		-	High energy if ---> pixels are different class + similar intensity + in close proximity
		-	![](../images/binary_grabcut.png)
	-	Graph Cut encourages coherence in regions of similar grey-levels.
		-	No need to define it as a fully-connected graph - look only at 8-neighbors (direct neighbors)
	-	Energy function can be optimized w.r.t. alphas, using standard minimum cut algorithm (GraphCut 2001)

-	GrabCut 
	-	Works with color data
	-	Replaces the histograms of grey values with **GMMs**
	-	One-shot minimum cut is replaced with an **iterative procedure** to update the GMM parameters, but also use Min-Cut/Max-Flow to estimate the segmentation in each iteration
	-	Only requires a **bounding-box** input interaction (or lasso)

# Lazy Snapping (2005)

## Related Work
-	They list related work as boundary- and region-based
	-	Boundary - cut out the foreground by user input surrounding curve which evolves iteratively
		-	Cons: Requires concentration and is error-prone
	-	Region 
		-	Cons: Low-contrast regions and shadows require more user-inputs

## Method
-	Uses scribbles!
-	Combines graph cut with pre-computed over-segmentation
-	Two steps:
	-	Quick object marking step
		-	Additionally + over-segmentation
	-	Simple boundary editing step
		-	Image cutout is formulated as a graph cut in both steps
-	Unary terms are computed using K-means clustering of the foreground and background scribbles
	-	Pixels with a high minimum distance to the foreground clusters are given a high energy when assigned x=1 
-	![](../images/lazysnapping_unary.png)
-	Pairwise term just compares the colour distance of adjacent pixels 
	-	High penalty when similar adjacent nodes are assigned different labels
-	To improve efficiency - the watershed (1991) algorithm is used as a **pre-segmentation**
	-	Watershed locates boundaries well and preserves small differences inside each small region
	-	Instead of using pixels as nodes, use watershed regions
	-	Color is aggregated as the mean color for all pixels in the regions for the unary term

# GrowCut (2005)

## Related Work
-	Region Growing algorithms might have a problem with leaking out of the object
	-	They also only work for binary segmentation
-	Graph-Cut based are not easily extended to multi-label tasks
	-	They are also not really flexible/tunable - only the graph weighting and cost function coefficients could be altered
-	Authors claim that random walker is the best performing method for medical segmentation
	-	Alongside graph cut, but random walker works with multi-label setups

## Method
-	![](../images/growcut.png)
-	Multi-label segmentation
-	Scribble-based 
-	Uses Cellular Automation
	-	Iterative process, user can guide the algorithm **during** the segmentation
-	Pixel labeling acts as a process of growth of K-types of bacteria
	-	Bacteria start to spread from the seed pixels and try to occupy all the image
	-	If the difference in color is small and the attacking cell's strength is large enough, the attacked cell takes the same label and strength of the attacking cell
-	The smoothness of the boundary can be achieved by prohibiting a cell to attack if there are too many different labels to it
-	A user can add a brush stroke at any time during the automata evolution process
	-	This changes the states in the cell automata
-	The brush strokes can also consider the maginitude of the certainty
	-	"This region is "maybe" foreground"
	-	This makes the process more tolerable to inaccurate paint strokes
		-	E.g. boundaries of objects like flowers or plant leaves
-	Initial strength of the cells is set by the user's certainty of the brush strokes

## Results
-	Comparison to Graph Cuts, Grab Cut and Random Walker

# GeoS (2007) - Predecessor of DeepIGeoS

## Motivation
-	Previous work (GraphCut (2001) and GrabCut (2004)) are computationally expensive and cannot be applied to high resolution images
-	Geodesic distance resticts the possible space of solutions -> it leads to a better search efficiency
-	Segmentation posterior is also found ---> uncertainty analysis
-	Algorithm can be sped up even further through parallelisation (which was not feasible with graph cuts as easily)
	-	Based on the raster scan from Toivanen (O(N)) (1998(

## Method
-	Approximate energy minimization in a conditional random field
-	Used for **binary segmentation**
-	Parallel fitlering operator for efficient geodesic distance computation
	-	Proposes spatially smooth, contrast-sensitive segmentations
-	Lowest energy solution is found in linear time
	-	With both Toivanen (Raster Scan) and Yatziv (Fast Marching Methods - FMM) algorithms
		-	However, Yatziv's algorithms employs FMM, which accesses the image lacations far from each other in memory
		-	Toivanen's technique accesses the image in contiguous blocks
-	The geodesic filter acts on the energy unaries in the CRF
	-	Large jumps in the geodesic distance correspond to stronger edges (due to the image gradient in the distance computation)
	-	![](../images/geos_distance.png)
-	Based on geodesic distance - dilation, erosion, closing and opening can be defined 
	-	Theta parameters can be set to filter foreground and background noise speckles and should be set to the maximum size of those
	-	Theta parameters, hence, regulate the **spatial smoothness** 
			- 	larger theta yield smoother segmentation boundaries
			- 	Smaller thetas are "locked" by the brightness structures
-	![](../images/closing_distance_geos.png)
-	Energy function can be formulated w.r.t. thetas - as they regulate the smoothness, they also regulate the energy
	-	Weak unaries (high uncertainty of the classifier) lead to vastly different results when thetas are altered
-	Search space for the theta parameters can be traversed linearly and in parallel

# Active Learning (2011) 

## Method
-	Evaluate a given segmentation by constructing an uncertainty field over the image
	-	Based on: boundary, regional, smoothnes and entropy terms
	-	The method highlights the plane of maximal uncertainty
	-	User can guide the labeling of the data providing additional training data where the classifier has the least confidence
	-	Compared to random region selection --> 64% faster convergence (good segmentation results)
-	Active Learning Definition
	-	Supervised machine learning, where instead of the user, a component of the algorithm called the **query strategy** is responsible for choosin the training data to be labeled.
	-	Using active learning makes the training more efficient (faster convergence)
-	Probabilistic algorithms are preferred to energy minimizing as they implicitly encode uncertainty (e.g. softmax predictions)
-	Only works for **binary segmentation**
-	Query strategies
	-	Uncertainty sampling
	-	Low confidence (same as above?)


		-	




# CRF-RNN (2015)

## Motivation 
-	Deep learning techniques alone have a limited capacity to delineate visual objects (exact position and border in the image)
-	It is important to explicitly consider factors such as image edges, appearance consistency, and spatial consistency
	-	Solution: Combine deep learning with CRF
-	Traditional CNNs have large receptive fields + max-pooling -> they obtain a coarse boundaries and blob-like shapes
-	CNNs also lack **smoothness contraints**, which can result in poor object delineation and small spurious regions
-	However, CRFs are able to refine weak and coarse pixel-level label predictions to produce **sharp** boundaries and fine-grained segmentation

**Note**: This paper has a very simple and precise explanation of the CRF mean-field approximation algorithm.

## Related Work
-	Previous work (DeepLab) uses CRF simply as a post-processing step to a CNN output
	-	Con: CNN cannot adjust its weight to the CRF (no updates, no knowledge transfer)

## Method
-	Formulate CRF with Gaussian pairwise potentials and mean-field approximation as RNNs
	-	RNN refines output of CNN and also passes back the differentials back during training
-	Unary potentials are the **inverse likelihoods** of a pixel belonging to a label
-	Q_i(l) is initialized as the softmax probability of pixel i to belong to label l
-	**Message passing** is done by applying 2 Gaussian kernels, 1 Spatial and 1 Bilateral kernels
-	**Weighted Filter Outputs** is done by a 1x1 convolution with M input channels (M = #Labels) and 1 output channel
-	**Compatibility Transform** is also a 1x1 convolution with in- and output channels = L. This learns the compatibility function which is more flexible than the Potts model with fixed penalty
-	Formulation as RNN
	-	Each iteration is simply a stack of CNN layers. 
	-	The original unary values from the segmentation model are fixed for all iterations
	-	The output Q values are used as an input to the next Q values estimation
	-	Around 10 iterations are enough for convergence (no vanishing or exploding gradients)
		-	Vanilla RNN is enough, as opposed to LSTM
		-	RNN timesteps correspond to the iterations from the classical algrotithm - 10 are enough
-	Con: Not all parameters are trainbable due to the permutohedral lattice implementation
	-	e.g. Gaussian, Spatial, and Bilateral filter parameters
-	![](../images/crf-rnn.png)


# SlicSeg (2015)

## Related work
-	Single Shot Fast Spin Echo (SSFSE) allows motion artefacts to be absent in the slices 
	-	However, an inhomogenous appearance between slices can occur, caused by the interleaved spatial order of the slices
-	Authors list Active Contours, Graph Cuts, Geodesic Frameworks (GeoS), Random Walks, GrowCut
	-	Most of these methods rely on low-dimensional features and need a large number of user interactions to deal with images with low contrast and weak boundaries

## Method
-	Placenta Segmentation
	-	Challenges are:
		-	Sparse acquisition
		-	Inter-slice motion
		-	Inter-Subject Variance in position and shape of the placenta
-	Random Forests with high-level features
	-	Combination of Online Random Forests and CRFs and a user interaction for **only one slice**
	-	ORF are trained on data coming from scribbles from the user in one slice
	-	Slice-by-slice propagation
-	Binary segmentation
-	CRF uses the prediction probabilities of the ORF and generates new segmentations (used to self-train the ORF)
-	First slice is used to train the initial RF
	-	CRF outputs the label for it
	-	Then the background and foreground are eroded using a filter of size 10 pixels and are fed to train the ORF
	-	The ORFs are trained by minimizing the entropy and each tree is trained on a random number of new samples 
	-	The CRFs are used every time to obtain pseudo-labels
-	
-	![](../images/slicseg.png)

## Result
-	Higher accuracy than sota interactive segmentation methods
	-	Compare segmentations using the same initial scribbles



# ScribbleSup (2016) 

## Method
-	Train CNN for semantic segmentation using scribbles as annotations
	-	In this paper a FCN 
-	Propose a graphical model which propagates scribble information to unmarked pixels 
	-	Using spatial, appearance and semantic constraints
-	In general - how to train a CNN not with pixel-wise labels, but with scribbles
	-	Easier and faster to annotate datasets this way
	-	Especially for ambiguous objects, sucha as water, sky, grass etc.
	-	Inspired by interactive segmentation with scribbles
-	Algorithm can be considered a weakly-supervised learning
-	To propagate the scribble labels to the whole image they employ CRFs over **superpixels**
-	The unary term are defined from the scribbles and the output of the FCN
	-	Scribbles:
		-	Zero cost to assign the label of the scribble if the pixel is part of a scribble
		-	Infinite cost to assign a scribble pixel to another label
		-	Uniform cost for all other labels, but only consider labels from the scribbles (subset of all classes) -> reduces the false positives
	-	FCN:
		-	Just the negative log likelihood of the FCN outputs
-	Pairwise potential
	-	Only for adjacent super-pixels
	-	This time, the potential uses the color and edge histogram distances (similar to GrabCut (2004))
-	Optimization has two variables - Y (labels) and FCN-parameters
	-	Fix theta -> solve for Y ---> Simple graph cut solution
	-	Then vice versa and repeat ---> Simple backpropagation
-	![](../images/scribblesup.png)

# ITK-Snap (2016)

## Method
-	Interactive tool for semi-automatic segmentation of multi-modality biomedical images
-	Focuses on **multi-modal** imaging, which uses all information simultaneously
	-	Modalities here are the multiple MRI contrasts
-	**Random Forsts** trained by several brushstrokes with **active contour** segmentation
	-	Based on texture, location and intensity features
-	User interaction for labeling though polygon and paintbrush (scribble) tool
-	**Stage 1**: A feature vector is formed for each labeled voxel by the user
	-	Intensity (appearance), intensities of neighbouring voxels (texture), and spatial coordinates (location)
	-	A classifier is trained on these features and then applied to the whole image to produce a (posterior) probability map
	-	The sum over all modalities is built to compute the speed image P(x) - P(not(x))
-	**Stage 2**: Active contours guided by speed function and user seeds


# Deep Interactive Object Selection (2016) - DIOS
## Note: First deep learning method (using CNNs) for interactive object segmentation

## Related Work
-	Authors list GraphCut, GrabCut, Weighted Geodesic Distances, Random Walk as alternative approaches
	-	Cons: All these require substantial user interactions to have a good estimation of the Fg, Bg distributions
	-	Authors claim their method is able to achieve this with a few clicks (even 2 are enough)
	-	They also claim that deep learning models can understand **objectness** and **semantics** better than classical approaches
	-	Authors claim this is the **first interactive object segmentation with deep learning**
-	Why deep learning?
	-	Classical approaches estimate the Fg, Bg distributions from low-level features
		-	Insufficient at distinguishing the Fg, Bg, when similar foreground and background appearances, complex textures, and bad lighting conditions'
			-	In these cases, the classical algorithms require extensive user interactions
		-	In contrast an FCN can learn **objectness** and **semantics**
			-	Which simplifies the interaction to a few clicks


## Method
-	Click-based interaction (positive and negative clicks)
-	Clicks are used to generate Euclidean Distance Maps, which are concatenated to the RGB images
-	For training, clicks are simulated
-	Segmentation model is based on FCN
-	Graph cut optimization is used to refine the output of the FCN
	-	Based on the probability maps from the FCN
-	Task is **binary segmentation**
-	Click = Foreground or Background
-	Euclidean Distance Maps are calculate the same way as with the Geodesic Distance Transform
	-	The pixel corresponds to the minimum Euclidean Distance to the Set of Mouse Clicks
-	![](../images/euclidean_maps.png)
-	Positive Click samples are generated by sampling the GT Pixels but:
	-	Only sample pixels which are at least d_step away from each other
	-	Only sample pixles which are at leats d_margin away from the boundary
-	Negative pixels are sampled from the complement of the GT pixels but at least a **d** distance away from the boundary
	-	Three sampling strategies are employed to simulate a user annotation
		-	n negative clicks randomly sampled from whole Bg space (again with d_step and d_margin)
		-	m negative clicks are sampled from negative objects (again with d_step and d_margin)
		-	p negative clicks to sample as close as possible to the positive object's boundary (+ some margin)
-	CRF is modelled as follows:
	-	Unary terms correspond to the cross-entropy loss
	-	Binary terms are the intensity difference / euclidean distance (no additional smoothness term)
-	Inference with User Interactions:
	-	User clicks and euclidean maps are calculated on the fly
	-	FCN produces probability map, which is solved by Graph Cut for the segmentation

## Experiments
-	Results show a clear improvement of the classical approaches
-	A comparison to FCN and CRF-RNN shows that interaction leads a better refinement
	-	The interactive method also allows **instance segmentation** when multiple intances of the same object are present
	-	Interactive method segments unseen classes (zero-shot learning) and generalizes better

# DeepLab (2016) - First CNN scores + CRF method (non-interactive)

## Motivation
-	Responses at the final layer of DCNNs are not sufficiently localized for accurate object segmentation.
	-	Caused by the invariance properties, which make DCNNs good for high-level tasks.
-	CNNs have a built-in invariance to local image transformations, which underpins their ability to learn hierarchical abstractions of data (Zeiler/Fergus 2014)
-	However, in semantic segmentation, we want precise localization and are dependent on low-level features
	-	Rather than abstraction of spatial details
-	Challenges for DCNNs for semantic segmentation
	-	Reduced feature resolution
	-	Existence of objects at multiple scales
	-	Reduced localization accuracy due to DCNN invariance


## Method
-	They show how to combine DCNNs with CRFs - cascade of two fairly well-established methods
-	They propose using **atrous convolutions** (dilated convs) to capture features in different scales
	-	Upsampling of the feature maps in last layers instead of max pooling
	-	Powerful alternative to the deconvolutional layers
	-	Larger field of view of filters without increasing the parameters
-	They propose an atrous spatial pyramid to segment objects at different scales
	-	Multiple parallel atrous convolutional layers with different sampling rates
		-	Parallel branches share their parameters
-	Pooling and Upsampling leads to the **checkerboard** pattern
	-	Atrous convolutions leads to a smoother output
-	Unary CRF potentials are set to negative-log-likelihoods of the DCNN predictions
-	For the binary potentials
	-	Potts model for the compatibility function
	-	Bilateral and Spatial filter for the pairwise potentials
-	![](../images/deeplab.png)
-	Training
	-	DCNN is trained first and fixed when training the CRF
	-	DCNN unary terms are fixed during training of CRF
		-	CRF "training" via mean-field-approximation iterations  
		-	Hyperparameters are found by cross-validation on 100 images with a coarse-to-fine search


# DeepMedic (2016)

## Related Work
-	Manual Annotation is prone to human error and results in significant intra- and inter-rater variability.

## Method
-	Architecture:
	-	Two streams - input at original and at a lower resolution to achieve a large receptive field for the final classification while keeping the computational cost low.
-	![](../images/deepmedic.png)
-	Fully-connected CRF as a post-processing step

# DeepCut (2017) - Extension of GrabCut (2004)

## Method
-	Train NN from bounding box annotations
-	Problem formulated as an energy minimization problem over a densely-connected CRF
-	Advantage of using bounding boxes vs. scribbles/pixels
	-	Allows to spatially contrain the problem
-	Why not use a classical approach (e.g. GrabCut)?
	-	We can assume that objects share common shape and appearance information which can be learned
	-	Instead of direct image-by-image object segmentation - re-use the learned features
	-	Particularly interesting for medical images, where an entire dataset is to be analysed for a specific organ or region
		-	Large class similarity in terms of shape and appearance
-	This paper has a good explanation of the history of improvements of GrabCut
-	CNN + CRF is used to obtain the final segmentations
-	NLL of the output probabiities constitute the unary term
-	Binary term is simply taken from the Krähenbühl paper
-	CNN in this paper is trained on **binary segmentation**
-	![](../images/deepcut.png)

# PolygonRNN (2017) 

## Method
-	Instead of predicting pixelwise labels, predict a polygon surrounding the object-of-interest
-	Input: Image crop
-	Output: Sequentially produce vertices
	-	Annotator can interfere at any time to correct the polygon
-	**Assumption**: Ground-truth bounding boxes are given
	-	Training data consists of cropped objects
	-	The CNN extracts a feature map which is fed to an RNN
	-	The RNN predicts a vertex, one at a time
		-	Input to the RNN is the feature map + the first, n-1, and n-2^th vertice maps
		-	Authors claim that the previous two vertices help the RNN follow a particular orientation
		-	The first vertex helpts the RNN to decide when to close the polygon cycle
-	![](../images/polygon-rnn.png)
-	CNN backbone is VGG-16
	-	Last FC layers and pool5 are removed
	-	Skip connections from low and high-level layers are concatenated to have multi-level features (edges, corners + semantic information)
-	RNN is a Convolutional LSTM model
	-	Conserves the spatial infromation received from the CNN
	-	Each vertex is outputed as a one-hot-encoding for a position on a DxD grid
	-	

# DeepIGeoS (2017) - Follow-up to Slic-Seg

## Method
-	CNN model **p**roposes segmentation (P-Net)
-	User adds **corrections**  to proposal and feeds both to R-Net (+ original image), which outputs the refined segmentation
		-	Combination through geodesic distance transforms 
-	Scribbles labels the pixes as background and foreground (binary segmentation)
-	![](../images/deepigeos_overview.png)
-	The geodesic distance helps to better differentiate neighboring pixels with different appearances, and improves label consistency in homogeneous regions.
-	Model runs in real-time and reduces number of user interactions, due to the already good features produced by the P- and R-Net
-	Given an image the Geodesic Distance Transform computes the distance map with the same resolution
	-	Requires a scribble to separate forground from background
	-	Image + Initial Segmentation + Distance Maps for fore- and background are stacked and fed to the R-Net
-	Resolution preservation through dilated convolutions instead of max-pooling in the P- and R-Net
	-	Different dilation help capture features in different scales
-	Each block has a fixed output channel size C
	-	Features from all blocks are forward-propagated and concatenated to have a combine feature vector with different scales (length 5*C)
-	CRF-Net is applied to the output of the CNNs (probabilities of each pixel belonging to background or foreground)
	-	CRF Net is extension of the CRF-RNN model
	-	User refinement context is also integrated to the output of R-Net
-	Unary potential is set to the **(inverse)** outputs of P- and R-Net
-	Sum of weighted Gaussians is substituted by a FCN (Pairwise-Net), (typically: Pairwise potential = compatibility * weighted kernels)
-	The fully-connected CRF cannot be computed for all pixel pairs, when the pairwise potentials are not weighted Gaussians
	-	DeepIGeoS uses patches with centers instead 
-	Cons: Requires 100k training samples for the **PairWise-Net**

# LooseCut (2017) - Extension of GrabCut

## Method
-	Previous models require a very tight bbox around the object of interest
-	LooseCut proposes how to tackle this with loose bounding boxes
-	New Markov Random Fields model
	-	Label consistency term to the GrabCut energy. Similar labels to similar-appearance pixels (even non-adjacent)
		-	Image pixels are first clustered using a superpixel algorithm that preserves both feature and spatial consistency
		-	Each cluster is assigned a label. The label consistency energy constitutes the number of pixels which differ in label from their cluster pixel
	-	Global Similarity Constraint - Explicitly emphasize appearance difference between Fg and Bg models
		-	Similarity between foreground and background Gaussian components is the reciprocal of the minimum distance between their means
		-	The similarity term sums up all these similiarities 
			-	The aim is: the closes background component should be as dissimiliar as possible to any foreground component
-	![](../images/loosecut.png)

# UI-Net (2017)

## Motivation
-	Semi-automatic methods are more suitable than fully-automatic ones, when only few data sets need to be processed for a highly accurate results
	-	e.g. single patient scan

## Method
-	Active user model (also called robot user)
	-	Fast and scalable method to simulate plausible user interactions
	-	The model can be learned by a database of interactions or defined by a rule-based system
		-	Authors of UI-Net opt for a rule-based
	-	Erosion on the boundary is used as a seed for foreground
	-	Dilation on the boundary is used as as seed for background
		-	Prevents initial seed placement too near to the boundary
	-	Samples from missclassified pixels are sampled and added with their GT labels to the initial seeds
		-	This is repeated iteratively
-	![](../images/ui-net-seeds.png)

# Suggestive Annotation (2017)

## Motivation
-	Different modalities, image settings, object in medical images ofter require creating a new set of training data
-	What instances should be annotated to attain the best peformance, given limited time?
	-	FCN + Active Learning
-	Annotating medical images is difficult
	-	Only trained biomedical experts can annotate data
	-	Biomedical images often contain much more object instances than natural scene images

## Method
-	Utilize uncertainty and similarity information from the FCN
	-	Formulate a maximum set cover problem to determine the most representatitve uncertain areas	
-	![](../images/suggestive-annotation.png)
-	Active learning through uncertainty and similarity
	-	Uncertainty with **Bootstrapping**
		-	Train a set of FCNs on different subsets of the training data (sampling with replacement)
		-	Uncertainty is calculated using the variance (disagreement) among these models
		-	Uncertainty is highly correlated with the testing errors
		-	Image uncertainty is aggregated through the mean-pixel uncertainty
	-	![](../images/suggestive-annotation-uncertainty.png)
	-	Similarity is simply the cosine similarity between the FCN-embeddings of the images
	-	First top-K uncertain images are sampled from the unannotated set
		-	Then the top-N images with highest similarity to the whole annotated set are sampled
	-	No human-in-the-loop --> fully automatized


# Deep GrabCut (2017) - Extension of GrabCut and DIOS

## Related Work
-	GrabCut, BoxPrior do not work when the bounding box is inside the object (does not cover it completely)
	-	Assume that the entire object is within the bbox
	-	Especially large problem when detections come from an algorithm/model
-	GrabCut, OneCut and BBox priors only rely on basic color and edge information and do not use higher-order knowledge
-	Inspired by DIOS
	-	DIOS requires post-processing with graph cut
	-	Deep GrabCut works also for **multi-object** segmentation

## Method
-	Tranform **rectangles** into Euclidean distance maps
-	Multi-object segmentation (zero-shot)
	-	Detections may overlap 
		-	Apply dense CRF to convert individual segments into instance-level semantic labels
-	![](../images/deep-grabcut.png)
-	Training
	-	Train set is composed of (image, rectangle) pairs
	-	Training rectangles are sampled by applying random jitter to the ground-truth bounding box
	-	The Euclidean distance map is concatenated to the image as a 4-channel input
	-	Encoder FCN is with the VGG-16 backbone
		-	Trained end-to end on semantic segmentation
	-	Trained semantic segmentation model is then extended to instance segmentation using CRFs
		-	Non-max supression for all bbox candidates
		-	The confidence of each bbox is used to multiply the pixels underneath it to obtain new probability values
		-	Probability of background is the multiplied probabilities of all the bounding boxes
		-	CRF can be applied directly to these probabilities
-	![](../images/deep-grabcut-eval.png)
-	Pro: Works quite well for bad bounding boxes in comparison to traditional non-DNN approaches
-	Pro: Deep GrabCut can also generalize to scribbles (but only closed curves)

# Regional IIS (RIS-Net) (2017)

## Related work
-	Graph Cut, Random Walker, Geodesics, Geodesics + Graph Cut, Growcut all rely on low-level cues
	-	Difficult to predict fg/bg when similar colors, textures 
-	Snakes and Intelligent Scissors mainly consider boundary properties, thus performing poorly at weak edges
-	DIOS has a higher level understanding of objectness and semantics
	-	Often struggles to refine its predictions given additional inputs

## Method
-	Expands the field-of-view of the input to capture the local regional information surrounding them
	-	Stronger ability to refine local segmentations (than DIOS)
-	Adopts multiscale global contextual information to augment each local region for improving feature representation
-	![](../images/RIS-net.png)
-	Training:
	-	Input is the image with the Euclidean maps of the user clicks
	-	FCN is used with DeepLab (VGG-16) backbone
		-	These make up the Global branch: coarse aggregation 
	-	Sampling ROI proposals
		-	A ROI is sampled by iterating over all positive clicks
			-	The nearest negative click makes up the whole region
			-	Additional 2 ROIs are sampled by interpolating between the rest of the 3 ROIs (sampled by clicks)
	-	Global context
		-	The conv7 layer of the Global branch is used with filters with different scales to obtain multi-scale global features
			-	The global features are then concatenated to each region embedding
	-	Click discount
		-	Inspired by RL
		-	The ROI with the worst segmentation error is processed first in the loss
		-	The following ROIs are then discounted exponentially (worst ROI has the largest penalty in the loss)
		-	This forces the model to reduce the number of needed ROIs
	-	Local-Global Fusion
		-	Each local segmentation prediction is padded with zeros and then all patches are fused with max-pooling



# PolygonRNN++ (2018) - extension of PolygonRNN 

## Method
-	New CNN backbone
	-	Based on Resnet50 with atrous convolutions 
	-	Skip features are computed with the lower-level features
-	Use Reinforcement Learning to train the model
	-	RNN is viewed as a sequential decison making agent
	-	Reward is the IoU between the mask enclosed by the generated polygon and the GT mask
	-	Use the REINFORCE trick to compute the gradient over the expected (negative) reward
-	Increase output resolution of the polygon with a GNN (Graph NN)
	-	From 28x28 to 112x112
	-	GCN takes the predicted vertices and builds an input graph
	-	Nodes in the center of each edge are added and a message-passing framework with a GRU cell is trained
	-	Targets of the GNN are the relative displacements of each RNN-predicted vertex w.r.t. GT vertices
-	Model performs quite well on **cross-domain** tasks
-	Authors argue that the RNN incorporates the learning of a **shape prior** - helps find boundaries in shadows, low saturated areas, noise etc.
-	RNN is redisigned to incorporate visual attention at each time step
-	Separate evaluation network 
	-	Input: Skip Features + Predicted Polygon
	-	Output: IoU with GT (tries to predict the performance)
-	![](../images/polygon-rnn++.png)

# Guide Me (2018)

## Method
-	Guide a visual-learning CNN with user text input **online** *(also mouse clicks)*
	-	Add a layer that acts as a spatio-semantic guide into the network
		-	Either as a energy minimization scheme
		-	Or a recurrent model which translates human language queries to interaction weights
-	Language-based corrections instead of scribbles
-	The authors claim that online learning is important in medical image analysis, where the experience of trained practitioners matters and segmentation models are not always robust
	-	Another relevant example is speeding up labor-intensive and repetitive labeling tasks, such as creating datasets for semantic segmentation
-	Interactions update the network at **test time**
-	Closely related to VQA
	-	Question posed as text
	-	Answer is based on the image context
	-	However, this method does not output a textual answer (caption, answer), but an **image**
-	The approach can be applied to any **fixed CNN** - CNN-model agnostic
-	The authors also propose how to guide the CNN with **clicks**
-	Guiding block adjusts activation maps of the CNN
	-	Guiding vector multiplies each channel of the CNN bottleneck feature tensor
	-	And two separate guiding vectors for the width and height
-	![](../images/guide_me.png)
## Training procedure
-	Pre-trained semantic segmentation model (e.g. FCN) outputs prediction
-	Hint generator takes prediction + GT and generates a hint
	-	Hint consists of **class** + **location**, e.g. cat is missing on the right
-	The Guide (GRU-cell) takes the textual input and produces alpha, beta, and gamma to modulate the segmentation network
-	The segmentation model then adjusts its prediction and a pixel-wise cross entropy is applied
-	The guide learns to process textual cues, which **improve** the segmentation results
-	The hint generator is a form of **simulation of textual input**, which can be real-user input during test time

## Alternative training
-	Guide can be just the alpha, beta, and gamma vectors (not a network)
-	Parameters can be estimated by applying the loss only to masked regions 
	-	Mask corresponds to pixels which are pointed by the hint
	-	A masks could be generated by labeling pixels with a mouse during inference
-	For training, to simulate mouse clicks:
	-	Only the pixel with the smallest posterior probability difference between the two most confident classes is taken, since it has the highest interclass uncertainty.
	-	This is repeated for multiple pixels (0, 1, 5, 10, 15, 20) (which make up the mask)



# BIFSeg (2018) ---> extension of DeepIGeoS for unseen objects

BIF - **B**ounding Box and **I**mage-Specific **F**ine-Tuning
## Related Work
-	They list DeepCut, DeepIGeoS, Deep Interactive Object Selection
	-	One challenge when using CNNs is the requirement of large amounts of annotated images for training
	-	CNNs also do not generalize well to unseen objects
	-	Authors discourage using DeepMedic or HighRes3DNet as they have a high memory requirement
	-	DeepIGeoS 
		-	Good interactivity
		-	Lack of adaptability to uneen image contexts
## Method
-	Bounding box to extract foreground from background
-	Image-specific fine-tuning on **test** data
	-	Either with or without the user interactions (scribbles)
	-	**Contribution**: First method to train on a dataset and **adapt/fine-tune** on individual test-images (in contrast: GrabCut creates a new model for each test image)
-	Low memory requirements for fast inference
-	Con: Only binary segmentation
-	Training:
	-	GT bbox is used to crop the target objects and a network is trained on **binary segmentation**
		-	Training on multiple different organs ---> not organ specific!
		-	This way the CNN learns common features
			-	Saliency, contrast, hyperintensity across different objects, which helps to generalize to unseen objects
	-	Training dataset with multiple labels is converted to a binary classification dataset (with more sample as there could be multiple object instances in one image)
	-	Training dataset is also pre-processed to only have cropped regions (+- 10 pixels / voxels margin)

-	Testing:
	-	User provides bounding box to crop a region
	-	CNN outputs initial segmentation
		-	User can additionally provide a scribble to correct
			-	Scribbles can correspond to fore- or background for a correction
			-	Loss function applied to scribbles is similar to GrabCut (2004, GMMs)
				-	Solved in two steps:
					-	First fix network parameters and update the label -> Results in a CRF solved by Graph Cuts 
					-	Then fix label and update weights --> simple CE-Loss
				-	Weight in CE-Loss is larger for scribbles, since they should have a higher impact (higher update)
		-	CNN updates its weights on a single image and refines the result at the same time
			-	Update only on the final classification layers (block 6)
	-	CNN can be applied to unseen objects! **(zero-shot learning)**
-	Architecture
	-	They use the proposed P-Net model from DeepIGeoS
	-	They also show that P-Net generalizes better than FCN and U-Net for unseen objects/organs
-	![](../images/bifseg.png)

## Results
-	They compare in 2D with
	-	GrabCut, FCN, U-Net, Slic-Seg, Random Walks
-	In 3D with
	-	3D Grab Cut, DeepMedic, HighRes3DNet, GeoS, GrowCut
	-	They use T1c for training and evaluate on T1c and FLAIR images 


# DEXTR (2018)
-	![](../images/dextr.png)
##  Method
-	Uses extreme points (left, right, top bottom) and creates a heatmap which is concatenated to the RGB input of the segmentation model
-	During inference, the user can input these points manually for known objects
-	During training
	-	The image is cropped with the implicit bbox from the extreme points
	-	The gaussian map is concatenated to the RGB image and the network does a forward pass
	-	The ResNet-101 backbone is adopted, but the max pooling and FC layers are replaced with atrous convolutions
	-	Last layer is replaced with Pyramid Scene Parsing Module to aggregate global context
-	To simulate interactive segmentation
	-	They train a model, where each image has 4 labeled extreme points
	-	If the classification IoU lower than 0.8, they label a 5th point and use it again during training 
-	Network inference for a single object takes only 80ms (fully interactive)
-	Binary classification (Fg/Bg)
## Related Work
-	[25] has already done this, but with GrabCut, not a DNN

## Results
-	They show that clicking is better than bounding box annotations on PASCAL, COCO and Grabcut

# Inter-CNN (2018)

## Related Work
-	GrabCut, GeoS, Random Walker
	-	Require a large number of interactions
-	UI-Net, Iteratively trained IIS, and BIFSeg have shown better results than traditional methods
-	DeepIGeoS does not take into account the fact that a user may be interacting with the tool over several iterations
	-	Each time providing scribbles based on the results of the last update
	-	Model is also trained on only one interaction (scribbles) without multiple steps

## Motivation
-	The need for interaction increases when there is a domain shift (e.g. scanner differences)

## Method
-	CNN-agnostic scheme which is trained on **simulated** user interactions
-	**Multi-label** segmentation
-	![](../images/interCNN.png)
-	Inter-CNN is not trained on a **single** interaction but on a fixed number of **K** consecutive interactions (simulated by a *robot*)
	-	Scribbles are provided based on the discrepancy between GT and current predictions
	-	Input of Inter-CNN is not only the image, but also the (Image + Prediction + Scribble Mask)
	-	K is set to 20 in the paper
-	U-Net is used for both the automatic and refinement models
-	Simulating the user (Robot User):
	-	First the set of missegmented pixels is identified
	-	Then a pixel from this set is chosen randomly for each class separately
	-	We look up the class of this pixel and:
		-	All the pixel with the same GT class as this pixel in the 9x9 region around it are marked as new scribbles
		-	This is done for each class
## Results
-	Compared to UI-Net and BIFSeg
	-	Much faster convergence than BIFSeg, similar results to UI-Net
	-	Much simpler concept than BIFSeg (personal opinion)

# Latent Diversity (2018)

## Related Work
-	DIOS relies heavily on post-processing with Graph Cut
	-	Latent Diversity only needs a forward pass

## Method
-	Two networks
	-	First one synthesizes a diverse set of plausible segmentations that conform to the user's input
	-	The second one is trained to select among these
-	User input as clicks
-	Binary Segmentation
	-	Confidence thresholded at 0.5 for hard segmentation
-	![](../images/latent-diversity.png)
-	The authors claim that the segmentation network can fully commit to one of these semgentations instead of "splitting the difference"
	-	This would result in partial segmentations, because the network would have to "meet them in the middle"
	-	Adopting a network which chooses one segmentation alleviates this problem
-	Workflow
	-	Distance transforms are the same euclidean ones from DIOS
	-	VGG-19 is used as a feature extractor
	-	Segmentation network outputs a feature with M channels - for M segmentation proposals
		-	Sigmoid with 0.5 threshold to output segmentations
	-	Selection network is trained with cross-entropy on the proposed masks with the largest IoU



# IFSeg (2019)

## Related Work
-	Manual approaches are tedious and have a inter- and intra-observer variability
-	Fully-automatic methods are sensible to domain shifts 
	-	Not easy to recover from these domain shifts, there is no mechanism of user interaction to correct mistakes
	-	Hard to generalize to unseen structures
-	Traditional approaches work well when regions are homogoenous, but because they rely on low-level features, they fail in more difficult cases

## Method
-	2D CNN trained on limited training data
-	User provides Fg and Bg clicks and Gaussian Heatmaps are generated in the shape of the image
	-	Concatenated with the image and fed to the U-Net model
	-	Very similar to DEXTR, but the points do not need to be extreme
-	Model is trained on binary segmentation for multiple organs
	-	Authors claim that any region that shows a distinctive pattern or some form of spatial coherence can be segmented this way (zero-shot)


# Two-stream FCN (2019) - extension of DIOS

## Motivation
-	The fusion network reduces the number of layers between the user-interaction features and the output
	-	The output is influenced more by the user's intent

## Related Work
-	Traditional algorithms (GraphCut, GrabCut, RandomWalk, Geodesic cut) only utilize low-level features and hence cannot deal with:
	-	Similar colors of fore- and background 
	-	Inhomogenous foreground intensities
	-	Require a lot of user interaction
-	Most DNN-based interactive segmentation methods use **early fusion** to incorporate the user interactions
	-	An early fusion of the user interactions might weaken the influence of these interactions
	-	Especially considering sota DNNs consist of a large number of parameters

## Method
-	Late fusion, as opposed to all the early fusion related work
-	Two parts
	-	Two-stream late fusion network (TSLFN)
		-	One stream for the image features
		-	One stream fo the user interactions
	-	Multi-scale refining network (MSRN)
		-	Fuses features from different layers of TSLFN, hence different scales
			-	Local and global features
-	![](../images/two-stream.png)

## Architecture
-	Base network is based on VGG16
-	Authors claim they have tried slow-fusion, but it did not work well
	-	Slow-fusion is originally designed for RGB + Depth, where the depth information already includes boundary information
	-	Whereas user interaction-transoforms do not include such fine-grained information
-	![](../images/two-stream-arch.png)


# UGIR (2020)

## Related Work
-	List BIFSeg, DeepIGeoS and other method (2019)
	-	Better than traditional interactive methods
	-	Require the user to manually go through the slices to inspect where the models have failed (would benefit from active learning)
-	Active Learning paper (2011) shows that this is a beneficial topic to explore
-	Uncertainty estimation
	-	MC-Dropout, Model Ensemble, Test-Time augmentations 
		-	Require multiple forward passes
	-	Bayesian Network
		-	Not yet investigated for interactive refinement

## Method
-	Grouped-Convolution CNN to obtain multiple predictions
-	Uncertainty estimation with a single forward pass
	-	User can guide the model on the subset of slices with the highest prediction uncertainty
-	The results are refined using an interactive level-set method
	-	Using the initial segmentation and user predictions
-	![](../images/ugir.png)

-	Multi-Group CNN (MG-CNN)
	-	Modified U-Net 
	-	Input is split into N groups 
		-	Each channel is convolved independently -> N independent feature maps are obtained
		-	Predictions are obtained by N-softmax layers (i.e. group-wise softmax)
		-	Model can be seen as an ensemble of N parallel sub-networks
		-	Bottleneck feature merges the groups to "allow the communication of these N sub-networks"
-	![](../images/mg-net.png)
-	Uncertainty of the image is the sum of uncertainty pixels, normalized by the area of the segmented region
-	Refinement is done with an Interactive Distance Regularized Level Set Evolution (I-DRLSE)
	-	Compared to using Graph Cut, or R-Net
## Results
-	Uncertainty estimations correlated with mis-segmentations
-	Similar performance as DeepIGeoS, but twice as fast

# IteR-MRL (2020)

## Motivation
-	The dynamic process for successive interactions is largely ignored when incorporating user hints

## Related Work
-	Automatic segmentation models are not accurate not robust enough for medical applications
-	IterCNN and DeepIGeoS incorporate a first **automatic** stage with a **coarse** segmentation 
	-	Followed by an updating/refinement **interactive** stage with user-inputs
	-	Authors refer to these methods as **update methods**
-	Problems with update methods (InterCNN, DeepIGeoS):
	-	Each refinement step is isolated from the rest - absense of previous information
	-	No uncertainty prediction
-	![](../images/update-method.png)
-	Other RL-based
	-	SeedNet - agent simulates a user giving hints
	-	Polygon-RNN++
		-	Not applicable to 3D because of the polygons and the extremely large actions space, even with the meshing strategy
## Method
-	Model the dynamic process of iterative image segmentation as a Markov Decision Process
	-	Solved by Reinforcement Learning
	-	Each voxel is treated as an agent with a shared voxel-level behaviour strategy so that it can be solved with multi-agent RL
	-	The prediction uncertainty is reserved in the state space of MDP
		-	Adjustment action space for a segmentation refinement
-	At each refinement step the model needs to decide on the labels on all voxels
	-	Given a predicted segmentation map *and* user input
	-	After that the model gets feedback on the segmentation 
		-	This is repeated until the max. number of iterations ensues
-	MDP is solved with RL
	-	Maximization of the reward at each refinement step by finding the best segmentation strategy
-	Each voxel is an agent which decides on its label
	-	The agents share their policy and collaborate through convolutional kernels
	-	Intead of a loss function
		-	Cross-entropy-based reward to prompt more exploration
		-	Positive reward at an improvement of the segmentation
		-	The reward is **relative** to the previous step --> steps are no longer **isolated**
-	Uncertainty is incorporated in the RL-state
	-	The task is to find the series of actions to adjust the segmentation probability with a certain level
-	Initial segmentation can be obtained using any accessible segmentation method
	-	Then the coarse initial segmentation is updated iteratively by their method
-	![](../images/iter-mrl.png)
-	Methodology step-by-step:
	-	And FCN is used as an actor network
	-	The state of an actor (voxel) is
		-	The voxel grayscale value
		-	The previous predicted probability by the FCN
		-	The user inputs for this voxel (positive or negative)
	-	Hint maps are generated by the geodesic distance to the hints (user clicks)
	-	Actions of an agent are a scalar a_i^(t) which adjusts the previous probability of the voxel belonging to the object
		-	Actions are added to the previous probabilities
	-	Reward is the Cross-Entropy at step (t-1) minus the CE at step t, i.e. relative improvement from previous step
	-	Actor network adopts the R-Net structure from DeepIGeoS
-	Actor R-Net
	-	Policy head predicts probability of each action, given the state
	-	Value head estimates how good the current combination of the image, hint-map and previous segmentation is
	-	Input is the state (img+prob+hints)
## Results
-	RL is better than Min-Cut, DeepIGeoS, InterCNN given any initial segmentation (without clicks)
-	Increasing the number of clicks shows a large improvement for IteR-MRL
	-	Due to the relation gain between successive predictions	

# MIDeepSeg (2021)

## Motivation
-	Automatic DNN methods do not have a high enough accuracy and robustness.
-	Interactive frameworks usually improves this, but previous approaches still require a lot of user input. 
	-	Too much input and the model collapses to a manual annotation
-	Challenges in medical images
	-	Low contrast, different imaging and segmentation protocols, variation among patients
	-	Low-level features are not enough to effectively distinguish the object from the background in many situations with low contrast
-	The geodesic distance transform is spatially smooth and contrast-sensitive to encode user interactions.

## Related Work
-	Deep Interactive Object Selection, DEXTR and Two-Stream4Med lack evaluation on medical images with low contrast and ambiguous boundaries and are not adapted for 3D.
-	PolygonRNN and PolygonRNN++'s ability to deal with objects with complex shapes and 3D medical images is limited.
-	DeepIGeoS, IF-Seg, DeepCut and BIFSeg are specially designed to segment medical images
	-	DeepCut and Weakly-Supervised 3D method were designed for training over a large dataset rather than interactively editing a single segmentation result at test time
	-	DeepIGeoS lacks adaptability to unseen objects
		-	It is time-consuming to find an appropriate threshold value for the geodesic distance map 
	-	IFSeg is easy to use, but it has only been tested on **one** unseen structure to test its generalization
	-	BIFSeg is limited to only a few unseen objects in the same image modality and context
		-	It is also time-consuming to fine-tune it on each test image
	-	DEXTR
		-	Extreme points are not enough to capture the main shape of irregular and concave shapes
	-	DeepLab, DeepMedic
		-	Use CRF, but not designed for **interactive** segmentation

## Method
-	CNN guided with only a few clicks of user interactions
	-	Interior margin points
	-	Points inside the object and close to the boundary
	-	EGD transform can compute the saliency map of the object
-	![](../images/egd.png)
-	Interactions are encoded via the Exponentialized Geodesic Distance Transform (EGD)
	-	Context aware and parameter-free
	-	Combination of Geodesic distance transform and Exponential Transform
	-	Just an exponential of the (negative) geodesic distance
		-	Small geodesic distances to the margin points lead to a high value in the cue map (white values)
-	Information fusion, which uses additional clicks to refine the segmentation
	-	Followed by Graph Cut 
-	![](../images/MIDeepSeg.png)

-	The user provided points are used to infer a relaxed bounding box to crop the input image
-	A cue map is computed based on the points and the EGD
-	Training
	-	Interior margin point generation
		-	3-4 points close to the EXTREME points of the object 
			-	make sure the object is inside the implicit bounding box
		-	n more points (0-5) along the boundary of the object mask
		-	Implicit bounding box is relaxed to contain some background context information
-	CNN can be any model - they use 2D/3D U-Net to show the proof-of-concept
	-	CNN is trained on Image + Cue Map concatenated along the channels
-	Refinement stage
	-	User provides clicks for mis-segemented background and foreground 
	-	New margin points are added to the initial points
		-	New EGD cue maps are generated for Bg and Fg and fused with the probabilities of the Fg, Bg of the CNN
		-	The probability maps are refined by the new cue maps with a linear interpolation
			-	Interpolation factor depends on the minimum distance to the provided clicks
		-	EGD can be considered as a probabilistic map (bound to [0,1]), which can seemlessly be integrated into a CRF framework
-	![](../images/edg-refinement.png)
-	The refined probability maps are fed to a CRF 
	-	Unary term is NLL of refined maps
	-	Binary term is typical CRF term (image intensity vs. euclidean distance)
-	Refined segmentation can be achieved via Graph Cut

## Results
-	Comparison to other distance transform to show the EGD is the best
-	Comparison to with and without user interactions
-	Comparison to Graph Cuts, Randow Walks, SlicSeg, DeepIGeoS, DIOS, DeepGrabCut, DEXTR for 2D
-	Comparison to ITK-Snap, 3D Graph Cutsm DeepIGeoS, DIOS, DEXTR for 3D
	-	DIOS - Deep Interaction Object Selection




# Simple CRF + Geodesic Distance (2022)
- Paper claims it is the first fully-connected CRF for interactive medical image segmentation. 

## Goal
-	Convert image segmentation into a graph optimization using Fully-Connected Conditional Random Field

## Related work
-	Recommends which slice to annotate next (similar to active learning)
-	No need for parameter tuning for each application (imaging type + objects of interest)

-	Manual annotation tools
	-	Line tracing
	-	Polynomial curve fitting
	-	Area painting
	-	ITK-Snap
	-	3D-Slicer
		-	Con of ITK-Snap and 3D-Slicer is the tremendous time for annotation

-	Semi-automatic
	-	Allow user interaction to improve labels
	-	Iterative improvement by adding scribbles - grow cuts and graph cuts

-	Types of user annotation
	-	Point-based
	-	Contour-based
	-	**Scribble-based**
	-	Bbox-based
-	Comment of Graph Cut methods
	-	GC and GrabCut work well, but only for binary segmentation and are slow for 3D
-	Comment on CNN approaches
	-	The main drawback of these DCNN-based methods is the requirement of a training process using either weakly or fully labeled data.

## Method
-	Problem is formulated as Maximum A-Posteriori (MAP) inference in a CRF
-	Each class has its own energy function - the energy must be minimized to solve which pixel is assinged to a certain class
-	The pairwise energy is lower for pixels which are 
	-	Similar in appearance
	-	Closer to each other (smoothness)
-	The unary **class** energy is lower for pixels which are:
	-	Similar in appearance to mean class intensity
	-	Closer to the nearest annotated pixel of the class (using geodesic distance)
-	Uncertainty of slice through entropy 
-	Annotation is propagated to 3D volumes trought the labels of few slices (smoothness term)
-	Pro: 
	-	Does not require accurate contour tracing which needs lower concentration levels from the annotator
-	Cons:
	-	Not particulary suitable for blood vessel segmentation
	-	Or any long and thin linear structures
-	![](../images/crf-geodesic.png)





		

























