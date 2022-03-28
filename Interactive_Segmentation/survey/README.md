# Survey (2020)

-	Interactive Segmentation: 
	-	The task of extracting an image region or object of interest from the background using **prior knowledge** provided by user interaction.
		-	Points, scribbles, bounding box
		-	Input provides constraints on size, color, location to guide the segmentation
	-	Pros:
		-	Reduces runtime
		-	Improves quality
	-	Classify based on
		-	User interaction type
			-	Seed based - scribble, line, point
			-	ROI based - bbox, lasso
				-	Problem: Accuracy is related to how tight the bbox is delimited (a tighter bbox is better)
			-	Active, passive - help during inference
		-	Methodology
			-	Contour
			-	Graph Cut
			-	Random Walk
			-	Region Merging / Region Growing
			-	CNN-based
		-	Processing level
			-	Pixel-wise
			-	Superpixel-wise
			-	Hybrid
-	![](../../images/user-inputs.png)


## Contour Methods
The main principle of contour-based methods is to
extract object contours using **edge features** and prior
knowledge provided by the initial user interaction.
-	Intelligent Scissors (1998) - implementation of Live wire (1992)
	-	Used in GIMP
	-	Contour is extracted using the shortest path linking the seed pointy using Djikstra's algorithm
-	Riverbed (2012)
	-	IIS based on boundary seeds which requires fewer user interactions 
	-	Uses an optimum boundary tracking process - Image Foresting Transform

-	Interactive Convex Active Contours
	-	Input: Scribbles + Output of another IIS (e.g. Random Walk)
	-	Method: Energy equation of the convex active contours with (scribbles + probability map from random walk)
	-	Output: Minimize energy with split Bregman method
-	Snakes
	-	Conforms to image edges
	-	Is a smooth varying curve vs. lot of jagged pieces
	-	Snakes: parametric **closed** curve (x(s), y(s)) s \in [0,1]
	-	Energy function matches the intuition about what a good curve is
		-	Curve will iteratively **evolve** minimizing the energy function at each step
		-	Energy decomposed in internal and external part 
			-	Internal depends only on the shape
				-	Integral over the first and second derivatives of the curve
				-	First derivative is the **elasticity** - keeps points together
				-	Second derivative means the curve is not too "bendy" - no shard edges / oscillations (smoothness term)
			-	External depends on the image intensities
				-	Final curve should lie on top of strong edges
				-	Integral over the (negative) gradients (both x and y) along the points on the curve
					-	Negative as we want to have larger gradient magnitudes 
				-	Problem: Edges might not be strong enough --> Apply Gaussian smoothing to enhance the edges 	
					-	Or Gradient Vector Flow - don't use image gradient but create a **gradient vector field** over the image plane
					-	No zero-values in plateaus -> values point toward stronger edges 
	-	**Level sets**
		-	Define level-set function phi(x,y) over whole image and evolve the function instead of the curve
		-	phi(x, y) = 0 defines the point in the image of the object of interest
		-	Isosurface algorithm to extract the final segmentation
**Cons**:
	-	Computationally expensive - not *really* interactive
	-	Can get stuck in local minima
	-	Cannot effectively handle intensity inhomogeneity (especially important in medical images)
	-	Graph cuts make it possible to have an **interactive rate**
	-	Initial contour must be fairly accurate - requires user concentration
	-	Snakes can never wrap around multiple objects
	-	Snakes can never get inner boundaries (donuts)
		-	Both solved with level sets
## Graph Cut Methods
-	Conditional Random Field by optimizing the min-cut/max-flow algorithm (Boykov)
-	Unary term can be derived from
	-	Intensity Histogram (GraphCut)
	-	GMMs (GrabCut)
	-	
-	







