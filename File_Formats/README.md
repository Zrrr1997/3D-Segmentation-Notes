# File Formats for Medical Storage

## DICOM
-	Loading time is x100 as high as Nifti or NRRD
-	Preserves all metadata and UIDs

## Nifti and NRRD
-	Compression leads to x2 less storage but x5-x10 loading time
-	Preserves essential metadata 
	-	Image geometry 
-	Suitable for single cohorts 



## NPY (less common)
-	No metadata
-	Supports arbitrary dimensions and non-image vector data
