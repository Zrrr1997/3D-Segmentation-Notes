# CT (Computed Tomography)
## History
Found by Sir Godfrey Hounsfiel, Electrical Engineer. 

## Summary of Workflow
Multiple images in succession of a different layers of the body. Organs and tissues of the body attenuate the X-Rays to difference degrees. X-Rays pass through the body, are attenuated, and are detected through sensors, which quantisize them into discrete brightness values. Amount of transmitted radiation = Brightness. Specific density values correspond to brightness steps. 

**Summation of shadows**: Density of overlapping regions sum up and contribute to a higher HU value. 

**Silhouette sign**: Edges of an object are seen if interfece is with a different density (if edge of between heart and lung is fuzzy, it might indicate pneumonia).

## Comparison to X-Ray
-	In X-Rays, orthogonal views are needed to localize objects (e.g. coronal + lateral view). 
-	CT-Scans have a higher resolution, due to multiple sensors. CT can differentiate between tissues that differ in radiographic density of 1% or less.
-	CT-Scans can be projected to an arbitrary plane, creating any view of the subject.

## Density Metric
Density is measured with the Hounsfield Unit (HU). Water = 0 HU, Air = -1000 HU. The higher the HU, and hence the density, the higher the brightness in the image.
![](../images/HU.png)

## Contrastive medium
Patients might have to administer a contrast medium to make it easier to differentiate between different kinds of tissues. It drastically increases the stomach's contrast (appears brighter). Can also be administered intravenously to contrast adjacent regions.
-	High level of iodine, absorbed by x-rays (appears white)
-	Barium Sulphate for oral contrast (small and large bowel, and stomach)

## Window levels
Change quantization of the brightness levels (JND is around 2%) so that structures with very close density levels can be contrasted.

## Radiation dose
Relatively low - average natural radiation per year. A larger radiation might incur if the resolution of the CT scan is increased.

## Application
-	Tumor detection
-	Visualize blood vessels and detect deposits
-	Pulmonary embolisms
-	Pneumonia
-	...

# PET Scans

## Summary of Workflow
Reveals the **metabolic** and **biochemical** function of the tissues and organs. The PET scan uses a radioactive drug (tracer) to show both the normal and abnormal metabolic activity. PET scans can often detect the abnormal metabolism of the tracer in diseases before the disease shows up on CT or MRI scans. 

The tracer clusters into areas of the body with higher levels of metabolic activity and pinpoints the location of the disease. PET images are usually combined with MRI or CT.

Radiating photons (gamma-radiation) is captured with the sensors for the acquisition of a 3D image.

## Use Cases
Physiological processes lead to anatomical changes in the body. Hence, PET can provice molecular-level information long before any anatomical changes have occured.

## MRI Scans
## Ultrasound
## X-Ray
## OCT
## EHR
## Microscopy
## Pathology







