# Reading Group 04.05.2022

| Paper | Year | Conference | Guidance | Backbone | Train data | Robot user | Active Learning | Evaluation Metrics | Online Learning | Weakly Supervised | Main Keypoints | Comparisons | Loss Function | Iterative Training | Attention | Cropped Input |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FocalClick | 2022 | CVPR | Gaussian Heatmaps | HR-Net or SegFormer | SBD or COCO + LVIS | Train: Random bbox around objects + (see iterative), Test: largest erroneous area | no | NoC@IoU, NoF@IoU | no | no | Only clicked RoI is updated - local changes.| RITM, EdgeFlow, f-BRS, 99%-Acc, FCA-Net, CDNet, BRS, CM-guidance, RIS-Net, LD, DIOS, Geo-Star, Eucl-Star, RW, GeoMat, GC | Binary CE + Normalized Focal Loss | Center of largest erroneous region (+ erosion beforehand) | no | yes |
