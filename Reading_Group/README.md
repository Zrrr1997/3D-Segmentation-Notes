# Reading Group 04.05.2022

| Paper | Year | Conference | Guidance | Backbone | Train data | Robot user | Active Learning | Evaluation Metrics | Online Learning | Weakly Supervised | Main Keypoints | Comparisons | Loss Function | Iterative Training | Attention | Cropped Input | Multilabel | 3D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ECONet | 2022 | MIDL | no | 3D-Conv + 3 FC Layers | only user-annotated scribbles from UESTC-Covid19 (online) | Largest erroneous volume - N random samples within (only during evaluation) | no | DICE, ASSD, #Scribbles to reach DICE, Online inference time  | yes  | yes (incomplete labels)  | Lightweight-Online learning with 3D-DNN. | DybaORF, GrabCut, GC | Class-Fraction-Weighted BCE | yes (online) | no | yes | no | yes |
| FocalClick | 2022 | CVPR | Gaussian Heatmaps | HR-Net or SegFormer | SBD or COCO + LVIS | Train: Random bbox around objects + (see iterative), Test: largest erroneous area | no | NoC@IoU, NoF@IoU | no | no | Only clicked RoI is updated - local changes. Coarse segmentation + Local refinement. | RITM, EdgeFlow, f-BRS, 99%-Acc, FCA-Net, CDNet, BRS, CM-guidance, RIS-Net, LD, DIOS, Geo-Star, Eucl-Star, RW, GeoMat, GC | Binary CE + Normalized Focal Loss | Center of largest erroneous region (+ erosion beforehand) | no | yes | no | no |

