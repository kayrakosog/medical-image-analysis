# Designing a CNN classifier for colon tissue image classification by transfer learning using the pretrained AlexNet network.

Pre-Processing and Preparation of the Data

Bias Field and Noises:
The artifacts are very common in medical imaging and are referred to as “Bias Fields”. To correct intensity bias (non-homogeneity) recognized in the MRI image in this study, one of the adopted methods was Z-score-based intensity normalization and another one was applying the histogram-based nonparametric nonuniformity normalization (N4) algorithm. In addition to bias correction, adaptive non-local means denoising was used to get a higher signal-to-noise ratio in pre-processing step. Another solution is to handle false-positive predictions introduced by imaging artifacts in the post-processing part. A dense CRF refinement was applied after the last soft-max layer to improve segmentation performance.

Small Sized Data:
One approach was augmenting the data by randomly flipping the images in left/right-up/down direction and adding random gaussian noises. The resulting images were then treated as new inputs and were given to the model. In the training part, one of the top-ranked contestants proposed using transfer learning to overcome limited data issues.	

Class Imbalance:
One method that I came across was oversampling the positive class slices (images) to balance the positive/negative ratio. Another discussed issue was the domination of the background voxels and a probable shift in the model’s prediction tendency towards them. Usage of a weighted loss function was not suggested as it is likely to increase false-positive predictions. Instead, the offered method was non-uniform patch extraction from each subject. It was stated that the foreground voxels show high variance and harder to segment thus more patches should be extracted from the foreground. To make sure that each patch lies within the image, a probability of being a valid patch center is calculated and then the corresponding patch was extracted by looking at the calculated center probability valu. Another proposed solution was using Generalized Dice overlap as the loss function to overcome the class imbalance problem.  
