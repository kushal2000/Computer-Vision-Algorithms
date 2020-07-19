## Assignment 4

### Execution of Code

* Execute using ``` python range.py ```
* You will be first prompted to choose a point whose topology estimation is done in main itself.
* You have to prompt the point in choose window

** There will be 5 active windows at the end:
* Optical -> Displays Optical Image
* Range -> Displays Range Image
* Principal Segmented -> Displays segmented image using Principal Curvatures
* Gaussian Segmented -> Displays segmented image using Gaussian and Mean Curvatures
* NPS Segmented -> Displays NPS Segmented image

### Principal, Gaussian and Mean Curvatures

* Computed using curvatures() function which uses the required first and second order gradients to compute the Mean and Gaussian Curvatures from which Principal Curvatures are computed.
* Topology analysis is done on a point chosen by user :
	a) Using signs of Principal Curvatures 
	b) Using signs of Gaussian and Mean Curvatures


### Neighbourhood Plane Set

* We assume cardinality = 3
* a = 2, b=2, k = 20 is taken as default
* Computed for sample point in NPS()

### Image Segmentation

* Based on labels generated from above (a),principalSegmented() gives the segmented image as output
* Based on labels generated from above (b),gaussianSegmented() gives the segmented image as output
* Using NPSSegmentation(), we call NPS_DFS() on all sets to generate all the segments

### Segmentation Quality Analysis 

* Principal Segmented > Gaussian Segmentation, bugs in NPS_DFS did not permit the analysis of NPS



