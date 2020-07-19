## Assignment 3

* Execute using ``` python a3.py ```
* Epipolar Lines stored in  'EpipolarLines_First.JPG' and 'EpipolarLines_Second.JPG'
* A couple of scene_points matched in 'Scene_Points_1.JPG' and 'Scene_Points_2.JPG' to compare real depths with image points

### Feature Matching

* Orb Descriptor used in match_features functions which returns valid set of points

### Fundamental Matrix

* fundamental_matrix function computes F from input point correspondences using opencv library inbuilt function cv.findFundamentalMat

### Epipolar Lines

* draw_epipolar_lines functions uses helper draw_lines function to actually sketch the lines. cv.computeCorrespondEpilines is used to find the line equations from F and point correspondences.

### Epipoles

* Epipoles are computed from the epipolar lines computed above as well as from the Fundamental Matrix' zeros. epipoles_from_lines and epipoles_from_F are the 2 functions implementing it.

### Estimation of Projection Matrices

* get_projection_matrices function computes an estimated projection matrix. 

### Depth Comparison of World Coordinates

* get_scene_point computes the geometry required for this part 
* We compute the world coordinate for a point on the statue's neck -> 
	[-1.05799744e+17 -5.98195634e+16  4.72867346e+19]
* We compute the world coordinate for a point on the tree background in the left half of the image -> 
	[ 2.06515713e+17  4.16687147e+17 -1.56016402e+20]
* As expected, the depth of the trees has a higher magnitude -> a fact which is intuitevly confirmed
