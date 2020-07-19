import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def match_features(img1, img2):
	orb = cv.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	bf = cv.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)
	# Apply ratio test
	good = []
	pts1 = []
	pts2 = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)

	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)

	return pts1, pts2

def fundamental_matrix(pts1, pts2):
	F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]

	return F, mask, pts1, pts2

def drawlines(img1,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)

    return img1

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img3 = drawlines(img1,lines1,pts1,pts2)
	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img4 = drawlines(img2,lines2,pts2,pts1)
	
	cv.imwrite('EpipolarLines_First.JPG', img3)
	cv.imwrite('EpipolarLines_Second.JPG', img4)

	return lines1, lines2

def epipoles_from_lines(lines1, lines2):
	(a1, b1, c1) = lines1[0]
	(a2, b2, c2) = lines1[1]

	a = b1*c2 - c1*b2
	b = c1*a2 - c2*a1
	c = a1*b2 - a2*b1

	el = np.asarray([a/c,b/c])

	(a1, b1, c1) = lines2[0]
	(a2, b2, c2) = lines2[1]

	a = b1*c2 - c1*b2
	b = c1*a2 - c2*a1
	c = a1*b2 - a2*b1

	er = np.asarray([a/c,b/c])

	return el, er

def epipoles_from_F(F):
	l_sub = F[:2,:2]
	l_col3 = F[:2,2]
	el = -np.linalg.inv(l_sub).dot(l_col3)

	F = F.T
	r_sub = F[:2,:2]
	r_col3 = F[:2,2]
	er = -np.linalg.inv(r_sub).dot(r_col3)

	return el, er

def get_projection_matrices(F, er):
	P1 = np.zeros((3,4))
	P2 = np.zeros((3,4))

	P1[:,:3] = np.eye(3)

	er = np.asarray([er[0],er[1],1])
	P2[:,:3] = np.cross(er,F)
	P2[:,3] = er
	
	return P1, P2

def get_scene_point(p1, p2, P1, P2, img1, img2, color):
	img1 = cv.circle(img1,tuple(p1),5,color,-1)
	img2 = cv.circle(img2,tuple(p2),5,color,-1)

	p1 = np.asarray([p1[0],p1[1], 1]).reshape((3,1))
	p2 = np.asarray([p2[0],p2[1], 1]).reshape((3,1))

	C1 = -np.linalg.inv(P1[:,:3]).dot(P1[:,3])
	C2 = -np.linalg.inv(P2[:,:3]).dot(P2[:,3])
	
	d1 = np.linalg.inv(P1[:,:3]).dot(p1)
	d2 = np.linalg.inv(P2[:,:3]).dot(p2)
	d1 = np.squeeze(d1)
	d2 = np.squeeze(d2)

	scene_point = np.cross(d1,d2) 
	
	return img1, img2, scene_point

def main():
	img1 = cv.imread('./Amitava_first.JPG',0)
	img2 = cv.imread('./Amitava_second.JPG',0)
	# print(img1.shape)
	# print(img2.shape)
	pts1, pts2 = match_features(img1,img2)
	# print(pts1)

	F, mask, pts1, pts2 = fundamental_matrix(pts1, pts2)
	print('+++ Fundamental Matrix = ')
	print(F)
	print()

	lines1, lines2 = draw_epipolar_lines(img1, img2, pts1, pts2, F)

	el1, er1 = epipoles_from_lines(lines1, lines2)
	print('+++ Epipoles from lines = ')
	print(el1,er1)
	print()

	el2, er2 = epipoles_from_F(F)
	print('+++ Epipoles from Fundamental Matrix = ')
	print(el2,er2)
	print()

	print('Distance between computed left epipoles = '+str(np.linalg.norm(el1-el2)))
	print('Distance between computed right epipoles = '+str(np.linalg.norm(er1-er2)))
	### Error is due to float precision difference in lines and F
	print()

	PL, PR = get_projection_matrices(F, er1)
	print('+++ Estimated Left Projection Matrix = ')
	print(PL)
	print()
	print('+++ Estimated Right Projection Matrix = ')
	print(PR)
	print()

	img1 = cv.imread('./Amitava_first.JPG',1)
	img2 = cv.imread('./Amitava_second.JPG',1)
	
	img1, img2, scene_point1 = get_scene_point(pts1[0], pts2[0], PL, PR, img1, img2, (255,0,0))
	print('+++ World coordinate for neck of statue: ')
	print(scene_point1)
	img1, img2, scene_point2 = get_scene_point(pts1[1], pts2[1], PL, PR, img1, img2, (0,255,0))
	print('+++ World coordinate for trees in background: ')
	print(scene_point2)
	cv.imwrite('Scene_Points_1.JPG', img1)
	cv.imwrite('Scene_Points_2.JPG', img2)
	
if __name__ == "__main__":
	main()

