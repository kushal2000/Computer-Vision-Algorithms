import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

points = []
colors = np.array([[0,0,128],
		[0,128,128],
		[128,0,0],
		[0,0,0],
		[255,255,255],
		[48,130,245],
		[25,255,255],
		[180,30,145],
		[195,255,170]])
def draw_point(event,x,y,flags,param):
	if event == cv2.EVENT_FLAG_LBUTTON:
		points.append((x,y))

def Choose_Point():	
	cv2.setMouseCallback("Choose",draw_point)
	while(1):
		if len(points) == 1:
			break
		cv2.waitKey(10)
	point = points[0]
	points.clear()
	return point

def curvatures(range_image):
	h_y, h_x = np.gradient(range_image)
	h_xy, h_xx = np.gradient(h_x)
	h_yy, h_yx = np.gradient(h_y)

	assert (np.linalg.norm(h_xy-h_yx)==0) , 'unequal gradients'

	K = (h_xx*h_yy-h_xy*h_yx)/((1+h_x*h_x+h_y*h_y)**2)
	H = ((1+h_x*h_x)*h_yy+(1+h_y*h_y)*h_xx-2*(h_x*h_y)*h_xy)/(((1+h_x*h_x+h_y*h_y)**1.5))
	
	P_max = H + np.sqrt(H**2-K)
	P_min = H - np.sqrt(H**2-K)

	return H, K, P_min, P_max

def principleSegmentation(P_min, P_max, p_seg):
	for x in range(p_seg.shape[0]):
		for y in range(p_seg.shape[1]):
			k1, k2 = P_max[x,y], P_min[x,y]
			if k1*k2 < 0:
				p_seg[x,y] = colors[0]
			elif k1 < 0 and k2 < 0:
				p_seg[x,y] = colors[1]
			elif k1 > 0 and k2 > 0:
				p_seg[x,y] = colors[2] 
			elif k1 == 0 and k2==0:
				p_seg[x,y] = colors[3]
			elif k1 == 0:
				if k2 < 0:
					p_seg[x,y] = colors[4]
				else:
					p_seg[x,y] = colors[5]
			elif k2 == 0:
				if k1 < 0:
					p_seg[x,y] = colors[4]
				else:
					p_seg[x,y] = colors[5]
	return p_seg

def gaussianSegmentation(H, K, g_seg):
	for x in range(g_seg.shape[0]):
		for y in range(g_seg.shape[1]):
			h = H[x,y]
			k = K[x,y]
			if k<0:
				if h < 0:
					g_seg[x,y] = colors[0]
				elif h == 0:
					g_seg[x,y] = colors[1]
				else:
					g_seg[x,y] = colors[2]
			elif k == 0:
				if h < 0:
					g_seg[x,y] = colors[4]
				elif h==0:
					g_seg[x,y] = colors[3]
				else:
					g_seg[x,y] = colors[5]
			else:
				if h<0:
					g_seg[x,y] = colors[6]
				elif h==0:
					g_seg[x,y] = colors[7]
				else:
					g_seg[x,y] = colors[8]
	return g_seg

def isValid(img, x, y):
	if x > 0 and x<img.shape[0] and y>0 and y<img.shape[1]:
		return True
	return False

def NPS(range_image, point):
	(x,y) = point
	a = 1
	b = 1
	k = 20
	nps_set = []
	for i in range(x-a, x+a):
		for j in range(y-b, y+b):
			if isValid(range_image, i, j):
				if i==x and j==y:
					continue
				if abs(range_image[i,j]-range_image[x,y]) < k :
					nps_set.append((i,j))
	return nps_set

def NPSSegmentation(nps_seg, range_image):
	num = 1
	a = 1
	b = 1
	k = 1
	num = 0
	for x in range(range_image.shape[0]):
		for y in range(range_image.shape[1]):
			for i in range(x-a, x+a):
				for j in range(y-b, y+b):
					if isValid(range_image, i, j):
						if i==x and j==y:
							continue
						if abs(range_image[i,j]-range_image[x,y]) < k :
							num = nps_seg[i,j]
							
	return nps_seg, num
	

def main():
	opt_image = cv2.imread('./RGBD_dataset/1.jpg',1)
	range_image = cv2.imread('./RGBD_dataset/1.png', 0)
	
	H, K, P_min, P_max = curvatures(range_image)

	cv2.imshow("Choose",opt_image)
	print('Select a point for topology estimation')
	print('-----------------------------------------------')

	topo_point = Choose_Point()
	(x,y) = topo_point
	cv2.rectangle(opt_image,(x-2,y-2),(x+2,y+2),(0,255,0),-1)
	cv2.imshow("Choose",opt_image)

	print('Principle Curvature (min) =', P_min[y,x])
	print('Principle Curvature (max) =', P_max[y,x])
	print('Gaussian Curvature =', K[y,x])
	print('-----------------------------------------------')

	k1, k2 = P_max[y,x], P_min[y,x]
	print('Based on Principle Curvatures:')
	if k1*k2 < 0:
		print('SADDLE POINT')
	elif k1 < 0 and k2 < 0:
		print('PEAK')
	elif k1 > 0 and k2 > 0:
		print('PIT') 
	elif k1 == 0 and k2==0:
		print('FLAT')
	elif k1 == 0:
		if k2 < 0:
			print('RIDGE')
		else:
			print('VALLEY')
	elif k2 == 0:
		if k1 < 0:
			print('RIDGE')
		else:
			print('VALLEY')

	print('-----------------------------------------------')

	h = H[y,x]
	k = K[y,x]
	print('Based on mean and Gaussian: ')
	if k<0:
		if h < 0:
			print('SADDLE RIDGE')
		elif h == 0:
			print('MINIMAL SURFACE')
		else:
			print('SADDLE VALLEY')
	elif k == 0:
		if h < 0:
			print('RIDGE')
		elif h==0:
			print('FLAT')
		else:
			print('VALLEY')
	else:
		if h<0:
			print('PEAK')
		elif h==0:
			print('NONE')
		else:
			print('PIT')

	print('-----------------------------------------------')
	print('We use cardinality = 3 for NPS with a threshold k for range values')
	nps_set = NPS(range_image, (y,x))
	print('NPS set for selected point is: ')
	print(nps_set)
	nps_seg = np.zeros_like(range_image)
	# nps_seg, num = NPSSegmentation(nps_seg, range_image)
	# print(num)
	print('-----------------------------------------------')


	cv2.imshow('Optical Image', opt_image)
	cv2.imwrite('Optical_Image.jpg', opt_image)

	cv2.imshow('Range Image', range_image)
	cv2.imwrite('Range_Image.jpg', range_image)

	p_seg = np.zeros_like(opt_image)
	p_seg = principleSegmentation(P_min, P_max, p_seg)
	cv2.imshow('Principle Segmented', p_seg)
	cv2.imwrite('Principle_Segmented.jpg', p_seg)

	g_seg = np.zeros_like(opt_image)
	g_seg = gaussianSegmentation(H, K, g_seg)
	cv2.imshow('Gaussian Segmented', g_seg)
	cv2.imwrite('Gaussiam_Segmented.jpg', g_seg)
	cv2.waitKey(0)

if __name__ == "__main__":
	main()

