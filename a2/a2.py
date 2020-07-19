import numpy as np 
import cv2
import sys

image=None
mod_image=None
points = []
lines = []
p_h=0
p_w=0

def draw_point(event,x,y,flags,param):
	if event == cv2.EVENT_FLAG_LBUTTON:
		cv2.rectangle(image,(x-2,y-2),(x+2,y+2),(0,255,0),-1)
		points.append((x,y))

def store_line(points):
	(x2,y2) = points[1]
	(x1,y1) = points[0]
	if x2==x1:
		b = 0
		a = 1
		c = -x1
	else:
		a = (y1-y2)/(x2-x1)
		c = -a*x1 - y1
		b = 1
	lines.append((a,b,c))
	print('Equation of the line is  = '+str(a)+'x + ' + str(b) + ' y '+str(c))
	print('P2 representation of line = '+str(lines[len(lines)-1]))

def drawP():
	cv2.namedWindow("Original")
	cv2.setMouseCallback("Original",draw_point)
	while(1):
		if len(points) == 1:
			break
		cv2.waitKey(10)
	cv2.imshow("Original",image)
	point = points[0]
	points.clear()
	return point

def drawline():
	cv2.namedWindow("Original")
	cv2.setMouseCallback("Original",draw_point)
	while(1):
		cv2.imshow("Original",image)
		if len(points) == 2:
			cv2.line(image, points[1], points[0], (255,0,0), 3)
			store_line(points)
			cv2.imshow("Original",image)
			break
		cv2.waitKey(10)
	points.clear()

def vanish_point(lines):
	n = len(lines)
	(a2, b2, c2) = lines[n-1]
	(a1, b1, c1) = lines[n-2]
	a = b1*c2 - b2*c1
	b = c1*a2 - c2*a1
	c = a1*b2 - a2*b1
	if c != 0:
		vanish_point = (a/c,b/c,1)
	print("Vanishing Point of the lines lies at "+ str(vanish_point))
	return vanish_point

def vanishing_line(point1, point2):
	rows = image.shape[0]
	points.append((point1[0],point1[1]))
	points.append((point2[0],point2[1]))
	store_line(points)
	(a,b,c) = lines[len(lines)-1]
	start_point = None
	end_point = None
	for x in range(image.shape[1]):
		y = (-a*x - c)/b
		if y >=0 and y < rows:
			if start_point == None:
				start_point = (x,(int)(y))
			else:
				end_point = (x,(int)(y))
	points.clear()
	cv2.line(image, start_point, end_point, (255,255,255), 3)
	cv2.imshow("Original",image)
	return (a,b,c)

def centre_line():
	rows = image.shape[0]
	(a,b,c) = lines[len(lines)-1]
	(x,y) = (image.shape[1]/2,image.shape[0]/2)
	c = -(a*x + b*y)
	start_point = None
	end_point = None
	for x in range(image.shape[1]):
		y = (-a*x - c)/b
		if y >=0 and y < rows:
			if start_point == None:
				start_point = (x,(int)(y))
			else:
				end_point = (x,(int)(y))
	cv2.line(image, start_point, end_point, (0,0,255), 3)
	cv2.imshow("Original",image)
	return (a,b,c)

def vertical_bar(p, vanish_line, centre_line):
	x = p[0]
	(a,b,c) = vanish_line
	y1 = -(a*p[0]+c)/b
	(a,b,c) = centre_line
	y2 = -(a*p[0]+c)/b
	y1 = (int)(y1)
	y2 = (int)(y2)
	start_point = (x,y1)
	end_point = (x,y2)
	cv2.line(image, start_point, end_point, (255,0,0), 3)
	cv2.imshow("Original",image)
	return start_point, end_point

def plausible_Homography(img):
	rows,cols,ch = img.shape
	pts1 = np.float32([[70,575],[250,420],[800,430],[850,583]])
	pts2 = np.float32([[70,650],[70,400],[800,400],[800,650]])

	H = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,H,(cols,rows))
	return dst

def affine_Transform(img):
	rows,cols,ch = img.shape
	pts1 = np.float32([[70,575],[760,430],[850,583]])
	pts2 = np.float32([[70,575],[850,350],[850,575]])

	M = cv2.getAffineTransform(pts1,pts2)

	dst = cv2.warpAffine(img,M,(cols,rows))
	dst = dst[350:575,70:850]
	return dst

def start(image_name):
	global image, mod_image, points
	image=cv2.imread(image_name)
	mod_image = image.copy()
	cv2.imshow("Original",image)
	print("Choose two parallel lines")
	drawline()
	drawline()
	print('-------------------------------------------------------------------')
	vanish1 = vanish_point(lines)
	print('-------------------------------------------------------------------')
	print("Choose two parallel lines")
	drawline()
	drawline()
	print('-------------------------------------------------------------------')
	vanish2 = vanish_point(lines)
	print('-------------------------------------------------------------------')
	
	print('-------------------------------------------------------------------')
	print('Vanishing Line given by')
	vanish_line = vanishing_line(vanish1,vanish2)
	centre = centre_line()
	print('-------------------------------------------------------------------')

	print('Click a point p')
	p = drawP()
	start_point, end_point = vertical_bar(p, vanish_line, centre) 
	points.append(start_point)
	points.append(end_point)
	store_line(points)
	points.clear()
	cv2.imwrite("vertical_bar.jpg",image)

	transformed = plausible_Homography(image)
	cv2.imshow('Transformed', transformed)

	mod_image = affine_Transform(mod_image)
	cv2.imshow("Affine",mod_image)
	cv2.imwrite("affine.jpg",mod_image)
	cv2.waitKey(0)

if __name__ == "__main__":
	start(sys.argv[1])

