import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

points = []

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

def chooseRectangle(img):
	cv2.namedWindow("Choose")
	cv2.imshow("Choose",img)

	p1 = Choose_Point()
	(x1,y1) = p1
	cv2.rectangle(img,(x1-2,y1-2),(x1+2,y1+2),(0,255,0),-1)
	cv2.imshow("Choose",img)
	
	p2 = Choose_Point()
	(x2,y2) = p2
	cv2.rectangle(img,(x2-2,y2-2),(x2+2,y2+2),(0,255,0),-1)

	cv2.rectangle(img,(x1-2,y2-2),(x1+2,y2+2),(0,255,0),-1)
	cv2.rectangle(img,(x2-2,y1-2),(x2+2,y1+2),(0,255,0),-1)

	cv2.line(img, (x1,y1), (x1,y2), (255,0,0), 3)
	cv2.line(img, (x2,y2), (x1,y2), (255,0,0), 3)
	cv2.line(img, (x2,y2), (x2,y1), (255,0,0), 3)
	cv2.line(img, (x2,y1), (x1,y1), (255,0,0), 3)
	cv2.destroyWindow("Choose")

	diagonal = [p1,p2]
	return img, diagonal

def domColour(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h_bin = 6
	s_bin = 8
	hist = cv2.calcHist([hsv_img], [0, 1], None, [h_bin, s_bin], [0, 180, 0, 256])
	ind = np.unravel_index(np.argmax(hist, axis=None), hist.shape)

	h_lower = (179.0/h_bin)*ind[0]
	h_upper = (179.0/h_bin)*(ind[0]+1)
	s_lower = (255.0/s_bin)*ind[1]
	s_upper = (255.0/s_bin)*(ind[1]+1)

	mask = cv2.inRange(hsv_img,(h_lower, s_lower, 0), (h_upper, s_upper, 255) )
	# mask = mask.reshape((mask.shape[0],mask.shape[1],1))
	return mask, (h_lower, h_upper), (s_lower, s_upper)

def transfer(img, H, S, color):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i,j,0] > H[0] and img[i,j,0] <= H[1] and img[i,j,1] > S[0] and img[i,j,1] <= S[1]:
				img[i,j,0] = color[0]
				img[i,j,1] = color[1]
	img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	return img

def main():
	img1 = cv2.imread('./pink.jpg',1)
	img2 = cv2.imread('./purple.jpg',1)
	img3 = cv2.imread('./white.jpg',1)

	src_img = img3
	tgt_img = img2

	cv2.imwrite('src_img.jpg', src_img)
	cv2.imwrite('tgt_img.jpg', tgt_img)

	print('Choose two points specifying diagonal of region in source image')
	src_img, src_diag = chooseRectangle(src_img)
	cv2.imshow("Source",src_img)

	print('Choose two points specifying diagonal of region in target image')
	tgt_img, tgt_diag = chooseRectangle(tgt_img)
	cv2.imshow("Target", tgt_img)

	cv2.imwrite('src_img_ROI.jpg', src_img)
	cv2.imwrite('tgt_img_ROI.jpg', tgt_img)

	x1, y1 = src_diag[0]
	x2, y2 = src_diag[1]
	src_rect = src_img[y1:y2, x1:x2]
	src_mask = np.zeros((src_img.shape[0],tgt_img.shape[1]))
	src_mask[y1:y2, x1:x2], src_H, src_S = domColour(src_rect)
	cv2.imshow("Source Mask", src_mask)

	x1, y1 = tgt_diag[0]
	x2, y2 = tgt_diag[1]
	tgt_rect = tgt_img[y1:y2, x1:x2]
	tgt_mask = np.zeros((tgt_img.shape[0],tgt_img.shape[1]))
	tgt_mask[y1:y2, x1:x2], tgt_H, tgt_S = domColour(tgt_rect)
	cv2.imshow("Target Mask", tgt_mask)

	cv2.imwrite('src_img_masked.jpg', src_mask)
	cv2.imwrite('tgt_img_masked.jpg', tgt_mask)

	color = ((src_H[0]+src_H[1])/2, (src_S[0]+src_S[1])/2)
	result = transfer(tgt_img, tgt_H, tgt_S, color)
	cv2.imshow("Result Image", result)

	cv2.imwrite('final_color_transfer.jpg', result)

	cv2.waitKey(0)

if __name__ == "__main__":
	main()

