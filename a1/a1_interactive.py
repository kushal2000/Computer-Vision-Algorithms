import cv2
import numpy as np


def isValid(img, i, j):
	if i < img.shape[0] and i >= 0 and j < img.shape[1] and j >= 0:
		return True
	return False

def gaussian(var1, var2, stdvar):
	dist = np.linalg.norm(var1-var2)
	val = np.exp(-(dist**2/(2*stdvar**2)))
	return val

def scaled_intensity(img, point, stdvar, k):
	val = 0
	x = point[0]
	y = point[1]
	k = (int)(k/2)
	for i in range(x-k,x+k+1):
		for j in range(y-k,y+k+1):
			if isValid(img, i, j):
				val += gaussian(np.array([img[x,y]]),np.array([img[i,j]]), stdvar)/(2*np.pi*stdvar*stdvar)
	return val

def sbf(img, sigma_s = 4, sigma_r = 12, sigma_g = 3, k = 5):
	sbf_img = np.zeros(img.shape, dtype=np.uint8)
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			val1 = 0
			val2 = 0
			k = (int)(k/2)
			for i in range(x-k,x+k+1):
				for j in range(y-k,y+k+1):
					if isValid(img, i, j):
						scaled_int = scaled_intensity(img, (i,j), sigma_g, k)
						scaled_range = gaussian(scaled_int, img[x,y], sigma_r)
						spatial_gaussian = gaussian(np.array([x,y]),np.array([i,j]),sigma_s)
						val1 += spatial_gaussian*scaled_range*img[i,j]
						val2 += spatial_gaussian*scaled_range
			sbf_img[x,y] = (int)(val1/val2);
	return sbf_img

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0):
    #Return a sharpened version of the image, using an unsharp mask
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened

def edgeExtractor(img):
	edges = cv2.Laplacian(img,cv2.CV_16S)
	edges = cv2.convertScaleAbs(edges)
	return edges

def adaptiveThresh(img, kernel_size = 11, sigma = 2):
	th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
								cv2.THRESH_BINARY,kernel_size,sigma)
	return th

def cornerDetector(img, blockSize = 3, ksize = 5, freeparam = 0.04):
	img1=np.float32(img)
	dest = cv2.cornerHarris(img1, blockSize, ksize, freeparam)
	img2 = np.zeros(img.shape, dtype=np.uint8)
	img2[dest>0.01*dest.max()]=255
	return img2

def meanShiftFilter(img, ksize = 9, diff = 15):
	img1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	shifted = cv2.pyrMeanShiftFiltering(img1, ksize, diff)
	gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
	thresh = adaptiveThresh(gray)
	edges = edgeExtractor(thresh)
	return thresh,edges

def morphologicalOperation(img, operator = 'opening', kernel = np.ones((5,5),np.uint8)):
	if operator == 'erode':
		morph = cv2.erode(img,kernel,iterations = 1)
	if operator == 'dilate':
		morph = cv2.dilate(img,kernel,iterations = 1)
	if operator == 'opening':
		morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	if operator == 'closing':
		morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	return morph

img=cv2.imread('cavepainting1.JPG')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:
	cv2.imshow('Result',gray)
	cv2.waitKey(1000)
	print('Press 1 for sbf filter')
	print('Press 2 for image sharpening')
	print('Press 3 for edge extraction')
	print('Press 4 for binary thresholding')
	print('Press 5 for corner detection')
	print('Press 6 for image segmentation')
	print('Press 7 for morphed image')
	print('Press 8 for exit')

	val = input("Enter your value: ")

	if val == '1':
		gray = sbf(gray)
	elif val == '2':
		gray = unsharp_mask(gray)
	elif val == '3':
		gray = edgeExtractor(gray)
	elif val == '4':
		gray = adaptiveThresh(gray)
	elif val == '5':
		gray = cornerDetector(gray)
	elif val == '6':
		gray, lines = meanShiftFilter(gray)
		cv2.imshow('Result_lines',lines)
	elif val == '7':
		gray = morphologicalOperation(gray)
	elif val == '8':
		cv2.destroyAllWindows()
		break
	else:
		print('Wrong input! Try again')

	    