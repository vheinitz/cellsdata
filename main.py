from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)
parser = argparse.ArgumentParser(description='Code for Image Segmentation with Distance Transform and Watershed Algorithm.\
 Sample code showing how to segment overlapping objects using Laplacian filtering, \
 in addition to Watershed and Distance Transformation')
parser.add_argument('--input', help='Path to input image.', default='cards.png')
args = parser.parse_args()
src = cv.imread("//tmp/img/8.png")
src = cv.pyrDown(src)
#src = cv.pyrDown(src)
_, green, _ = cv.split(src)



# Apply Gaussian blur to reduce noise and smooth the image
blured = cv.GaussianBlur(green, (5, 5), 0)

cv.imshow('Source Image', src)
#src[np.all(src == 255, axis=2)] = 0
# Show output image
#cv.imshow('Black Background Image', src)
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated


#imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
#sharp = np.float32(src)
#imgResult = sharp - imgLaplacian
## convert back to 8bits gray scale
#imgResult = np.clip(imgResult, 0, 255)
#imgResult = imgResult.astype('uint8')
#imgLaplacian = np.clip(imgLaplacian, 0, 255)
#imgLaplacian = np.uint8(imgLaplacian)

#cv.imshow('Laplace Filtered Image', imgLaplacian)
#cv.imshow('New Sharped Image', imgResult)
imgResult = blured
bw = imgResult #cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
_, th1 = cv.threshold(bw, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
th2 = cv.adaptiveThreshold(bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 2)

bw = th2 + th1

kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size as needed
bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)


cv.imshow('Binary Image', bw)
dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
cellsize = 20
dist[dist > cellsize] = 0
_,th1 = cv.threshold(dist.astype('uint8'), 1, 255, cv.THRESH_BINARY )
dist = cv.distanceTransform(th1, cv.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', dist)
_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
# Dilate a bit the dist image
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)
cv.imshow('Peaks', dist)
dist_8u = dist.astype('uint8')
# Find total markers
contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# Create the marker image for the watershed algorithm
markers = np.zeros(dist.shape, dtype=np.int32)
# Draw the foreground markers
for i in range(len(contours)):
 cv.drawContours(markers, contours, i, (i+1), -1)
# Draw the background marker
cv.circle(markers, (5,5), 3, (255,255,255), -1)
markers_8u = (markers * 10).astype('uint8')
cv.imshow('Markers', markers_8u)
cv.watershed(src, markers)
#mark = np.zeros(markers.shape, dtype=np.uint8)
mark = markers.astype('uint8')
mark = cv.bitwise_not(mark)
# uncomment this if you want to see how the mark
# image looks like at that point
#cv.imshow('Markers_v2', mark)
# Generate random colors
colors = []
for contour in contours:
 colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
# Fill labeled objects with random colors
for i in range(markers.shape[0]):
 for j in range(markers.shape[1]):
  index = markers[i,j]
  if index > 0 and index <= len(contours):
   dst[i,j,:] = colors[index-1]
# Visualize the final image
cv.imshow('Final Result', dst)
cv.waitKey()