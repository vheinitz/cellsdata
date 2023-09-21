import cv2 as cv
import numpy as np
import os

# Specify the directory path where your PNG images are located
directory_path = "."

# List all files in the directory
file_list = os.listdir(directory_path)

# Filter the list to include only PNG files
png_files = [file for file in file_list if file.lower().endswith(".png")]

png_files.sort()

for png_file in png_files:
    orig = cv.imread(png_file, cv.IMREAD_COLOR)
    orig = cv.pyrDown(orig)
    image = np.copy(orig)
    #
    blue_channel, green_channel, red_channel = cv.split(image)

    # Use the green channel for segmentation
    gray = green_channel

    # Apply Gaussian blur to reduce noise and smooth the image
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    _, th1 = cv.threshold(blurred, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)


    binary_image = th1
    cv.imshow('binary_image Cells', binary_image)

    # Perform morphological operations to remove noise and enhance cell structures
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    cv.imshow('sure_bg', sure_bg)
    # Compute the distance transform to find the foreground markers
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    cv.imshow('dist_transform', dist_transform)
    _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    cv.imshow('sure_fg', sure_fg)
    # Subtract the sure foreground from the sure background to get the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Markers for watershed segmentation
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed segmentation
    markers = cv.watershed(image, markers)

    mimg = np.zeros((image.shape[0],image.shape[1],1), np.uint8)

    mimg[markers > 1] = [255]  # Mark the boundaries in red

    # Find contours
    contours, _ = cv.findContours(mimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Loop through all the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv.contourArea(contour)

        # Fill contours of a defined size (e.g., between 100 and 1000)
        if 500 < area < 1000:
            cv.drawContours(image, [contour], -1, (0, 255, 0), thickness=cv.FILLED)  # Filled contour

    # Display the segmented image15
    cv.imshow('Original', orig)
    cv.imshow('Segmented Cells', image)
    cv.imshow('Mask', mimg)
    cv.waitKey(0)

cv.destroyAllWindows()