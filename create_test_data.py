import cv2
import numpy as np

# Define the resolution (1MP)
f = 1.4
width, height = int(1920/f), int(1080/f)  # 1920x1080 pixels


for intensity in range(10):
    noisy_image = np.zeros((height, width, 3), dtype=np.uint8)
    mean_channel = [5, 20, 5]
    stddev_channel = [1, 5, 1]  # Adjust these values to control noise for each channel
    for i in range(3):
        noise = np.random.normal(mean_channel[i], stddev_channel[i], (height, width)).astype(np.uint8)
        noisy_image[:, :, i] = cv2.add(noisy_image[:, :, i], noise)

    # Define parameters for the grid of green circles
    min_diameter = 50
    max_diameter = 60
    step = 0

    circle_color = (0, 20 + (intensity * 5 ), 0)  # Green color in BGR format

    # Calculate the number of rows and columns for the grid
    num_rows = (height - min_diameter) // max_diameter + 1
    num_cols = (width - min_diameter) // max_diameter + 1

    diameter = max_diameter
    margin=50
    x = diameter + margin
    y = diameter + margin

    while diameter > min_diameter:
            cv2.circle(noisy_image, (x,y), diameter // 2, circle_color, -1)
            diameter -= step
            x += diameter + margin
            if x > width - ( diameter + margin ):
                x = diameter + margin
                y += diameter*2 + margin
                if y > height - (diameter + margin):
                    break

    cv2.imwrite("intensity_%d.png" % (intensity), noisy_image)
    cv2.imshow("Noisy Black Image", noisy_image)
    cv2.waitKey(250)
cv2.destroyAllWindows()
