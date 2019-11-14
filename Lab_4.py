import cv2
import numpy as np
import matplotlib as plt
import scipy as sc
from pprint import pprint


def convolve(image,	kernel):
    (iH,	iW) = image.shape[:2]
    (kH,	kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros([iH, iW], dtype="float32")
    for y in np.arange(pad,	iH + pad):
        for x in np.arange(pad,	iW + pad):
            roi = image[y-pad:y+pad+1, x-pad:x+pad+1]
            k = (roi*kernel).sum()
            output[y-pad, x-pad] = k
    # output = cv2.normalize(output,output,1,0,cv2.NORM_MINMAX)
    return output


def hough(threshold_image, direction_image):
    height, width = threshold_image.shape[:2]

    r0 = np.min(np.array([height, width]))//2
    H = np.zeros((height, width, r0))

    # for angle in range(0, 360):
    #     sinang[angle] = np.sin(angle * np.pi/180)
    #     cosang[angle] = np.cos(angle * np.pi/180)

    hough_image = np.zeros((height+1, width+1))
    for r in range(30, r0):
        for y in range(height):
            for x in range(width):
                direction = direction_image[y, x]
                # print(direction)
                if threshold_image[y, x] > 0:

                    x_0 = x+(r*np.cos(direction))
                    y_0 = y+(r*np.sin(direction))

                    x_0_2 = x-(r*np.cos(direction))
                    y_0_2 = y-(r*np.sin(direction))

                    if (x_0 > 0 and x_0 < width and y_0 > 0 and y_0 < height):
                        #print("inc 1")
                        H[int(y_0), int(x_0), r] += 1
                        hough_image[int(y_0), int(x_0)] += 1

                    if (x_0_2 > 0 and x_0_2 < width and y_0_2 > 0 and y_0_2 < height):
                        #print("inc 2")
                        H[int(y_0_2), int(x_0_2), r] += 1
                        hough_image[int(y_0_2), int(x_0_2)] += 1

    return H, hough_image


def threshold(image, T):
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            if image[y, x] > T:
                image[y, x] = 255
            else:
                image[y, x] = 0
    return image


def sobel(image):
    x_deriv_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_deriv_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    y_deriv_image = convolve(image, y_deriv_kernel)
    x_deriv_image = convolve(image, x_deriv_kernel)
    magnitude_image = image
    direction_image = image
    height, width = image.shape[:2]
    # Generate magnitude image
    for y in range(height):
        for x in range(width):
            magnitude_image[y, x] = np.sqrt(
                np.square(y_deriv_image[y, x])+np.square(x_deriv_image[y, x]))

    # magnitude_image = cv2.normalize(magnitude_image,magnitude_image, 1, 0, cv2.NORM_MINMAX)
    # cv2.imshow("mag_norm",magnitude_image)

    # Generate direction image
    direction_image = np.arctan(np.divide(y_deriv_image, x_deriv_image))
    direction_image = np.nan_to_num(direction_image)

    # direction_image = cv2.normalize(
    #     direction_image, direction_image, 1, 0, cv2.NORM_MINMAX)
    # cv2.imshow("direction", direction_image)

    # Generate dx image
    # x_deriv_image = cv2.normalize(x_deriv_image,x_deriv_image,1,0,cv2.NORM_MINMAX)
    # cv2.imshow("x_deriv",x_deriv_image)

    # Generate dy image
    # y_deriv_image = cv2.normalize(y_deriv_image,y_deriv_image,1,0,cv2.NORM_MINMAX)
    # cv2.imshow("y_deriv",y_deriv_image)

    # cv2.imshow("y_deriv")
    # cv2.waitKey()
    return magnitude_image, direction_image


image = cv2.imread("coins2.png")
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
mag_image, direction_image = sobel(image)
#cv2.imshow("dir", direction_image)
threshold_image = threshold(mag_image, 225)
hough_space, hough_image = hough(threshold_image, direction_image)
pprint(np.median(hough_image))
cv2.imshow("hough", threshold(hough_image, 12))
cv2.waitKey()
