# import 

import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Helper function

def get_binary_image(undist_img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    
    """
    This function accepts undistorted_image as input, sobel threshold and color threshold as input.
    
    The function returns binary_image
    
    input: undist_img, s_thresh, sx_thresh
    args:
    return: binary_image
    
    """
    
    img = np.copy(undist_img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    
    # Combine the two binary thresholds
    binary_image = np.zeros_like(sxbinary)
    binary_image[(s_binary == 1) | (sxbinary == 1)] = 1
    return binary_image
    
def visualizeColorTheshold(image1, image2):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()

    ax1.imshow(image1)
    ax1.set_title('Color_binary', fontsize=15)

    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Binary_image', fontsize=15)
