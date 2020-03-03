import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage import morphology


def calculate_surface_normals(depth_img):
    shape = (*depth_img.shape[:2], 3)
    normal_data = np.zeros(shape, np.float32)

    for y in range(1, shape[0] - 1):
        for x in range(1, shape[1] - 1):
        dzdx = (depth_img[y, x+1] - depth_img[y, x-1]) / 2.0
        dzdy = (depth_img[y+1, x] - depth_img[y-1, x]) / 2.0

        d = np.asarray([-dzdx, -dzdy, 1.0])
        
        n = np.linalg.norm(d)
        normal_data[y, x] = n

    cv.imshow("normal img", normalize_depth)
    cv.waitKey(0)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
      

    # apply Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def normalize_depth(depthimg, colormap=None):
    # Normalize depth image to range 0-255.
    min, max, minloc, maxloc = cv2.minMaxLoc(depthimg)
    adjmap = np.zeros_like(depthimg)
    dst = cv2.convertScaleAbs(depthimg, adjmap, 255 / (max - min), -min)
    if colormap:
        return cv2.applyColorMap(dst, colormap)
    else:
        return dst
        
def clahe(img, iter=1):
    # evenly increases the contrast of the entire image
    # ref: http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    for i in range(0, iter):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img

# get image s
img_path = f"./data/img/test0.png"

img = cv2.imread(img_path, -1)
s_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1) # convolve2d(img, s_kernel)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1) # convolve2d(img, s_kernel.T)
sobelx = convolve2d(img, s_kernel.T).astype(np.float32)
sobely = convolve2d(img, s_kernel).astype(np.float32)
cv2.imshow("sobelx", sobelx)
cv2.imshow("sobely", sobely)

angle = cv2.phase(sobelx, sobely)
graddir = np.fix(180 + angle)
print(graddir.min())
print(graddir.max())

dimg1 = (((graddir - graddir.min()) / (graddir.max() - graddir.min())) * 255.9).astype(np.uint8)

cv2.imshow("grad", dimg1)


blur = cv2.bilateralFilter(dimg1, 9, 25, 25)
blur2 = cv2.bilateralFilter(blur, 9, 25, 25)

cv2.imshow("blur", blur2)

# Eliminate salt-and-pepper noise
median = cv2.medianBlur(blur2, 7)
cv2.imshow("median_blur", median)


v = np.mean(median)
    

# apply Canny edge detection using the computed median
lower = int(max(0, (1.0 - 0.53) * v))
upper = int(min(255, (1.0 + 0.53) * v))

dimg1 = cv2.Canny(median, lower, upper)

cv2.imshow("Canny stem", dimg1)

kernel = np.ones((7, 7), np.uint8)
dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow("dilation_ stemp", dilation)
skel = morphology.skeletonize(dilation > 0)

skel = skel.astype('uint8')
cv2.imshow("cd_final", skel)

#cv2.imshow("Grady", grad_y)

img = cv2.imread(img_path, -1)
img = normalize_depth(img)
img = clahe(img, 2)
dimg = auto_canny(img)
cv2.imshow("depth disc", img)


cv2.waitKey()
