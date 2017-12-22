# Standard imports
import cv2
import numpy as np;

# Read image

im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)

#image enhancement 

imgd = cv2.bilateralFilter(im,9,75,75)
threshq = cv2.threshold(imgd, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#finding edges

ime = cv2.Canny(threshq,75,200)

#extract contours

img, contours, hierarchy = cv2.findContours(ime,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#imgg = cv2.drawContours(im, contours, -1, (0,255,0), 3)



cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
crop_im = im[y:y+h,x:x+w]

img_filter = cv2.bilateralFilter(crop_im,9,75,75)
img_thresh = cv2.threshold(img_filter, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img_thresh,kernel,iterations = 1)
img_edge = cv2.Canny(erosion,75,200)
#cv2.imwrite("hello.jpg",erosion)
img_g, cos, hierarchy = cv2.findContours(img_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#check_img = cv2.drawContours(crop_im, contours, -1, (0,255,0), 3)
#cv2.imwrite("heli.jpg",check_img)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
#params.minThreshold = 250;
#params.maxThreshold = 255;

# Filter by Area.
params.filterByArea = True
params.minArea = 50

#Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.4

# # Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.2

# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

#color
params.filterByColor = True
params.blobColor = 255

detector = {}
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else :
    detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(erosion)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(crop_im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)
cv2.imwrite("image.jpg", im_with_keypoints)
