import cv2
import os


def image_resize(path, image):

    file_stats = os.stat(path + "/" + image)
    sizebyte = int(file_stats.st_size / 1024)

    if (sizebyte >= 3000):
        scale_percent = 20
    elif (sizebyte >= 2000) and (sizebyte < 3000):
        scale_percent = 30
    elif (sizebyte >= 1500) and (sizebyte < 2000):
        scale_percent = 40
    elif (sizebyte >= 1000) and (sizebyte < 1500):
        scale_percent = 60
    elif (sizebyte >= 500) and (sizebyte < 1000):
        scale_percent = 70
    elif (sizebyte >= 300) and (sizebyte < 500):
        scale_percent = 90
    else:
        scale_percent = 100

    img = cv2.imread(path + "/" + image, cv2.IMREAD_UNCHANGED)

 
    print('Original Dimensions : ',img.shape)
 
    # scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    cv2.imwrite(path + "/" + image, resized)

 