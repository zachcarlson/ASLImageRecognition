#reference: https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio

import os
import sys
import PIL.Image as Image
import numpy as np

#this function assumes an image larger or equal in both width and height compared to dimension parameter value
def SquareCropStartOfContrast(im, dimension = 224, threshold = 60):
    #loop rows until find a pixel with high contrast compared to previous
    #start crop at that row, crop as a square
    im_arr = np.array(im, dtype=np.uint8)
    prev = sum(im_arr[0][0])/len(im_arr[0][0])
    borderY = dimension/3
    borderX = dimension/2
    while threshold > 0: #loop until we have found no contrasting pixels, reducing contrast threshold each loop through image
        for r, row in enumerate(im_arr):
            prev = sum(im_arr[r][0])/len(im_arr[r][0])
            for p, pixel in enumerate(row):
                pixel_mean = sum(pixel)/len(pixel)
                if abs(prev - pixel_mean) > threshold:
                    upper = max(0,r - borderY)
                    upper = min(upper,im_arr.shape[0]-dimension)
                    lower = upper+dimension
                    left = max(0,p - borderX)
                    left = min(left,im_arr.shape[1]-dimension)
                    right = left+dimension #assuming same image width as dimension
                    return im.crop((left, upper, right, lower))
                prev = pixel_mean
        print('.')
        threshold = threshold - 1
    return im


def ResizeImages(uncroppedDir="", croppedDir="",dimension = (224,398), squareCropStartOfContrast = False):

    if not os.path.exists(uncroppedDir):
        print('No uncropped directory found.')
        return
    os.makedirs(os.path.dirname(croppedDir), exist_ok=True)

    numUncropped = len(os.listdir(uncroppedDir))
    if numUncropped < 1:
        print('No images in uncropped directory.')
        return
    for i,infile in enumerate(os.listdir(uncroppedDir)):
        print(int(i/numUncropped*100), '%', end='\r')
        outfile = croppedDir + infile
        try:
            im = Image.open(uncroppedDir+infile)
            width, height = im.size
            if dimension[0] == dimension[1]:
                min_dim = min(width,height)
                box_dim = [width/2-min_dim/2, height/2-min_dim/2, width/2+min_dim/2, height/2+min_dim/2]
                im = im.resize(dimension, Image.ANTIALIAS,box=box_dim)
            else:
                im = im.resize(dimension, Image.ANTIALIAS)
            if squareCropStartOfContrast:
                im = SquareCropStartOfContrast(im)
            im.save(outfile, "png")
        except IOError as e:
            print(e,"ERROR - failed to crop '%s'" % infile)
    print("100%")

def LaunchCrop(subdirectories = True, inpath = "uncropped_frames/",outpath = "cropped_frames/", dimensionX = 224, dimensionY = 398, squareCropStartOfContrast = True):
    print("Cropping...")
    size = dimensionX, dimensionY

    if subdirectories:
        dirs = os.listdir(inpath)
        for d in dirs:
            print("Cropping files in directory:", inpath+d+'/')
            ResizeImages(inpath+d+'/', outpath+d+'/', size, squareCropStartOfContrast)
    else:
        ResizeImages(inpath, outpath, size, squareCropStartOfContrast)


args = sys.argv
if len(args) < 2:
    LaunchCrop(subdirectories = True, inpath = "uncropped_frames/",outpath = "cropped_frames/", dimensionX = 224, dimensionY = 398, squareCropStartOfContrast = True)
elif len(args) < 6:
    print("Not enough arguments. Please provide: 1)subdirectories (boolean), 2)inpath (relative to working directory, ending in /), 3)outpath (ending in /), 4)dimensionX (integer), 5)dimensionY (integer), 6)squareCropStartOfContrast boolean") 
else:
    LaunchCrop(subdirectories = (sys.argv[1].lower() == 'true'), 
                inpath = args[2], outpath = args[3], dimensionX = int(args[4]), dimensionY = int(args[5]),squareCropStartOfContrast=(sys.argv[6].lower() == 'true'))
