#reference: https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio

import os
import PIL.Image as Image

def CropImages():
    print("Cropping...")
    size = 28, 28 #MNIST dataset has the most samples, so match the MNIST dimensions

    uncroppedDir = "uncropped/"
    cropDir = "cropped/"
    if not os.path.exists(uncroppedDir):
        print('No uncropped directory found.')
        return
    os.makedirs(os.path.dirname(cropDir), exist_ok=True)

    numUncropped = len(os.listdir(uncroppedDir))
    if numUncropped < 1:
        print('No images in uncropped directory.')
        return
    for i,infile in enumerate(os.listdir(uncroppedDir)):
        print(int(i/numUncropped*100), '%', end='\r')
        outfile = cropDir + infile
        try:
            im = Image.open(uncroppedDir+infile)
            width, height = im.size
            min_dim = min(width,height)
            box_dim = [width/2-min_dim/2, height/2-min_dim/2, width/2+min_dim/2, height/2+min_dim/2]
            im = im.resize(size, Image.ANTIALIAS,box=box_dim)
            im.save(outfile, "png")
        except IOError as e:
            print(e,"ERROR - failed to crop '%s'" % infile)
    print("100%")


CropImages()
