#reference: https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio

import os
import sys
import PIL.Image as Image

def CropImages(uncroppedDir="", croppedDir="",dimension = (135,240)):

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
            im.save(outfile, "png")
        except IOError as e:
            print(e,"ERROR - failed to crop '%s'" % infile)
    print("100%")

def LaunchCrop(subdirectories = True, inpath = "uncropped/",outpath = "cropped/", dimensionX = 135, dimensionY = 240):
    print("Cropping...")
    size = dimensionX, dimensionY

    if subdirectories:
        dirs = os.listdir(inpath)
        for d in dirs:
            print("Cropping files in directory:", inpath+d+'/')
            CropImages(inpath+d+'/', outpath+d+'/', size)
    else:
        CropImages(inpath, outpath, size)


args = sys.argv
if len(args) < 2:
    LaunchCrop(subdirectories = True, inpath = "uncropped/",outpath = "cropped/", dimensionX = 135, dimensionY = 240)
elif len(args) < 6:
    print("Not enough arguments. Please provide: 1)subdirectories (boolean), 2)inpath (relative to working directory, ending in /), 3)outpath (ending in /), 4)dimensionX (integer), 5)dimensionY (integer)") 
else:
    LaunchCrop(subdirectories = True if args[1]=="True" else False, 
                inpath = args[2], outpath = args[3], dimensionX = int(args[4]), dimensionY = int(args[5]))
