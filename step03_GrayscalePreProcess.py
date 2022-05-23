#reference: https://bit.ly/3vku3bt

import os
import sys
from PIL import Image, ImageOps

def ConvertImagesToGrayscale(colorDir="", grayDir=""):
    if not os.path.exists(colorDir):
        print('No RBG image directory found.')
        return
    os.makedirs(os.path.dirname(grayDir), exist_ok=True)

    numColored = len(os.listdir(colorDir))
    if numColored < 1:
        print('No images in cropped_frames (i.e. RBG image) directory.')
        return

    for i,infile in enumerate(os.listdir(colorDir)):
        print(int(i/numColored*100), '%', end='\r')
        outfile = grayDir + infile

        try:
            im = Image.open(colorDir+infile)
            gray_im = ImageOps.grayscale(im)
            gray_im.save(outfile, "png")

        except IOError as e:
            print(e,"ERROR - failed to convert '%s'" % infile)
    print("100%")

def LaunchConvert(subdirectories = True, inpath = "cropped_frames/",outpath = "gray_frames/"):
    print("Converting...")

    if subdirectories:
        dirs = os.listdir(inpath)
        for d in dirs:
            print("Converting image files in directory:", inpath+d+'/')
            ConvertImagesToGrayscale(inpath+d+'/', outpath+d+'/')
    else:
        ConvertImagesToGrayscale(inpath, outpath)

args = sys.argv
if len(args) < 2:
    LaunchConvert(subdirectories = True, inpath = "cropped_frames/",outpath = "gray_frames/")
elif len(args) < 4:
    print("Not enough arguments. Please provide: 1)subdirectories (boolean), 2)inpath (relative to working directory, ending in /), 3)outpath (ending in /)") 
else:
    LaunchConvert(subdirectories = True if args[1]=="True" else False, 
                inpath = args[2], outpath = args[3])