import os
import sys
from PIL import Image, ImageOps
import numpy as np
import csv


def ConvertImagesToCSV(imageDir="", csvDir=""):
    if not os.path.exists(imageDir):
        print('No image directory found.')
        return
    os.makedirs(os.path.dirname(csvDir), exist_ok=True)

    numImg = len(os.listdir(imageDir))
    if numImg < 1:
        print('No images in cropped_frames (i.e. RBG image) directory.')
        return

    for i, infile in enumerate(os.listdir(imageDir)):

        try:
            img_file = Image.open(imageDir + infile)

            #get the letter
            letter = infile[0]
            print(letter)

            value = np.asarray(img_file.getdata(), dtype=int).reshape((img_file.size[1], img_file.size[0]))
            value = value.flatten()
            value = np.concatenate([[letter], value])
            print(value)
            with open(csvDir + letter + "_img_pixels.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(value)
        except IOError as e:
            print(e, "ERROR - failed to convert '%s'" % infile)

    print("100%")


def LaunchConvert(subdirectories=True, inpath="gray_frames/", outpath="CSV_Files/"):
    print("Converting...")

    if subdirectories:
        dirs = os.listdir(inpath)
        for d in dirs:
            print("Converting image files in directory:", inpath + d + '/')
            ConvertImagesToCSV(inpath + d + '/', outpath + d + '/')
    else:
        ConvertImagesToCSV(inpath, outpath)


args = sys.argv
if len(args) < 2:
    LaunchConvert(subdirectories=True, inpath="gray_frames/", outpath="CSV_Files/")
elif len(args) < 4:
    print(
        "Not enough arguments. Please provide: 1)subdirectories (boolean), 2)inpath (relative to working directory, ending in /), 3)outpath (ending in /)")
else:
    LaunchConvert(subdirectories=True if args[1] == "True" else False,
                  inpath=args[2], outpath=args[3])
