import os
import sys
from PIL import Image, ImageOps
import numpy as np
import csv


def ConvertImagesToCSV(imageDir="", csvDir="",label=0):
    
    #Can't find Image Directory
    if not os.path.exists(imageDir):
        print('No image directory found.')
        return

    #Make CSV Directory if it doesn't exist
    os.makedirs(os.path.dirname(csvDir), exist_ok=True)

    #Check there is at least one image in the Image Directory
    numImg = len(os.listdir(imageDir))
    if numImg < 1:
        print('No images in cropped_frames (i.e. RBG image) directory.')
        return

    #Clear CSV file if it already exists, to avoiding appending too much data
    csv_file_path = csvDir + f"{csvDir[-2:-1]}" + "_img_pixels.csv"
    if os.path.exists(csv_file_path):
        #clear file
        open(csv_file_path, 'w').close()

    #Create CSV File from Images
    for i, infile in enumerate(os.listdir(imageDir)):

        try:
            #Open Image
            img_file = Image.open(imageDir + infile)

            #get the letter for saving
            letter = infile[0]

            value = np.asarray(img_file.getdata(), dtype=int).reshape((img_file.size[1], img_file.size[0]))
            value = value.flatten()
            value = np.concatenate([[label], value])
            with open(csvDir + letter + "_img_pixels.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(value)

        except IOError as e:
            print(e, "ERROR - failed to convert '%s'" % infile)


def LaunchConvert(subdirectories=True, inpath="gray_frames/", outpath="CSV_Files/"):
    print("Converting...")

    if subdirectories:
        dirs = os.listdir(inpath)
        for i,d in enumerate(dirs):
            print("Converting image files in directory:", inpath + d + '/')
            ConvertImagesToCSV(inpath + d + '/', outpath + d + '/',label=i)
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
