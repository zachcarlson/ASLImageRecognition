import pandas as pd
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # no UI backend

def CSV_To_DF(path):
    print("Reading CSV in directory:", path)
    if not os.path.exists(path):
        print('No directory found.')
        return None
    files = os.listdir(path)
    if len(files) > 1:
        print('This script expects only one file per class directory. More than one file was detected. ')
        return None
    return pd.read_csv(path+files[0])

def DF_To_Image(df, image_num):
    df = df.iloc[:, 1:] #remove labels column
    image = df.iloc[image_num, :]
    image = image.to_numpy().reshape(224, 224)
    return image

def FeatureMeansPerClass(inpath):
    dirs = os.listdir(inpath)
    for d in dirs:
        df = CSV_To_DF(inpath+d+'/')
        if df is None:
            print('Error encountered, script terminated.')
            return
        feature_means = df.mean(axis=0,numeric_only=True)
        fig, ax = plt.subplots(1, 3, figsize=(15,10))
        plt.gray()
        ax[0].imshow(DF_To_Image(df, image_num=1), aspect="auto") #plot first image for each class
        ax[1].imshow(feature_means.to_numpy().reshape(224, 224), aspect="auto") #plot feature means image
        ax[2].plot(range(len(feature_means)),feature_means)  #plot feature means
        
        #customize appearance
        ax[2].set_xlabel('Features')
        ax[2].set_ylabel('Mean Value')
        ax[0].set_title(d+' Sample Image')
        ax[1].set_title(d+' Means Image')
        ax[2].set_title(d+' Class Feature Means')
        plt.gray() #show images as grayscale
        fig.savefig(d+'_FeatureMeans')  #savefig, don't show
        fig.clear()
        plt.close(fig)


args = sys.argv
if len(args) > 1:
    if args[1] == 'FeatureMeansPerClass':
        if len(args) == 3:
            FeatureMeansPerClass(args[2])
        else:
            print('FeatureMeansPerClass needs one argument -- the relative directory to traverse for subdirectory CSVs')
else:
    print('not enough arguments')
    print('Try:')
    print('FeatureMeansPerClass')



# # Class Balance Plot
# import os
# import pathlib
# import matplotlib.pyplot as plt
# from collections import OrderedDict
#
#
# #Collect subdir counts into array
# rootdir = "/Users/andrew/PycharmProjects/ASLImageRecognition/gray_frames"
# keys = []
# values = []
# for subdir, dirs, files in os.walk(rootdir):
#     initial_count = 0
#     for path in pathlib.Path(subdir).iterdir():
#         if path.is_file():
#             initial_count += 1
#         subdirClean = os.path.basename(os.path.normpath(subdir))
#         count = initial_count
#     keys.append(subdirClean)
#     values.append(count)
#
# # Convert to Dict
# countDict = {}
# for key, value in zip(keys, values):
#     if key != 'gray_frames':
#         countDict[key] = value
#
# countDict = OrderedDict(sorted(countDict.items()))
#
# print(countDict)
#
# # Visualize
# names = list(countDict.keys())
# values = list(countDict.values())
# mean =  sum(countDict.values()) / len(countDict)
#
#
# plt.bar(range(len(countDict)), values, tick_label=names)
# ax = plt.gca()
# ax.set_ylim([800, 1000])
# plt.xticks(rotation = 45)
# plt.axhline(y=mean, color ='r', linestyle = '-', label = ('Mean Samples: ', int(mean)))
# plt.legend()
# plt.show()