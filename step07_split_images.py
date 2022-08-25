#to execute from command line: /bin/python3 split_images.py

# split images into train, val, test directories, with subdirectories of each class

import os
import sys
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

def MoveImage(open_path,save_path):
    try:
        img_file = Image.open(open_path)
        img_file.save(save_path)

    except IOError as e:
        print(e, "ERROR - failed to process '%s'" % open_path)


def SplitImages(inpath, outpath, split_outpaths):
    #get total count of files:
    ttl_samples = 0
    dirs = os.listdir(inpath)
    for d in dirs:
        ttl_samples += len([name for name in os.listdir(os.path.join(inpath, d)) if os.path.isfile(os.path.join(os.path.join(inpath, d), name))])

    #Make directories if they do not exist
    dirs = os.listdir(inpath)
    for d in dirs:
        os.makedirs(os.path.join(os.path.join(outpath,split_outpaths[0]),d), exist_ok=True)
        os.makedirs(os.path.join(os.path.join(outpath,split_outpaths[1]),d), exist_ok=True)
        os.makedirs(os.path.join(os.path.join(outpath,split_outpaths[2]),d), exist_ok=True)
    #train, val, test split of sample indices
    sample_indices = list(range(ttl_samples))
    print('total samples:',len(sample_indices))
    # 70% train. Stratify keeps class distribution balanced
    X_train, X_test = train_test_split(sample_indices, test_size=.3, random_state=42)
    # 20% validation, 10% test
    X_val, X_test = train_test_split(X_test, test_size=.33, random_state=42)

    X_train.sort()
    X_val.sort()
    X_test.sort()
    xtrain_i = 0
    xtest_i = 0
    xval_i = 0
    i = 0
    for d in dirs:
        for name in os.listdir(os.path.join(inpath, d)):
            if os.path.isfile(os.path.join(os.path.join(inpath, d), name)):
                if i == X_train[xtrain_i]:
                    MoveImage(os.path.join(os.path.join(inpath, d), name),os.path.join(os.path.join(os.path.join(outpath, split_outpaths[0]), d), name))
                    xtrain_i += 1
                elif i == X_val[xval_i]:
                    MoveImage(os.path.join(os.path.join(inpath, d), name),os.path.join(os.path.join(os.path.join(outpath, split_outpaths[1]), d), name))
                    xval_i += 1
                elif i == X_test[xtest_i]:
                    MoveImage( os.path.join(os.path.join(inpath, d), name),
                                os.path.join(os.path.join(os.path.join(outpath, split_outpaths[2]), d), name)
                                )
                    xtest_i += 1
                i += 1

SplitImages(inpath="cropped_frames/", outpath='split_images/', split_outpaths=['train_images/','val_images/','test_images/'])