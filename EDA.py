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

def FeatureMeansPerClass(inpath):
    dirs = os.listdir(inpath)
    for d in dirs:
        df = CSV_To_DF(inpath+d+'/')
        if df is None:
            print('Error encountered, script terminated.')
            return
        feature_means = df.mean(axis=0,numeric_only=True)
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        ax.plot(range(len(feature_means)),feature_means)
        ax.set_xlabel('Features')
        ax.set_ylabel('Mean Value')
        ax.title.set_text(d+' Class Feature Means')
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