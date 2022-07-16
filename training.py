import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
process = 'Apply label to samples'
csv_files_path = "CSV_Files/"

def LoadSamples(csvDir="",classStartAt=0):
    print('Loading samples...')
    samples = None
    for i, indir in enumerate(os.listdir(csvDir)):
        for j, infile in enumerate(os.listdir(csvDir+"/"+indir)):
            try:
                #get the letter
                letter = infile[0]
                print('Class', letter)
                print('Class label:', i)
                newSamples = np.genfromtxt(csvDir +indir+"/"+ infile, delimiter=',')
                if samples is None:
                    samples = newSamples
                else:
                    samples = np.concatenate((samples, newSamples), axis=0)
            except IOError as e:
                print(e, "ERROR - failed to convert '%s'" % infile)

    print("Sample load complete!")
    print("Number of bytes:",samples.nbytes)
    return samples[:,1:], np.ravel(samples[:,:1])

def DimensionReduce(X_data,y_data):
    #johnson lindenstrauss lemma computation for ideal minimum number of dimensions
    epsilon = .15 #permitted error %
    min_dim = johnson_lindenstrauss_min_dim(22344, eps=epsilon)
    # 5% error needs 33150 dimensions retained
    # 10% error = 8583, 15% = 3936, 35% = 853
    print('Minimum dimensions to retain',epsilon,'error:',min_dim)
    #LDA -- aims to model based off difference between classes... supervised dimension reduction (DR) will likely outperform unsupervised DR such as PCA.
    pca = PCA(n_components = min_dim)
    print("Beginning dimension reduction...")
    X_reduced = pca.fit_transform(X_data, y_data)
    print("Finished dimension reduction!")
    return X_reduced

def KNN(X, Y, num_neighbors = 3):
    # 
    #
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))

args = sys.argv

saveNames=["X_train_reduced.npy","y_train.npy","X_validation.npy","y_validation.npy","X_test.npy","y_test.npy"]

if len(args) == 3 and args[1] == 'DimReduce' and args[2] == 'True' or args[2] == "False":
    samples, labels = LoadSamples(csvDir = csv_files_path, classStartAt = 0)
    #70% train. Stratify keeps class distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.3, random_state=42, stratify=labels) 
    #20% validation, 10% test
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=.33, random_state=42, stratify=y_test) 

    #dimension reduction
    X_train_reduced = DimensionReduce(X_train, y_train)
    saveReducedData = True if args[2] == "True" else False
    data = [X_train, y_train, X_validate, y_validate, X_test, y_test]
    if saveReducedData:
        for i in range(len(saveNames)):
            print("Saving",saveNames[i],"...")
            with open(saveNames[i], 'wb') as f:
                np.save(f, data[i])

elif len(args) < 3:
    print("Please provide:")
    print("1) model  ---- (KNN, ...)")
    print("2) inpath ---- (relative to working directory, ending in /)")
elif len(args) > 3:
    print("Too many arguments.")
    print("Please provide:")
    print("1) model  ---- (KNN, ...)")
    print("2) inpath ---- (relative to working directory, ending in /)")
elif args[1] == "KNN":
    data = [X_train,  y_train, X_validate, y_validate, X_test, y_test] = (None,None,None,None,None,None)
    for i, saveName in enumerate(saveNames):
        with open(saveName, 'rb') as f:
            data[i] = np.load(f)
    KNN(samples, labels, num_neighbors = 10)
