import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
process = 'Apply label to samples'
csv_files_path = "CSV_Files/"

def ApplyLabelToSamples(csvDir="",classStartAt=0):
    print('Applying class labels to samples...')
    samples = None
    classNum = classStartAt
    for i, indir in enumerate(os.listdir(csvDir)):
        for j, infile in enumerate(os.listdir(csvDir+"/"+indir)):
            try:
                #get the letter
                letter = infile[0]
                print('Class', letter)
                print('Class label:', classNum)
                newSamples = np.genfromtxt(csvDir +indir+"/"+ infile, delimiter=',')
                labels = np.full((newSamples.shape[0], 1), classNum)
                newSamples = np.append(newSamples, labels, axis=1)
                if samples is None:
                    samples = newSamples
                else:
                    samples = np.concatenate((samples, newSamples), axis=0)
            except IOError as e:
                print(e, "ERROR - failed to convert '%s'" % infile)
            classNum += 1

    print("Label application complete!")
    print("Number of bytes:",samples.nbytes)
    return samples[:,0:-1], samples[:,-1]

def DimensionReduce(X_data,y_data):
    #johnson lindenstrauss lemma computation for ideal minimum number of dimensions
    epsilon = .15 #permitted error %
    min_dim = johnson_lindenstrauss_min_dim(22344, eps=epsilon)
    # 5% error needs 33150 dimensions retained
    # 10% error = 8583, 15% = 3936, 35% = 853
    print('Minimum dimensions to retain',epsilon,'error:',min_dim)
    #LDA -- aims to model based off difference between classes... supervised dimension reduction (DR) will likely outperform unsupervised DR such as PCA.
    lda = LinearDiscriminantAnalysis(n_components = min_dim)
    print("Beginning dimension reduction...")
    X_reduced = lda.fit_transform(X_data, y_data)
    print("Finished dimension reduction!")
    return X_reduced

def KNN(X, Y, num_neighbors = 3):
    # 
    #
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))

args = sys.argv

if len(args) == 3 and args[1] == 'DimReduce' and args[2] == 'True' or args[2] == "False":
    samples, labels = ApplyLabelToSamples(csvDir = csv_files_path, classStartAt = 0)
    saveCoalescedData = True
    if saveCoalescedData:
        print("Saving coalesced samples...")
        np.savetxt("combined_samples",samples,delimiter=",")
        print("Saving coalesced labels...")
        np.savetxt("combined_labels",labels,delimiter=",")
        print("Save complete!")
    #70% train. Stratify keeps class distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.3, random_state=42, stratify=labels) 
    #20% validation, 10% test
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=.33, random_state=42, stratify=y_test) 

    #dimension reduction
    X_train_reduced = DimensionReduce(X_train, y_train)
    saveReducedData = True if args[2] == "True" else False
    saveNames=["X_train_reduced.csv","y_train.csv","X_validation.csv","y_validation.csv","X_test.csv","y_test.csv"]
    data = [X_train, y_train, X_validate, y_validate, X_test, y_test]
    if saveReducedData:
        for i in range(len(saveNames)):
            print("Saving",saveName[i],"...")
            np.savetxt(saveName[i],data[i],delimiter=",")
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
    KNN(samples, labels, num_neighbors = 10)
