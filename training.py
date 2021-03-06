#################################################
# readme
#
### npy files ####
# npy files are the ideal data format for readin gin data when running prediction models
# this is due to their binary format (FAR more efficient - much faster loading times)
#
###### KNN ######
# KNN needs dimension reduced data in order to avoid overfitting and have shorter processing time
# To generate the necessary npy files, run the following in command line:
#
# /bin/python3 training.py DimReduce
#
# Exact python argument syntax may vary per setup. This will output all necessary npy files to your execution directory.
#
# To execute KNN run the following from command line:
#
# /bin/python3 training.py KNN HyperTweak NoPlotUI 3 5 10 15 20 30 40 50
#
# The numbers to the far right each represent a different number of neighbors to consider in the KNN model.
# In otherwords, they reach represent a hyperparameter value in the hyperparameter tweak loop.
# 'HyperTweak' Means KNN will run a loop of varying number of neighbors as defined in the right-most arguments
# You do not need 'NoPlotUI' unless you are running WSL (Windows Subsystem for Linux).
#
###### CNN ######
#
# CNN does NOT need dimension reduced data but npy files are still the ideal input format.
# To generate the necessary npy files, run the following in command line:
#
# /bin/python3 training.py csvToNpy
#
# This will output all necessary npy files to your execution directory.
# 
# To execute CNN run the following from command line:
#
# /bin/python3 training.py CNN
#
#################################################

import itertools
import os

import matplotlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix , classification_report
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


import seaborn as sns
process = 'Apply label to samples'
csv_files_path = "CSV_Files/"
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
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

def DimensionReduce(X_train, y_train, X_validate, X_test):
    #johnson lindenstrauss lemma computation for ideal minimum number of dimensions
    epsilon = .15 #permitted error %
    min_dim = johnson_lindenstrauss_min_dim(22344, eps=epsilon)
    # 5% error needs 33150 dimensions retained
    # 10% error = 8583, 15% = 3936, 35% = 853
    print('Minimum dimensions to retain',epsilon,'error:',min_dim)
    pca = PCA(n_components = min_dim)
    print("Beginning dimension reduction...")
    X_train_reduced = pca.fit_transform(X_train, y_train)
    X_validate_reduced = pca.transform(X_validate)
    X_test_reduced = pca.transform(X_test)
    print("Finished dimension reduction!")
    return X_train_reduced, X_validate_reduced, X_test_reduced

def ConvertToNpy(saveNames, reduce = False):
    samples, labels = LoadSamples(csvDir=csv_files_path, classStartAt=0)
    # 70% train. Stratify keeps class distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.3, random_state=42,
                                                        stratify=labels)
    # 20% validation, 10% test
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=.33, random_state=42,
                                                              stratify=y_test)
    # dimension reduction
    if reduce:
        X_train, X_validate, X_test = DimensionReduce(X_train, y_train, X_validate, X_test)

    data = [X_train, y_train, X_validate, y_validate, X_test, y_test]
    for i in range(len(saveNames)):
        print("Saving", saveNames[i], "...")
        with open(saveNames[i], 'wb') as f:
            np.save(f, data[i])

def KNN(data, num_neighbors = 3, havePlotUI=True):
    #split data
    X_train,  y_train, X_validate, y_validate, X_test, y_test = data
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(X_train, y_train)
    val_score = neigh.score(X_validate,y_validate)
    print("Validation score:",val_score)

    y_pred = neigh.predict(X_test)
    y_true = y_test
    #print(len(letters))
    print("Test Classification Report:")
    print(classification_report(y_true, y_pred, target_names=letters, labels=labels))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, hyperparameter = "num_neighbors "+str(n), classes=letters,haveUI=havePlotUI)

def CNN(data, epochs=3, kernel_size=3, dropout=.25):
    X_train, y_train,X_validate, y_validate, X_test, y_test = data

    #reshape the dim reduced array to fit the CNN
    X_train = X_train.reshape(X_train.shape[0], 224, 224, 1)
    X_validate = X_validate.reshape(X_validate.shape[0],224,224,1)
    y_train = keras.utils.to_categorical(y_train, 24)
    y_validate = keras.utils.to_categorical(y_validate, 24)


    print("Defining CNN")
    # Defining the Convolutional Neural Network

    cnn_model = Sequential()
    cnn_model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 1), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(dropout))
    cnn_model.add(Conv2D(32, kernel_size=kernel_size, activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(dropout))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(24, activation='softmax'))

    print("Compiling CNN")
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training CNN")
    cnn_model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=epochs)


def plot_confusion_matrix(cm, classes, hyperparameter,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, haveUI=True):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if haveUI:
        plt.show()
    else:
        plt.savefig("KNN "+hyperparameter)  #savefig, don't show
    plt.figure().clear()
    plt.close()


args = sys.argv

saveNamesKNN=["X_train_reduced.npy","y_train_reduced.npy","X_validation_reduced.npy","y_validation_reduced.npy","X_test_reduced.npy","y_test_reduced.npy"]
saveNamesCNN=["X_train.npy","y_train.npy","X_validation.npy","y_validation.npy","X_test.npy","y_test.npy"]

if('training.py' in args[0]):
    args = args[1:]

if len(args) == 1 and args[0] == "KNN":
    data = [X_train, y_train, X_validate, y_validate, X_test, y_test] = [None, None, None, None, None, None]
    print('Looking for data')
    for i, saveName in enumerate(saveNamesKNN):
        with open(saveName, 'rb') as f:
            data[i] = np.load(f)
    n = 10
    print('starting KNN with',n,"neighbors...")
    KNN(data, num_neighbors=n)

if len(args) == 2 and args[0] == "CNN":
    data = [X_train, y_train, X_validate, y_validate, X_test, y_test] = [None, None, None, None, None, None]
    print('Looking for data')


    for i, saveName in enumerate(saveNamesCNN):
        with open(args[1] + saveName, 'rb') as f:
            data[i] = np.load(f)
    print('starting CNN')
    CNN(data)

elif len(args) == 2 and args[0] == "KNN" and args[1] == "NoPlotUI":
    matplotlib.use('Agg') # no UI backend
    data = [X_train, y_train, X_validate, y_validate, X_test, y_test] = [None, None, None, None, None, None]
    print('Looking for data')
    for i, saveName in enumerate(saveNamesKNN):
        with open(saveName, 'rb') as f:
            data[i] = np.load(f)
    n = 10
    print('starting KNN with',n,"neighbors...")
    KNN(data, num_neighbors=n, havePlotUI=False)
elif len(args) > 3 and args[0] == "KNN" and args[1] == "HyperTweak" and args[2] == "NoPlotUI":
    matplotlib.use('Agg') # no UI backend
    n_neighbors_list = [int(n) for n in args[3:]]
    data = [X_train, y_train, X_validate, y_validate, X_test, y_test] = [None, None, None, None, None, None]
    print('Looking for data')
    for i, saveName in enumerate(saveNamesKNN):
        with open(saveName, 'rb') as f:
            data[i] = np.load(f)
    print("Running KNN hyperparameter tweaking loop...")
    for n in n_neighbors_list:
        print('starting KNN with',n,"neighbors...")
        KNN(data, saveNamesKNN, num_neighbors=n, havePlotUI=False)

elif len(args) == 1 and args[0] == 'csvToNpy':
        ConvertToNpy(saveNamesCNN)

elif len(args) == 1 and args[0] == 'DimReduce':
        ConvertToNpy(saveNamesKNN,reduce=True)
else:
    print("Invalid arguments. Please view training.py for argument options.")
