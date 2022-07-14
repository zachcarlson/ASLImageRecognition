import numpy as np
from sklearn.neighbors import KNeighborsClassifier

process = 'Apply label to samples'
csv_files_path = "CSV_Files/"

def ApplyLabelToSamples(csvDir="",classStartAt=0):
    print('Applying class labels to samples...')
    samples = np.empty()
    classNum = classStartAt
    for i, infile in enumerate(os.listdir(csvDir)):
        try:
            #get the letter
            letter = infile[0]
            print('Class', letter)
            print('Class label:', classNum)
            newSamples = np.genfromtxt(csvDir + infile, delimiter=',')
            labels = np.full((newSamples.shape[0], 1), classNum)
            newSamples = np.append(newSamples, labels, axis=1)
            samples = np.concatenate(samples, newSamples, axis=0)
        except IOError as e:
            print(e, "ERROR - failed to convert '%s'" % infile)
        classNum += 1

    print("100%")
    return samples[:,0:-1], samples[:,-1]

def DimensionReduce():
    #PCA?
    #LDA?
    #simple down rez?
    #return newly reduced data
def KNN(X, Y, num_neighbors = 3):
    # 
    #
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))

args = sys.argv

if len(args) < 3:
    print("Not enough arguments.")
    print("Please provide:")
    print("1) model  ---- (KNN, ...)")
    print("2) inpath ---- (relative to working directory, ending in /)")
elif len(args) > 3:
    print("Too many arguments.")
    print("Please provide:")
    print("1) model  ---- (KNN, ...)")
    print("2) inpath ---- (relative to working directory, ending in /)")
else if args[1] == "KNN":
    samples, labels = ApplyLabelToSamples()
    #70% train. Stratify keeps class distribution balanced
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.3, random_state=42, stratify=labels) 
    #20% validation, 10% test. Stratify keeps class distribution balanced
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=.33, random_state=42, stratify=y_test) 
    samples = DimensionReduce(samples)
    KNN(samples, labels, num_neighbors = 10)
    ApplyLabelToSamples(inpath=args[2])
