import numpy as np
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
    return samples


args = sys.argv
if len(args) < 2:
    ApplyLabelToSamples()
elif len(args) < 3:
    print(
        "Not enough arguments. Please provide: 1)inpath (relative to working directory, ending in /)")
else:
    ApplyLabelToSamples(inpath=args[1])