# American Sign Language Image Recognition Using Deep Learning.


## Project Overview:
This repository was created for our capstone project in DSCI 591+592, Capstone I & II, for the Master's in Data Science program at Drexel University.  The capstone spans two courses.  The first course focuses on data acquistion, pre-processing, and EDA.  The second course focuses on data training, modeling, and conclusions.  Both phases of the project are stored here.

Our project focused on American Sign Language (ASL).  The goal of this project was to use deep learning to accurately identify the different hand signs of the ASL alphabet.  We opted to create our own dataset for this project, as most ASL datasets available have incorrect hand signs.  We excluded any hand signs that require motion (such as the letters J and Z).  


## File Manifest: 
- `Folder /data` - Contains all small data files
  - `letter_means_df.csv` - Mean pixel values for each letter
- `Folder /documents` - Contains all reports as required for DSCI 591.
  - `Launch Report - Final Version.pdf` - PDF of first report for DSCI 591.
  - `DSCI 591 Data Acquisition and Pre-Processing Report.pdf` - PDF of second report for DSCI 591.
  - `DSCI 591 Exploratory Data Analytics Report.docx` - Third report for DSCI 591.
  - `Pitch.pptx` - Slides for first presentation in DSCI 591.
  - `Status Report.pptx` - Slides for second presentation in DSCI 591.
  - `DSCI 592 Predictive Modeling Report.pdf` - PDF for report in DSCI 592.
  - `G5-PMR-Slides.pptx` - Slides for first presentation in DSCI 592.
  - `G5-PMR-Slides-Final.pptx` - Slides for second presentation in DSCI 592.
  - `DSCI 592 PMR.docx` - PDF of major report for class (DSCI 592).
- `Folder /images` - Contains all relevant figures created for the project
  - `Folder /FeatureMeansSummary` - Contains all summary figures for each letter
  - `StandardDeviationOfAllClasses.png` - Composite image of standard deviations of each pixel mean value across each letter.  (See `ASLImageRecognition.ipynb` for more information.)
- `ASLImageRecognition.ipynb` - Jupyter Notebook used for some EDA visualizations
- `step01_VideoSplitPreProcess.py` - Python script to split raw video footage into series of frames
- `step02_CropPreProcess.py` - Python script to center and crop images 
- `step03_GrayscalePreProcess.py` - Python script to convert images to grayscale
- `step04_CSVConversionPreProcess.py` - Python script to convert grayscale images to CSV
- `step05_EDA.py` - Python script to produce some EDA visualizations


## Reason for Project:
Deep learning is a very powerful -- and equally dangerous -- tool.  Our goal is to use this technology to help more vulnerable populations.  Showing examples of data science being used for the betterment of society will inspire others to do so, rather than having newly graduated data scientists fall in the trap of exclusively working for private industry and rarely asking the question: who does this project serve and who does it harm?

The lack of diversity in technology is something we have been aware of for years.  While it may be less obvious in some cases, it became all too clear when working on this project.  As mentioned above, when researching datasets we soon came to realize most of the datasets were wrong.  The letter T hand sign was fully incorrect.  Either the thousands of individuals who downloaded the dataset or the hundreds of individuals who completed projects using it either didn't know ASL, didn't think to check if the ASL dataset was correct, or worse, didn't care to.  

It's here that a project changes from just a proof-of-concept to something more meaningful.  While we may not have answers now, we can at least start to ask "Who does this project serve?"  If the dataset is incorrect, how can algorithms that are built using it possibly serve those who are deaf or hard-of-hearing?  

An accurate ASL hand sign identification tool can serve both those who need this language and those who would like to learn.  Identifying the alphabet is the first step to identifying words, sentences, and translating ASL in real-time.  At the very least, this project -- and our newly created dataset -- can be used as a first step to implementing a tool that checks hand signs through a web, say, for an individual who is trying to learn ASL remotely.


## Team Members:

Our team consisted of the following individuals (alphabetized by last name): 

- Tyler Beard, tb3245@drexel.edu
- Adam Bennion, ab4657@drexel.edu
- Zach Carlson, zc378@drexel.edu
- Andrew Napolitano, asn65@drexel.edu



## Project Requirements
- Access to videos showing different ASL hand signs.  (For the time being, you can access ours [here](https://drive.google.com/drive/folders/1JfsDvx-Aq5ppHAef4y6wsKiAlfmYYm6V?usp=sharing), but these may not be availalbe to the public in the future.)
- IDE to run Python scripts
- Correct folder organization (see individual `.py` files and their functions for more information.)


## Python Requirements
- Python ≥ 3.8.
- The following modules and packages are required:
  - `csv`
  - `cv2`
  - `matplotlib`
  - `numpy`
  - `os`
  - `pandas`
  - `PIL.Image`, `PIL.ImageOps`
  - `sys`


## How to Execute Notebook: 

- Access the videos linked in **Project Requirements**, or create your own 30 second `.mp4` clips for each letter.  
- Ensure you have the correct folder organization.  For example, `step02_CropPreProcess.py` expects an "uncropped" directory in the working directory with images to be cropped.  
- Run each `.py` script.  After completing `step04_CSVConversionPreProcess.py`, it will be possible to run `ASLImageRecognition.ipynb`.

## Known Limitations of Project:
- **ASL is not universal**.  ASL differs from other English-speaking countries, with the existence of British Sign Language.  Because of this, this project can only serve those who know ASL.

- **Little testing**.  As this is a new dataset, it has not be tested by the data science community at large.  Because of this, we have no benchmark to compare our own results to nor the ability to have others assess the quality of our dataset/provide suggestions .  In time, however, this dataset will become available on Kaggle once it has been sufficiently pre-processed.

- **Skin color**.  While we improved the dataset so it could better serve those who are deaf and hard-of-hearing, we did not tackle the challenge of different skin color.  The dataset is composed of images from a single individual.  Adding additional images that feature many individuals with different skin color may lower the performance of our dataset by introducing more variety.  More importantly, it may improve the average performance across all skin colors and make the dataset (and subsequently created tools) more accessible.
