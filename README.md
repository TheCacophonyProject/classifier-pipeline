
# Overview

These scripts handle the data pre-processing, training, and execution of a Convolutional Neural Network based classifier 
for thermal vision.

The output is a TensorFlow model that can identify 48x48 video clips centered on the object of interest.

# Scripts

### extract.py
Processes tagged CPTV files extracting targets of interest into track files used for training.

### build.py
Builds a data set from extracted track files.

### train.py
Trains a neural net using a provided test / train / validation dataset.

### classify.py
Uses a pre-trained model to identifying and classifying any animals in a CPTV file.

### evaluate.py
Evaluates the performance of a classify.py run and generates reports.

# Notebooks

None yet, but coming soon.

# Setup

Create a virtual environment and install the necessary prerequisits 

`pip install -r requirements.txt`

Optionally install GPU support for tensorflow (note this requires additional [setup](https://www.tensorflow.org/install/)

`pip install tensorflow-gpu`  

MPEG4 output requires FFMPEG to be installed which can be found [here](https://www.ffmpeg.org/).  

On windows the installation path will need to be added to the system path. 

# Usage

## Downloading the Dataset

CPTV files can be downloaded using the [cptv-downloader](https://github.com/TheCacophonyProject/cptv-download) tool.

## Training the Model

First download the CPTV files by running
 
`python cptv-download.py --user x --password x`
 
Next extract the track files.  This can take some time
 
`python extract.py all -v -p`

Now we can build the data set

`python build.py data`

And finally train the model

`python train.py -dataset=data -model-name=model --epochs=10`
 
## Classifying animals within a CPTV File

A pre-trained model can be used to classify objects within a CPTV video

`python classify.py [cptv filename] -p`

This will generate a text file listing the animals identified, and create an MPEG preview file.   `