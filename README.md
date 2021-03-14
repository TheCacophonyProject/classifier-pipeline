# Overview

These scripts handle the data pre-processing, training, and execution of a Convolutional Neural Network based classifier
for thermal vision.

The output is a TensorFlow model that can identify thermal video clips of animals

# Scripts

### load.py
Processes CPTV clips and metadata, extracts the tagged frames (supplied by the metadata) and adds them to the hdf5 database.

The tracking algorithm tries to distinguish between animal tracks and false positives, but is not 100% reliable.  For this reason the output of the tracking algorithm should be checked by hand.


### build.py
Creates training, validation and testing datasets from database of clips & tracks.
Datasets contain frames and segments ( e.g. 45 frames).

Frames (important frames) are calculated by choosing frames with a mass ( the count of pixels that have been deemed an object by track extraction) between the lower and upper quartiles of a tracks mass distribution.

Frames are also checked to see if they are noisy frames.
This attempts to remove the following types of noisy frames:
- Frames where it has bad tracking e.g. in the middle of an animal
- Frames where only part of the animal is shown (maybe leaving the cameras field of vision)
- Frames where nothing is there but noise

Segments are calculated either by (depending on config):

Choosing random permutations of the important frames.
- Number of Segments (# of important frames - segment duration) // 9 segments are selected

or by choosing segment duration consecutive frames whose mass is above a certain amount        
- Number of Segments up to (# of frames - segment duration) // segment-frame-spacing

Datasets are split by camera and location ( to try and remove any bias that may occur from using a camera in multiple sets).

Some labels have low amounts of data so a single camera is split into 2 cameras e.g. Wallabies and Leoparidaes

### train.py
Trains a neural net using a provided test / train / validation dataset.

### classify.py
Uses a pre-trained model to identifying and classifying any animals in a CPTV file.

### evaluate.py
Evaluates the performance of a classify.py run and generates reports.

# Setup

Install the following prerequisites (tested with Ubuntu 18.0 and Python Python 3.6.9)
`apt-get install -y tzdata git python3 python3-dev python3-venv libcairo2-dev build-essential libgirepository1.0-dev libjpeg-dev python-cairo libhdf5-dev`
1. Create a virtual environment in python3 and install the necessary prerequisites </br>
`pip install -r requirements.txt`

2. Copy the classifier_Template.yaml to classifier.yaml and then edit this file with your own settings.   You will need to set up the paths for it work on your system. (Note: Currently these settings only apply to classify.py and extract.py)

3. Optionally install GPU support for tensorflow (note this requires additional [setup](
https://www.tensorflow.org/install/gpu))</br>
`pip install tensorflow-gpu`

4. MPEG4 output requires FFMPEG to be installed which can be found [here](https://www.ffmpeg.org/) On linux `apt-get install ffmpeg`.  On windows the installation path will need to be added to the system path.


5. Create a classifier configuration

Copy `classifier_TEMPLATE.yaml` to `classifier.yaml`. Edit.

# Usage

## Downloading the Dataset

CPTV files can be downloaded using the [cptv-downloader](https://github.com/TheCacophonyProject/cptv-download) tool.

## Training the Model

First download the CPTV files by running

`python cptv-download.py --user x --password x`

Next load the track files.  This can take some time

`python load.py all -v -p`

Now we can build the data set

`python build.py`

And finally train the model

`python train.py <build name>`

## Classifying animals within a CPTV File

A pre-trained model can be used to classify objects within a CPTV video

`python classify.py [cptv filename]`

This will generate a text file listing the animals identified, and create an MPEG preview file.

## Classification and Training Images

Single frame models use 48 x 48 frames to classify/train

![picture alt](readme/wallabyframe.png "Wallaby Frame")

Multi frame models use:
 * 25 frames arranged in a square for the red channel
 * Dots describing the centre of all tracked regions and lines connecting the dots for the green channel
 * Track Filtered frames overlaid where they have moved enough from the previous overlaid frame for the blue channel

![picture alt](readme/wallabymovement.png "Wallaby Movement")

## Release and Update

1. Create a release on GitHub (https://github.com/TheCacophonyProject/Project-Overview/wiki/Releases)

2. SSH into server

3. wget latest installer from GitHub

	`wget https://github.com/TheCacophonyProject/classifier-pipeline/releases/download/vx.x.x/classifier-pipeline_x.x.x_amd64.deb`

4. Install downloaded deb

	`sudo apt install ./classifier-pipeline_x.x.x_amd64.deb`

5. Make changes to config file if needed

	`/etc/cacophony/classifier.yaml`

6. Restart service

	`systemctl restart cacophony-processing.thermal@xx.service`

7. View logs

	`journalctl -u cacophony-processing.thermal@xx.service -f`

# Testing Classification and Tracking

## Generating Tests

- Tests can be generated from existing videos files on browse. The tests will contain
the tracks and tagged results as shown in browse by default.
- Test metadata will be saved to a yml file(tracking-tests.yml by default). This
may require manual editing to setup the tests if the original browse video did not track / classify
well
- Test CPTV files will be saved under out_dir and labelled as recordingid.cptv

`python generatetests.py out_dir Username Password <list of recording ids separated by a space>`

e.g.

`python generatetests.py test_clips Derek password123 12554 122232`

## Running Tests

- Once tests have been generated you can test your current tracking and model against thermal
- This will print out the results and also save a file called tracking-results.txt
- A default set of tracking tests is located in 'tests/tracking-tests.yml'
in order to run the clips they will need to be downloaded this can be done automatically
by adding a valid cacophny api user and password to trackingtest.py
`python trackingtest.py -t tests/tracking-tests.yml --user <User> --password <password>`


## Tracking results

Results for tests/tracking-tests.yml on tracking algorithms are located here
https://drive.google.com/drive/u/1/folders/1uGU9FhKaypadUVcIvItBZuZebZa_Z7MG
