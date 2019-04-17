# 0. Installation

Install:
* Python 3
* The Python requirements from  `requirements.txt`
* `ffmpeg`

# 1. Introduction
Instructions for performing various processes with the pipeline.

These assume you have already activated the correct virtual environment which has access to TensorFlow

These instructions are for Windows, on Linux `python` may have to be replaced with `python3`

# 2. Downloading new data

To download new data run

```commandline
python cptv-download.py [clip output folder] [username] [password] -v -x --limit=1000000 --ignore=['untagged'] -r 60
```

Which will download any new clips from the past 60 days.

A username and password is required to use the cacophony API.

Note: cptv-download is part of the cptv-download github project, not the classifier-pipeline project, so it will need to be installed separately.

# 3. Extracting Tracks

Track extraction processes the CPTV clips and extracts tracking frames around moving objects.

The tracking algorithm tries to distinguish between animal tracks and false positives, but is not 100% reliable.  For this reason the output of the tracking algorithm needs to be checked by hand.


## 3.0 Create a classifier configuration

Copy `classifier_TEMPLATE.yaml` to `classifier.yaml`. Edit.

## 3.1 Running the extractor

To run the extractor call

```commandline
python extract.py all -v -p
```

This process will be reasonably quick if only an small number of new clips need to be processed.  However if the entire dataset needs to be reprocessed it can take around 12 hours.

To speed up the process worker threads can be enabled.   This processes clips in parallel.  In general each process takes around 2 CPU cores, and 1GB of RAM.  So a quad core computer would run well with 2 workers.
To update the worker threads open 'classifier.yaml' and change the worker_threads parameter

### 3.1.1 Clean run

If a clean run is required delete or rename the output tracks folder.

The output file `dataset.hd5f` records which clips have been done, so removing this will cause all clips to be processed.

## 3.2 Checking the output

Sometimes invalid tracks will be generated.  Usually these are false positives, or perhaps a different animal that was in the video.

In the track output folder view the MPEG preview files to check if any contain invalid tracks.  Often this is obvious from the track preview MPEG thumbnails.

If it is unclear from the track preview, the clip preview MPEG often provides better context for understanding what the tracking algorithm has done.

The most common errors are

1/ A clip was miss-tagged.

Ideally the clip should be retagged through the webui and redownloaded.

2/ False-positive tracks are generated

For example picking up on a moving branch.  These can be fixed via step 3.3.

3/ A bird or small animal tagged as false-positive

Sometimes small birds are missed in the tagging process but picked up by the tracker.

## 3.3 Removing invalid tracks


To remove these poor quality tracks open 'hints.txt' and add the name of the cptv file to the list, along with the maximum number of tracks to use and the reason.

For example

```text
20171126-142109-akaroa10.cptv 1 "false-positive"
```

Which would tell the extractor to only use 1 track from the file `20171126-142109-akaroa10.cptv` when processing it.

Setting the max tracks to 0 will cause the clip to not be processed.

The reason field is not used, but is included for reference.

Once these changes have been made the track extractor will need to be rerun to apply the changes.

# 4 Building the dataset

To build the dataset run

```commandline
python build.py
```

This will take a minute or so.

By default only clips up to a certain date will be processed, so it may be necessary to edit the `build.py` and edit the line

```python
END_DATE = datetime(2018, 1, 31)
```

It can be quite useful to know that a dataset does not contain data past a certian date as any data after this date can safely be used for testing.

If new classes have been added, but do not have enough data the `EXCLUDED_LABELS` variable may also have to be updated. (see 4.1)

## 4.1 Adding a new class to the model

By default the validation set requires 300 segments taken from 10 different 'bins'.

A rule of thumb is that we require at least 2,000 segments, taken from at least 100 different tracks to enable classification on a specific class.

Training with less data that this often results is poor results, not only on the class in questions, but in training in general.

In the file `build.py` modify the line

```python
EXCLUDED_LABELS = ['insect','dog','human','mouse','rabbit', 'bird-kiwi']
```

So that the new class is no longer excluded.

When build.py is run it will output numbers for each class

```text

------------------------------------------------------------------------------------------
Class                Train                 Validation            Test
------------------------------------------------------------------------------------------
bird                 3610/673/152/3360.3  316/94/27/319.1      300/94/27/255.2
cat                  1325/61/21/3360.3    309/23/10/319.1      300/23/10/343.5
false-positive       7336/896/102/3360.3  339/46/10/319.1      300/46/10/277.4
hedgehog             4812/481/121/3360.3  321/46/17/319.1      300/46/17/283.4
possum               6118/516/83/3360.3   412/35/10/319.1      300/35/10/315.8
rat                  4075/554/19/3360.3   311/80/18/319.1      300/80/18/252.7
stoat                1745/286/19/3360.3   313/88/10/319.1      300/88/10/288.7
```

The numbers in each column represent the segments / tracks / bins / total weight for each class / dataset.

# 5. Training a new model.

Models can be trained via train.py.

Editing models and hyper-parameters needs to be done via code at this point.

In train.py edit the line

```python
model = ModelCRNN_HQ(labels=len(labels), enable_flow=True, l2_reg=0, keep_prob=0.2)
```

Additional hyper parameters can be added to the function.  For example `lstm_units=128`

Optical flow can we switched off and on via the parameter `enable_flow`, and the model type can be switched between the HQ version and the LQ version.

In general the HQ version performs better but takes 3 times longer to train.

To edit the model open model_crnn.py.  Much of the shared code between the HQ and LQ models is in the parent class, such as preprocessing.

The lines

```python
layer = self.conv_layer('thermal/1', layer, 32, [3, 3], conv_stride=2)
layer = self.conv_layer('thermal/2', layer, 48, [3, 3], conv_stride=2)
layer = self.conv_layer('thermal/3', layer, 64, [3, 3], conv_stride=2)
layer = self.conv_layer('thermal/4', layer, 64, [3, 3], conv_stride=2)
layer = self.conv_layer('thermal/5', layer, 64, [3, 3], conv_stride=1)
```
control the general architecture of the convolutional block where as the lines
```python
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['lstm_units'])
dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob, dtype=np.float32)
init_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_in[:, :, 0], self.state_in[:, :, 1])

lstm_outputs, lstm_states = tf.nn.dynamic_rnn(
    cell=dropout, inputs=out,
    initial_state=init_state,
    dtype=tf.float32,
    scope='lstm'
)
```

control the architecture of the Recurrent block.

Training is processed by

```python
model.train_model(epochs=30, run_name='HQ Novelty/V7 zeroed flow/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
```

Where the maximum number of epochs and the run name (used as the logging folder) are provided.

Each run should be given a unique run name.

When ready to train model simply run.

```commandline
python train.py
```

# 5.1 Using Tensorboard to monitor training

To view a model while training you can run Tensorboard from the command prompt.  Logs are placed in the
logs subdirectory of the base data directory.
It is often a good idea to group logs from related runs together in a folder.  For example 'c:/cac/logs/v030/optical flow test/'

```commandline
tensorboard --logdir=[LOG_FOLDER]`
```

Tensorboard will output a URL which you can open in browser

The most useful information are the metrics 'accuracy' and 'loss'.  The f1 scores also include useful per class information, as does the confusion matrix which is found under the images tab (generated very epoch).

Tensorboard will list all runs found in the provided log folder.  It can be therefore quite helpful to copy and paste a reference model from a previous set of tests, rename it to 'v0 reference' and use it to compare results against.  This can give an early indication of the performance of the model.

# 5.2 Testing a model

To test the performance of a model run the Model Test Notebook

```commandline
[classifier-pipeline folder]:> jupyter notebook
```

A browser should open.  Use the UI to select `Test Model.ipynb`

Edit the `model path` parameter to the model you want to test.

Note: you can run the debugging script on a model that is currently training by setting the path to a model in the
checkpoints folder.  e.g. c:/cac/checkpoints/model-most-recent'

Run each cell in the notebook.  Some of these cells will take a long time to run (more than 10 minutes).

The script will evaluate the per segment and per track classification performance.