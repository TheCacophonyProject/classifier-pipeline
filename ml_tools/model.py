import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os.path
import pickle
import math
import logging
import time
import json
import io
from collections import namedtuple
from sklearn import metrics

from ml_tools import tools
from ml_tools import visualise

# folder to save model while it's training.  Make sure this isn't on a dropbox folder and it will cause a crash.
CHECKPOINT_FOLDER = "c:\cac\checkpoints"

class Model:
    """ Defines a deep learning model """

    MODEL_NAME = "abstract model"
    MODEL_DESCRIPTION = ""

    def __init__(self, session=None):

        self.name = "model"
        self.session = session or tools.get_session()

        # datasets
        self.datasets = namedtuple('Datasets', 'train, validation, test')

        # ------------------------------------------------------
        # placeholders, used to feed data to the model
        # ------------------------------------------------------

        self.X = None
        self.y = None
        self.keep_prob = None
        self.is_training = None
        self.global_step = None

        # ------------------------------------------------------
        # tensflow nodes used to evaluate
        # ------------------------------------------------------

        # prediction for each class(probability distribution)
        self.prediction = None
        # accuracy of batch
        self.accuracy = None
        # total loss of batch
        self.loss = None
        # training operation
        self.train_op = None

        self.state_in = None
        self.state_out = None
        self.normalise = None

        # number of samples to use when evaluating the model, 1000 works well but is a bit slow,
        # 100 should give results to within a few percent.
        self.eval_samples = 500

        # number of samples to use when generating the model report,
        # atleast 1000 is recommended for a good representation
        self.report_samples = 2000

        # how often to do an evaluation + print
        self.print_every = 6000

        # restore best weights found during training rather than the most recently one.
        self.use_best_weights = True

        # the score this model got on it's final evaluation
        self.eval_score = None

        # our current global step
        self.step = 0

        # enabled parallel loading and training on data (much faster)
        self.enable_async_loading = True

        # folder to write tensorboard logs to
        self.log_dir = './logs'

        # number of frames per segment during training
        self.training_segment_frames = 27
        # number of frames per segment during testing
        self.testing_segment_frames = 27

        # dictionary containing current hyper parameters
        self.params = {
            # augmentation
            'augmentation': True,
            'filter_threshold': 20,
            'filter_noise': 1.0,
            'thermal_threshold': 10,
            'scale_frequency': 0.5,
            # dropout
            'keep_prob': 0.5,
            # training
            'batch_size': 16
        }

        """ List of labels this model can classifiy. """
        self.labels = []

        # used for tensorboard
        self.writer_train = None
        self.writer_val = None
        self.merged_summary = None

    def import_dataset(self, dataset_filename, ignore_labels=None):
        """
        Import dataset.
        :param dataset_filename: path and filename of the dataset
        :param ignore_labels: (optional) these labels will be removed from the dataset.
        :return:
        """
        datasets = pickle.load(open(dataset_filename,'rb'))
        self.datasets.train, self.datasets.validation, self.datasets.test = datasets

        # augmentation really helps with reducing over-fitting, but test set should be fixed so we don't apply it there.
        self.datasets.train.enable_augmentation = self.params['augmentation']
        self.datasets.train.scale_frequency = self.params['scale_frequency']
        self.datasets.validation.enable_augmentation = False
        self.datasets.test.enable_augmentation = False
        for dataset in datasets:
            dataset.filter_threshold = self.params['filter_threshold']
            dataset.filtered_noise = self.params['filter_noise']
            dataset.thermal_threshold = self.params['thermal_threshold']

            if ignore_labels:
                for label in ignore_labels:
                    dataset.remove_label(label)

        self.labels = self.datasets.train.labels.copy()

        logging.info("Training segments: {0:.1f}k".format(self.datasets.train.rows/1000))
        logging.info("Validation segments: {0:.1f}k".format(self.datasets.validation.rows/1000))
        logging.info("Test segments: {0:.1f}k".format(self.datasets.test.rows/1000))
        logging.info("Labels: {}".format(self.datasets.train.labels))

        label_strings = [",".join(self.datasets.train.labels),
                         ",".join(self.datasets.validation.labels),
                         ",".join(self.datasets.test.labels)]
        assert len(set(label_strings)) == 1, 'dataset labels do not match.'

    @property
    def batch_size(self):
        return self.params['batch_size']

    @property
    def steps_per_epoch(self):
        """ Number of steps per epoch"""
        return self.rows // self.batch_size

    @property
    def rows(self):
        """ Number of examples in training sest"""
        return self.datasets.train.rows

    @property
    def hyperparams_string(self):
        """ Returns list of hyperparameters as a string. """
        return "\n".join(["{}={}".format(param, value) for param, value in self.params.items()])

    def eval_batch(self, batch_X, batch_y, writer=None, include_detailed_summary=False):
        """
        Evaluates the accuracy on a batch of frames.  If the batch is too large it will be broken into smaller parts.
        :param batch_X:
        :param batch_y:
        :param writer: (optional) if given a summary will be written to this summary writer
        :param include_detailed_summary: (optional) includes detailed information on weights, paremeters etc
        :return:
        """

        total_samples = batch_X.shape[0]
        batches = (total_samples // self.batch_size) + 1
        score = 0
        loss = 0
        summary = None

        for i in range(batches):
            Xm = batch_X[i*self.batch_size:(i+1)*self.batch_size]
            ym = batch_y[i*self.batch_size:(i+1)*self.batch_size]
            if len(Xm) == 0:
                continue
            samples = Xm.shape[0]

            # only calculate summary on first batch, as otherwise we could get many summaries for a single timestep.
            # a better solution would be to accumulate and average the summaries which tensorflow sort of has support for.
            feed_dict = self.get_feed_dict(Xm, ym)
            if include_detailed_summary and writer is not None and i == 0:
                summary, acc, ls = self.session.run([self.merged_summary, self.accuracy, self.loss], feed_dict=feed_dict)
            else:
                acc, ls = self.session.run([self.accuracy, self.loss], feed_dict=feed_dict)

            score += samples * acc
            loss += samples * ls

        batch_accuracy = score / total_samples
        batch_loss = loss / total_samples

        if writer is not None:
            if include_detailed_summary:
                writer.add_summary(summary, global_step=self.step)
            # we manually write out the aggretated values as we want to know the total score, not just the per batch
            # scores.
            self.log_scalar('metric/accuracy', batch_accuracy, writer=writer)
            self.log_scalar('metric/error', 1-batch_accuracy, writer=writer)
            self.log_scalar('metric/loss', batch_loss, writer=writer)

        return batch_accuracy, batch_loss

    def get_feed_dict(self, X, y, is_training=False):
        """ returns a feed dictionary for TensorFlow placeholders. """
        return {
            self.X: X[:, 0:self.training_segment_frames + 1],          # limit number of frames per segment passed to trainer
            self.y: y,
            self.keep_prob: self.params['keep_prob'] if is_training else 1.0,
            self.is_training: is_training,
            self.global_step: self.step
        }

    def classify_batch(self, batch_X):
        """
        :param batch_X: input batch of shape [n,frames,h,w,channels]
        :return: list of probs for each examples
        """
        """ Classifies all segments in the given batch. """

        total_samples = batch_X.shape[0]
        batches = (total_samples // self.batch_size) + 1

        predictions = []

        for i in range(batches):
            Xm = batch_X[i*self.batch_size:(i+1)*self.batch_size]
            if len(Xm) == 0:
                continue
            probs = self.session.run([self.prediction], feed_dict={self.X: Xm})[0]
            for j in range(len(Xm)):
                predictions.append(probs[j,:])

        return predictions

    def eval_model(self, writer=None):
        """ Evaluates the model on the test set. """
        print("-"*60)
        self.datasets.test.load_all()
        test_accuracy, _ = self.eval_batch(self.datasets.test.X, self.datasets.test.y, writer=writer)
        print("Test Accuracy {0:.2f}% (error {1:.2f}%)".format(test_accuracy*100,(1.0-test_accuracy)*100))
        return test_accuracy

    def benchmark_model(self):
        """
        Runs a benchmark on the model, getting runtime statistics and saving them to the tensorboard log folder.
        """

        assert self.writer_train, 'Must init tensorboard writers before benchmarking.'

        # we actually train on this batch, which shouldn't hurt.
        # the reason this is necessary is we need to get the true performance cost of the back-prob.
        X, y = self.datasets.train.next_batch(self.batch_size)
        feed_dict = self.get_feed_dict(X, y, True)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # first run takes a while, so run this just to build the graph, then run again to get real performance.
        _, = self.session.run([self.train_op], feed_dict=feed_dict)

        _, = self.session.run([self.train_op], feed_dict=feed_dict,
                                            options=run_options,
                                            run_metadata=run_metadata)

        # write out hyper-params
        hp_summary_op = tf.summary.text('hyperparams', tf.convert_to_tensor(str(self.hyperparams_string)))
        hp_summary = self.session.run(hp_summary_op)
        for writer in [self.writer_train, self.writer_val]:
            writer.add_run_metadata(run_metadata, 'benchmark')
            writer.add_summary(hp_summary)


    def setup_summary_writers(self, run_name):
        """
        Initialises tensorboard log writers
        :return:
        """
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer_train = tf.summary.FileWriter(os.path.join(self.log_dir, run_name + '/train'), graph=self.session.graph)
        self.writer_val = tf.summary.FileWriter(os.path.join(self.log_dir, run_name + '/val'), graph=self.session.graph)
        merged = tf.summary.merge_all()
        self.merged_summary = merged


    def log_scalar(self, tag, value, writer=None):
        """
        Writes a scalar to summary writer.
        :param tag: tag to use
        :param value: value to write
        :param writer: (optional) summary writer.  Defaults to writer_val
        """
        if writer is None:
            writer = self.writer_val
        writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]),
            global_step=self.step)

    def log_image(self, tag, image, writer=None):
        """
        :param tag: tag to use
        :param image: image to write
        :param writer: (optional) summary writer.  Defaults to writer_val
        :return:
        """
        """Logs a list of images."""
        if writer is None:
            writer = self.writer_val

        # Write the image to a string
        s = io.BytesIO()
        plt.imsave(s, image, format='png')

        # Create an Image object
        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.shape[0],
                                   width=image.shape[1])
        # Create a Summary value
        im_summary = tf.Summary.Value(tag=tag, image=img_summary)

        # Create and write Summary
        summary = tf.Summary(value=[im_summary])
        writer.add_summary(summary, self.step)

    def generate_report(self):
        """
        Logs some important information to the tensorflow summary writer, such as confusion matrix, and f1 scores.
        :return:
        """

        # get some examples for evaluation
        examples, true_classess = self.datasets.validation.next_batch(self.report_samples)
        predictions = self.classify_batch(examples)
        predictions = [np.argmax(prediction) for prediction in predictions]

        pred_label = [self.labels[x] for x in predictions]
        true_label = [self.labels[x] for x in true_classess]

        cm = tools.get_confusion_matrix(pred_class=predictions, true_class=true_classess, classes=self.labels)
        f1_scores = metrics.f1_score(y_true=true_label, y_pred=pred_label, labels=self.labels, average=None)

        fig = visualise.plot_confusion_matrix(cm, self.labels)
        fig.canvas.draw()
        data = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        img = np.fromstring(data, dtype=np.uint8).reshape(nrows, ncols, 3)
        self.log_image("confusion_matrix", img)
        plt.close()

        for label_number, label in enumerate(self.labels):
            self.log_scalar("f1/" + label, f1_scores[label_number])

        errors = correct = 0
        for pred, true in zip(pred_label, true_label):
            if pred != true:
                errors += 1
            else:
                correct += 1

        accuracy = correct / (correct + errors)

        return accuracy, f1_scores

    def train_model(self, epochs=10.0, run_name=None):
        """
        Trains model given number of epocs.  Uses session 'sess'
        :param epochs: number of epochs to train for
        :param run_name: name of this run, used to create logging folder.
        :return:
        """

        assert self.datasets.train, 'Training dataset found, must call import_dataset before training.'
        assert self.train_op, 'Training operation has not been assigned.'

        if self.enable_async_loading:
            self.start_async_load()


        iterations = int(math.ceil(epochs * self.rows / self.batch_size))
        if run_name is None:
            run_name = self.MODEL_NAME
        eval_time = 0
        train_time = 0
        prep_time = 0
        best_step = 0
        examples_since_print = 0
        last_epoch_save = 0
        best_report_acc = 0

        # Run the initializer
        init = tf.global_variables_initializer()
        self.session.run(init)

        # setup writers and run a quick benchmark
        print("Initialising summary writers at {}.".format(os.path.join(self.log_dir, run_name)))
        self.setup_summary_writers(run_name)
        print("Starting benchmark.")
        self.benchmark_model()
        print("Training...")

        for i in range(iterations):

            self.step = i

            # get a new batch
            start = time.time()
            batch = self.datasets.train.next_batch(self.batch_size)
            prep_time += time.time()-start

            # evaluate every so often
            if examples_since_print >= self.print_every or (i == iterations-1):

                start = time.time()

                val_batch = self.datasets.validation.next_batch(self.eval_samples)
                train_batch = self.datasets.train.next_batch(self.eval_samples)

                train_accuracy, train_loss = self.eval_batch(
                    train_batch[0], train_batch[1],
                    writer=self.writer_train)
                val_accuracy, val_loss = self.eval_batch(
                    val_batch[0], val_batch[1],
                    writer=self.writer_val, include_detailed_summary=True)

                epoch = (self.batch_size * i) / self.rows

                eval_time += time.time()-start

                steps_remaining = (iterations - i)
                step_time = prep_time + train_time + eval_time
                eta = (steps_remaining * step_time / (examples_since_print / self.batch_size)) / 60

                print('[epoch={0:.2f}] step {1}, training={2:.1f}%/{3:.3f} validation={4:.1f}%/{5:.3f} [times:{6:.1f}ms,{7:.1f}ms,{8:.1f}ms] eta {9:.1f} min'.format(
                    epoch, i, train_accuracy*100, train_loss * 10, val_accuracy*100, val_loss * 10,
                    1000 * prep_time / examples_since_print,
                    1000 * train_time / examples_since_print,
                    1000 * eval_time / examples_since_print,
                    eta
                ))

                # create a save point
                self.save(os.path.join(CHECKPOINT_FOLDER, "training-most-recent.sav"))

                # save at epochs
                if int(epoch) > last_epoch_save:
                    print("Epoch report")
                    acc, f1 = self.generate_report()
                    print("results: {:.1f} {}".format(acc*100,["{:.1f}".format(x*100) for x in f1]))
                    print('Save epoch reference')
                    self.eval_score = acc
                    self.save(os.path.join(CHECKPOINT_FOLDER, "training-epoch-{:02d}.sav".format(int(epoch))))
                    last_epoch_save = int(epoch)

                    if acc > best_report_acc:
                        print('Save best model')
                        self.save(os.path.join(CHECKPOINT_FOLDER, "training-best.sav"))
                        best_report_acc = acc
                        best_step = i

                eval_time = 0
                train_time = 0
                prep_time = 0
                examples_since_print = 0

            # train on this batch
            start = time.time()
            feed_dict = self.get_feed_dict(batch[0], batch[1], is_training=True)
            _ = self.session.run([self.train_op], feed_dict=feed_dict)
            train_time += time.time()-start

            examples_since_print += self.batch_size

        # restore previous best
        if self.use_best_weights:
            print("Using model from step", best_step)
            self.load(os.path.join(CHECKPOINT_FOLDER, "training-best.sav"))

        self.eval_score = self.eval_model()

        summary_op = tf.summary.text('metric/final_score', tf.convert_to_tensor(str(self.eval_score)))
        summary = self.session.run(summary_op)
        self.writer_val.add_summary(summary)

        if self.enable_async_loading:
            self.stop_async()

    def start_async_load(self):
        # make sure the workers load the correct number of frames.
        self.datasets.train.segment_width = self.testing_segment_frames
        self.datasets.validation.segment_width = self.testing_segment_frames
        self.datasets.train.start_async_load(64)
        self.datasets.validation.start_async_load(64)

    def stop_async(self):
        self.datasets.train.stop_async_load()
        self.datasets.validation.stop_async_load()

    def close(self):
        """ 
        Cleans up memory used by model by closing any open sessions or aync loaders.
        :return: 
        """
        self.session.close()
        self.stop_async()

    def save(self, filename=None):
        """
        Saves the model and current parameters.
        :param filename: (optional) filename to save.  defaults to an autogenerated name.
        """

        if filename is None:
            score_part = "{:.3f}".format(self.eval_score)
            while len(score_part) < 3:
                score_part = score_part + "0"
            filename = os.path.join("./models/", self.MODEL_NAME + '-' + score_part)

        saver = tf.train.Saver()
        saver.save(self.session, filename)

        # save some additional data
        model_stats = {}
        model_stats['name'] = self.MODEL_NAME
        model_stats['description'] = self.MODEL_DESCRIPTION
        model_stats['notes'] = ""
        model_stats['labels'] = self.labels
        model_stats['score'] = self.eval_score
        model_stats['hyperparams'] = self.params

        json.dump(model_stats, open(filename+ ".txt", 'w'), indent=4)

    def load(self, filename):
        """ Loads model and parameters from file. """

        saver = tf.train.import_meta_graph(filename+'.meta', clear_devices=True)
        saver.restore(self.session, filename)

        # get additional hyper parameters
        stats = json.load(open(filename+".txt",'r'))

        self.MODEL_NAME = stats['name']
        self.MODEL_DESCRIPTION = stats['description']
        self.labels = stats['labels']
        self.eval_score = stats['score']
        self.params = stats['hyperparams']

        # connect up nodes.
        self._attach_nodes()

    def save_params(self, filename):
        """ Saves model parameters. """
        saver = tf.train.Saver()
        saver.save(self.session, filename)

    def restore_params(self, filename):
        """ Restores model parameters. """
        saver = tf.train.Saver()
        saver.restore(self.session, filename)

    def _attach_nodes(self):
        """ Gets references to key nodes in graph. """

        graph = self.session.graph

        # attach operations
        self.prediction = graph.get_tensor_by_name("prediction:0")
        self.accuracy = graph.get_tensor_by_name("accuracy:0")
        self.loss=graph.get_tensor_by_name("loss:0")
        self.train_op=graph.get_operation_by_name("train_op")

        # attach to IO tensors
        self.X = graph.get_tensor_by_name("X:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.is_training = graph.get_tensor_by_name("training:0")
        self.global_step = graph.get_tensor_by_name("global_step:0")
        self.state_out = graph.get_tensor_by_name("state_out:0")
        self.state_in = graph.get_tensor_by_name("state_in:0")

    def classify_frame(self, frame, state=None):
        """
        Classify a single frame.
        :param frame: numpy array of dims [C, H, W]
        :param state: the previous state, or none for initial frame.
        :return: tuple (prediction, state).  Where prediction is score for each class
        """
        if state is None:
            state_shape = self.state_in.shape
            state = np.zeros([1, state_shape[1], state_shape[2]], dtype=np.float32)

        batch_X = frame[np.newaxis,np.newaxis,:]
        feed_dict = self.get_feed_dict(batch_X, [0])
        feed_dict[self.state_in] = state
        pred, state = self.session.run([self.prediction, self.state_out], feed_dict=feed_dict)
        pred = pred[0]
        return pred, state

