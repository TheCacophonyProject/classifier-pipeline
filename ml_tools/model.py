import tensorflow as tf

import os.path
import pickle
import math
import logging
import time
import json
from collections import namedtuple

from ml_tools import tools

# folder to save model while it's training.  Make sure this isn't on a dropbox folder and it will cause a crash.
CHECKPOINT_FOLDER = "c:\cac\checkpoints"

class Model:
    """ Defines a deep learning model """

    MODEL_NAME = "abstract model"
    MODEL_DESCRIPTION = ""

    def __init__(self):

        self.name = "model"
        self.session = tools.get_session()

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
        self.pred = None
        # accuracy of batch
        self.accuracy = None
        # total loss of batch
        self.loss = None
        # training operation
        self.train_op = None

        # number of samples to use when evaluating the model, 1000 works well but is a bit slow,
        # 100 should give results to within a few percent.
        self.eval_samples = 500

        # how often to do an evaluation + print
        self.print_every = 200

        # restore best weights found during training rather than the most recently one.
        self.use_best_weights = True

        # list of (mean, standard deviation) for each channel
        self.normalisation_constants = None

        # the score this model got on it's final evaluation
        self.eval_score = None

        # our current global step
        self.step = 0

        # enabled parallel loading and training on data (much faster)
        self.enable_async_loading = True

        # folder to write tensorboard logs to
        self.log_dir = './logs'

        # dictionary containing current hyper parameters
        self.params = {
            # augmentation
            'augmentation': True,
            'filter_threshold': 20,
            'filter_noise': 1.0,
            'scale_frequency': 0.5,
            # dropout
            'keep_prob': 0.5,
        }

    def import_dataset(self, base_path, force_normalisation_constants=None):
        """
        Import dataset from basepath.
        :param base_path:
        :param force_normalisation_constants: If defined uses these normalisation constants rather than those
            saved with the dataset.
        :return:
        """
        datasets = pickle.load(open(os.path.join(base_path, "datasets.dat"),'rb'))
        self.datasets.train, self.datasets.validation, self.datasets.test = datasets

        # augmentation really helps with reducing over-fitting, but test set should be fixed so we don't apply it there.
        self.datasets.train.enable_augmentation = self.params['augmentation']
        self.datasets.train.scale_frequency = self.params['scale_frequency']
        self.datasets.validation.enable_augmentation = False
        self.datasets.test.enable_augmentation = False
        for dataset in datasets:
            dataset.filter_threshold = self.params['filter_threshold']
            dataset.filtered_noise = self.params['filter_noise']

        logging.info("Training segments: {0:.1f}k".format(self.datasets.train.rows/1000))
        logging.info("Validation segments: {0:.1f}k".format(self.datasets.validation.rows/1000))
        logging.info("Test segments: {0:.1f}k".format(self.datasets.test.rows/1000))
        logging.info("Labels: {}".format(self.datasets.train.labels))

        label_strings = [",".join(self.datasets.train.labels),
                         ",".join(self.datasets.validation.labels),
                         ",".join(self.datasets.test.labels)]
        assert len(set(label_strings)) == 1, 'dataset labels do not match.'

        if force_normalisation_constants:
            print("Using custom normalisation constants.")
            for dataset in datasets:
                dataset.normalisation_constants = force_normalisation_constants

    def set_ops(self, pred, accuracy, loss, train_op):
        """ Sets nodes to be used for various operations. """
        self.pred = pred
        self.accuracy = accuracy
        self.loss = loss
        self.train_op = train_op

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

    @property
    def labels(self):
        """ List of labels this model can classifiy. """
        return self.datasets.train.labels

    def eval_batch(self, batch_X, batch_y, writer=None):
        """
        Evaluates the accuracy on a batch of frames.  If the batch is too large it will be broken into smaller parts.
        :param batch_X:
        :param batch_y:
        :param writer:
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
            if writer is not None and i == 0:
                summary, acc, ls = self.session.run([self.merged_summary, self.accuracy, self.loss], feed_dict=feed_dict)
            else:
                acc, ls = self.session.run([self.accuracy, self.loss], feed_dict=feed_dict)

            score += samples * acc
            loss += ls

        if writer is not None:
            writer.add_summary(summary, global_step=self.step)

        # find per sample loss, but expected loss is per batch for some reason.... change this around later on to per
        # sample
        return score / total_samples, loss / total_samples * self.batch_size

    def get_feed_dict(self, X, y, is_training=False):
        """ returns a feed dictionary for TensorFlow placeholders. """
        return {self.X: X, self.y: y, self.keep_prob: self.params['keep_prob'] if is_training else 1.0,
                self.is_training: is_training, self.global_step: self.step}

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
            probs = self.session.run([self.pred], feed_dict={self.X: Xm})[0]
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

    def load_prev_best(self):
        saver = tf.train.Saver()
        saver.restore(self.session, os.path.join(CHECKPOINT_FOLDER, "training-best.sav"))

    def benchmark_model(self):
        """
        Runs a benchmark on the model, saving performance statistics to tensorboard log folder.
        """
        #todo
        pass

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

        print("Training...")

        init = tf.global_variables_initializer()
        iterations = int(math.ceil(epochs * self.rows / self.batch_size))
        if run_name is None:
            run_name = self.MODEL_NAME
        eval_time = 0
        train_time = 0
        prep_time = 0
        best_step = 0
        steps_since_print = 0
        last_epoch_save = 0
        best_val_loss = float('inf')

        # setup saver
        saver = tf.train.Saver()

        # setup tensorboard
        os.makedirs(self.log_dir, exist_ok=True)
        writer_train = tf.summary.FileWriter(os.path.join(self.log_dir,run_name+'/train'), graph=self.session.graph)
        writer_val = tf.summary.FileWriter(os.path.join(self.log_dir,run_name+'/val'), graph=self.session.graph)
        merged = tf.summary.merge_all()
        self.merged_summary = merged

        # Run the initializer
        self.session.run(init)

        for i in range(iterations):

            self.step = i

            # get a new batch
            start = time.time()
            batch = self.datasets.train.next_batch(self.batch_size)
            prep_time += time.time()-start

            # evaluate every so often
            if steps_since_print >= self.print_every or (i==iterations-1):

                start = time.time()

                val_batch = self.datasets.validation.next_batch(self.eval_samples)
                train_batch = self.datasets.train.next_batch(self.eval_samples)

                if writer_val is not None and writer_train is not None:
                    train_accuracy, train_loss = self.eval_batch(
                        train_batch[0], train_batch[1],
                        writer=writer_train)
                    val_accuracy, val_loss = self.eval_batch(
                        val_batch[0], val_batch[1],
                        writer=writer_val)

                    writer_train.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="metric_accuracy", simple_value=train_accuracy)]),
                        global_step=i)
                    writer_val.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="metric_accuracy", simple_value=val_accuracy)]),
                        global_step=i)

                    writer_train.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="metric_loss", simple_value=train_loss)]),
                        global_step=i)
                    writer_val.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="metric_loss", simple_value=val_loss)]),
                        global_step=i)

                else:
                    train_accuracy, train_loss = self.eval_batch(train_batch[0], train_batch[1])
                    val_accuracy, val_loss = self.eval_batch(val_batch[0], val_batch[1])

                epoch = (self.batch_size * i) / self.rows

                eval_time += time.time()-start

                steps_remaining = (iterations - i)
                step_time = prep_time + train_time + eval_time
                eta = (steps_remaining * step_time / steps_since_print) / 60

                print('[epoch={0:.2f}] step {1}, training={2:.1f}%/{3:.3f} validation={4:.1f}%/{5:.3f} [times:{6:.1f}ms,{7:.1f}ms,{8:.1f}ms] eta {9:.1f} min'.format(
                    epoch, i, train_accuracy*100, train_loss * 10, val_accuracy*100, val_loss * 10,
                    1000 * prep_time / steps_since_print  / self.batch_size,
                    1000 * train_time / steps_since_print  / self.batch_size,
                    1000 * eval_time / steps_since_print  / self.batch_size,
                    eta
                ))

                # create a save point
                saver.save(self.session, os.path.join(CHECKPOINT_FOLDER, "training-most-recent.sav"))

                # save at epochs
                if int(epoch) > last_epoch_save:
                    print('Save epoch reference')
                    saver.save(self.session, os.path.join(CHECKPOINT_FOLDER, "training-epoch-{:02d}.sav".format(int(epoch))))
                    last_epoch_save = int(epoch)

                if val_loss < best_val_loss:
                    print('Save best model')
                    saver.save(self.session, os.path.join(CHECKPOINT_FOLDER, "training-best.sav"))
                    best_val_loss = val_loss
                    best_step = i

                eval_time = 0
                train_time = 0
                prep_time = 0
                steps_since_print = 0

            # train on this batch
            start = time.time()
            feed_dict = self.get_feed_dict(batch[0], batch[1], is_training=True)
            _ = self.session.run([self.train_op], feed_dict=feed_dict)
            train_time += time.time()-start

            steps_since_print += 1

        # restore previous best
        if self.use_best_weights:
            print("Using model from step", best_step)
            self.load_prev_best()

        self.eval_score = self.eval_model()

        if writer_val:
            summary_op = tf.summary.text('metric/finalscore', tf.convert_to_tensor(str(self.eval_score)))
            summary = self.session.run(summary_op)
            writer_val.add_summary(summary)

        if self.enable_async_loading:
            self.stop_async()

    def start_async_load(self):
        self.datasets.train.start_async_load(self.eval_samples+32)
        self.datasets.validation.start_async_load(self.eval_samples+32)

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

    def save_model(self):
        """ Saves a copy of the current model. """
        score_part = "{:.3f}".format(self.eval_score)
        while len(score_part) < 3:
            score_part = score_part + "0"

        saver = tf.train.Saver()
        save_filename = os.path.join("./models/", self.MODEL_NAME + '-' + score_part)
        print("Saving", save_filename)
        saver.save(self.session, save_filename)

        # save some additional data
        model_stats = {}
        model_stats['name'] = self.MODEL_NAME
        model_stats['description'] = self.MODEL_DESCRIPTION
        model_stats['notes'] = ""
        model_stats['classes'] = self.labels
        model_stats['score'] = self.eval_score
        model_stats['normalisation'] = self.datasets.train.normalisation_constants

        json.dump(model_stats, open(save_filename + ".txt", 'w'), indent=4)
