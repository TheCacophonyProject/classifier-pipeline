import os.path
import numpy as np
import random
import pickle
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import itertools
import gzip
import json
import dateutil
import binascii
import time
from ml_tools import tools

# folder to save model while it's training.  Make sure this isn't on a dropbox folder and it will cause a crash.
CHECKPOINT_FOLDER = "c:\cac\checkpoints"

class Model:
    """ Defines a ML model """

    def __init__(self, datasets, X, y, keep_prob, pred, accuracy, loss, train_op, classes):

        self.datasets = datasets


        self.X = X
        self.y = y
        self.keep_prob = keep_prob

        # tensorflow node returning softmax prediction for each class
        self.pred = pred

        # tensorflow node returning accuracy of batch
        self.accuracy = accuracy

        # tensorflow node returning total loss of batch
        self.loss = loss

        # training operation
        self.train_op = train_op

        # list of classes this model can classify
        self.classes = classes

        # restore best weights found during training rather than the most recently one.
        self.use_best_weights = True

        self.sess = tools.get_session()
        self.batch_size = 16

        self.eval_score = 0.0

        self.normalisation_constants = None


    def eval_batch(self, batch_X, batch_y, include_loss = False):
        """ Evaluates the accuracy on a batch of frames.  If the batch is too large it will be broken into smaller parts. """

        total_samples = batch_X.shape[0]
        batches = (total_samples // self.batch_size) + 1
        score = 0
        loss = 0
        for i in range(batches):
            Xm = batch_X[i*self.batch_size:(i+1)*self.batch_size]
            ym = batch_y[i*self.batch_size:(i+1)*self.batch_size]
            if len(Xm) == 0:
                continue
            samples = Xm.shape[0]

            acc, ls = self.sess.run([self.accuracy, self.loss], feed_dict={self.X: Xm, self.y: ym})

            score += samples * acc
            loss += ls

        if include_loss:
            return score / total_samples, loss
        else:
            return score / total_samples

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
            probs = self.sess.run([self.pred], feed_dict={self.X: Xm})[0]
            for j in range(len(Xm)):
                predictions.append(probs[j,:])

        return predictions

    def eval_model(self):
        """ Evaluates the model on the test set. """
        print("-"*60)
        train, validation, test = self.datasets
        test.load_all()
        test_accuracy = self.eval_batch(test.X, test.y)
        print("Test Accuracy {0:.2f}% (error {1:.2f}%)".format(test_accuracy*100,(1.0-test_accuracy)*100))
        return test_accuracy

    def load_prev_best(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(CHECKPOINT_FOLDER,"training-best.sav"))

    def train_model(self, epochs=10, keep_prob=0.5, stop_after_no_improvement=None, stop_after_decline=None,
                    log_dir=None):
        """
        Trains model given number of epocs.  Uses session 'sess'
        :param epochs: maximum number of epochs to train for
        :param keep_prob: dropout keep probability
        :param stop_after_no_improvement: if best validation score was this many print statements back stop
        :param stop_after_decline: if validation ema declines for this many print statements in a row stop
        :return:
        """

        print("Training...")

        init = tf.global_variables_initializer()

        train, validation, test = self.datasets

        iterations = int(math.ceil(epochs * train.rows / self.batch_size))

        saver = tf.train.Saver()

        writer_train = None
        writer_val = None
        if log_dir:
            writer_train = tf.summary.FileWriter(os.path.join(log_dir,'train'), graph=tf.get_default_graph())
            writer_val = tf.summary.FileWriter(os.path.join(log_dir,'val'), graph=tf.get_default_graph())

        # Run the initializer
        self.sess.run(init)

        eval_time = 0
        train_time = 0
        prep_time = 0

        print_every = 200

        # number of samples to use when evaluating the model, 1000 works well but is a bit slow,
        # 100 should give results to within a few percent.
        eval_samples = 500

        best_val_accuracy = 0

        # counts cycles since last best weights
        cycles_since_last_best = 0

        # counts number of cycles in a row the the exponentialy smoothed validation accuracy has declined.
        depression_cycles = 0

        ema_val_accuracy = 0
        prev_ema_val_accuracy = 0
        best_step = 0

        merged = tf.summary.merge_all()

        steps_since_print = 0

        for i in range(iterations):

            # get a new batch
            start = time.time()
            batch = train.next_batch(self.batch_size)
            prep_time += time.time()-start

            # evaluate every so often
            if steps_since_print >= print_every or (i==iterations-1) or (i==50):

                start = time.time()

                val_batch = validation.next_batch(eval_samples)
                train_batch = train.next_batch(eval_samples)

                train_accuracy, train_loss = self.eval_batch(train_batch[0], train_batch[1], include_loss=True)
                val_accuracy, val_loss = self.eval_batch(val_batch[0], val_batch[1], include_loss=True)

                if writer_val:
                    run_metadata = tf.RunMetadata()
                    summary, _, _  = self.sess.run(
                        [merged, val_accuracy, val_loss],
                        feed_dict={self.X: val_batch[0], self.y: val_batch[1]},
                        run_metadata=run_metadata
                    )
                    writer_val.add_run_metadata(run_metadata, 'step%d' % i)
                    writer_val.add_summary(summary, i)

                ema_val_accuracy = val_accuracy if ema_val_accuracy == 0 else 0.9 * ema_val_accuracy + 0.1 * val_accuracy

                epoch = (self.batch_size * i) / train.rows

                eval_time += time.time()-start

                steps_remaining = (iterations - i)
                step_time = prep_time + train_time + eval_time
                eta = (steps_remaining * step_time / steps_since_print) / 60

                print('[epoch={0:.2f}] step {1}, training={2:.1f}%/{3:.1f} validation={4:.1f}%/{5:.1f} [times:{6:.1f}ms,{7:.1f}ms,{8:.1f}ms] (ema:{9:.3f}) eta {10:.1f} min'.format(
                    epoch, i, train_accuracy*100, train_loss, val_accuracy*100, val_loss,
                    1000 * prep_time / steps_since_print  / self.batch_size,
                    1000 * train_time / steps_since_print  / self.batch_size,
                    1000 * eval_time / steps_since_print  / self.batch_size,
                    ema_val_accuracy, eta
                ))

                # create a save point
                saver.save(self.sess, os.path.join(CHECKPOINT_FOLDER,"training-most-recent.sav"))
                if val_accuracy > best_val_accuracy:
                    saver.save(self.sess, os.path.join(CHECKPOINT_FOLDER,"training-best.sav"))
                    best_val_accuracy = val_accuracy
                    best_step = i
                    cycles_since_last_best = 0
                else:
                    cycles_since_last_best += 1

                if ema_val_accuracy < prev_ema_val_accuracy:
                    depression_cycles += 1
                else:
                    depression_cycles = 0

                prev_ema_val_accuracy = ema_val_accuracy

                if stop_after_no_improvement is not None and cycles_since_last_best >= stop_after_no_improvement:
                    print("Best validation score was too long ago, stopping.")
                    break
                if stop_after_decline is not None and depression_cycles >= stop_after_decline:
                    print("Validation scores are in decline.  Stopping.")
                    break

                eval_time = 0
                train_time = 0
                prep_time = 0
                steps_since_print = 0

            # train on this batch
            start = time.time()

            if writer_train:
                summary, _ = self.sess.run(
                    [merged, self.train_op],
                    feed_dict={self.X: batch[0], self.y: batch[1], self.keep_prob: keep_prob},
                )
                writer_train.add_summary(summary, i)
            else:
                self.train_op.run(feed_dict={self.X: batch[0], self.y: batch[1], self.keep_prob: keep_prob},
                                  session=self.sess)

            train_time += time.time()-start
            steps_since_print += 1

        # restore previous best
        if self.use_best_weights:
            print("Using model from step", best_step)
            self.load_prev_best()

        self.eval_score = self.eval_model()
