import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
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
    VERSION = "0.3.0"

    def __init__(self, session=None):

        self.name = "model"
        self.session = session or tools.get_session()
        self.saver = None

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
        # tensorflow nodes used to evaluate
        # ------------------------------------------------------

        # prediction for each class(probability distribution)
        self.prediction = None
        # accuracy of batch
        self.accuracy = None
        # total loss of batch
        self.loss = None
        # training operation
        self.train_op = None

        self.novelty = None
        self.novelty_distance = None

        self.state_in = None
        self.state_out = None
        self.logits_out = None
        self.hidden_out = None
        self.lstm_out = None

        # we store 1000 samples and use these to plot projections during training
        self.train_samples = None
        self.val_samples = None

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
        self.log_id = ''

        # number of frames per segment during training
        self.training_segment_frames = 27
        # number of frames per segment during testing
        self.testing_segment_frames = 27

        # dictionary containing current hyper parameters
        self.params = {
            # augmentation
            'augmentation': True,
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

    def get_feed_dict(self, X, y=None, is_training=False, state_in=None):
        """
        Returns a feed dictionary for TensorFlow placeholders.
        :param X: The examples to classify
        :param y: (optional) the labels for each example
        :param is_training: (optional) boolean indicating if we are training or not.
        :param state_in: (optional) states from previous classification.  Used to maintain internal state across runs
        :return:
        """
        result = {
            self.X: X[:, 0:self.training_segment_frames],        # limit number of frames per segment passed to trainer
            self.keep_prob: self.params['keep_prob'] if is_training else 1.0,
            self.is_training: is_training,
            self.global_step: self.step
        }
        if y is not None:
            result[self.y] = y
        if state_in is not None:
            result[self.state_in] = state_in
        return result

    def classify_batch(self, batch_X):
        """
        Classifies all segments in the given batch.
        :param batch_X: input batch of shape [n,frames,h,w,channels]
        :return: list of probs for each examples
        """
        result = self.classify_batch_extended(batch_X, [self.prediction])
        return result[self.prediction]

    def classify_batch_extended(self, batch_X, nodes):
        """
        Classifies all segments in the given batch and returns extended info
        :param batch_X: input batch of shape [n,frames,h,w,channels]
        :parram nodes: a list of node name to evaluate
        :return: dictionary mapping output node to data
        """

        assert None not in nodes, "Requests output of 'None' node."

        total_samples = batch_X.shape[0]
        batches = (total_samples // self.batch_size) + 1

        output_lists = {}
        for node in nodes:
            output_lists[node] = []

        for i in range(batches):
            Xm = batch_X[i*self.batch_size:(i+1)*self.batch_size]
            if len(Xm) == 0:
                continue

            outputs = self.session.run(nodes, feed_dict={self.X: Xm})
            for node, output in zip(nodes, outputs):
                for i in range(len(output)):
                    output_lists[node].append(output[i])

        return output_lists

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
        feed_dict = self.get_feed_dict(X, y, is_training=True)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # first run takes a while, so run this just to build the graph, then run again to get real performance.
        _, = self.session.run([self.train_op], feed_dict=feed_dict)

        _, = self.session.run([self.train_op], feed_dict=feed_dict,
                                            options=run_options,
                                            run_metadata=run_metadata)

        # write out hyper-params
        self.log_text('hyperparams', self.hyperparams_string)
        for writer in [self.writer_train, self.writer_val]:
            writer.add_run_metadata(run_metadata, 'benchmark')


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

    def log_histogram(self, tag, values, bins=1000, writer=None):
        """Logs the histogram of a list/vector of values."""

        if writer is None:
            writer = self.writer_val

        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, self.step)
        writer.flush()

    def log_text(self, tag, value, writer=None):
        """
        Writes a scalar to summary writer.
        :param tag: tag to use
        :param value: value to write
        :param writer: (optional) summary writer.  Defaults to writer_val
        """
        if writer is None:
            writer = self.writer_val

        text_tensor = tf.make_tensor_proto(str(value), dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
        writer.add_summary(summary)

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

    def tune_novelty_detection(self):

        # first take 1000 samples from the validation set.  These samples have not been seen by the model so their
        # distances should be reprentative
        seen_examples, _ = self.datasets.validation.next_batch(1000)

        # next classify the examples and check their distances
        data = self.classify_batch_extended(seen_examples, [self.novelty_distance])
        seen_distances = data[self.novelty_distance]

        # we can take a guess at a good threshold by looking at the examples from seen classes.  A threshold
        # that includes 75% of these examples seems to work well.
        threshold_distance = np.percentile(seen_distances , q=75)
        threshold_scale = np.std(seen_distances)

        # write these values to the model
        self.update_writeable_variable('novelty_threshold', threshold_distance)
        self.update_writeable_variable('novelty_scale', threshold_scale)

        self.log_scalar('novelty/threshold', threshold_distance)
        self.log_scalar('novelty/scale', threshold_scale)
        self.log_histogram('novelty/distances', seen_distances, bins=40)

        return threshold_distance, threshold_scale


    def update_training_data_examples(self):
        """ Classifies given data and stores it in model.  This can be used to visualise the logit layout, or to
            help identify examples that different significantly from that seen during training.
            A batch of 1000 examples samples from all classes is recommended.
         """

        sample_X = self.train_samples[0]

        # evaluate the stored samples and fetch the logits and hidden states
        data = self.classify_batch_extended(sample_X, [self.logits_out, self.hidden_out])

        # run the 'assign' operations that update the models stored varaibles
        for var_name, node in zip(['sample_logits', 'sample_hidden'], [self.logits_out, self.hidden_out]):
            self.update_writeable_variable(var_name, data[node])

    def generate_report(self):
        """
        Logs some important information to the tensorflow summary writer, such as confusion matrix, and f1 scores.
        :return:
        """

        # get some examples for evaluation
        examples, true_classess = self.datasets.validation.next_batch(self.report_samples)
        predictions = self.classify_batch(examples)
        predicted_classes = [np.argmax(prediction) for prediction in predictions]

        pred_label = [self.labels[x] for x in predicted_classes]
        true_label = [self.labels[x] for x in true_classess]

        cm = tools.get_confusion_matrix(pred_class=predicted_classes, true_class=true_classess, classes=self.labels)
        f1_scores = metrics.f1_score(y_true=true_label, y_pred=pred_label, labels=self.labels, average=None)

        fig = visualise.plot_confusion_matrix(cm, self.labels)
        self.log_image("confusion_matrix", visualise.fig_to_numpy(fig))
        plt.close()

        for label_number, label in enumerate(self.labels):
            self.log_scalar("f1/" + label, f1_scores[label_number])
        self.log_scalar("f1/score", np.mean(f1_scores))

        errors = correct = 0
        for pred, true in zip(pred_label, true_label):
            if pred != true:
                errors += 1
            else:
                correct += 1

        # generate a graph to show confidence levels
        fig = visualise.plot_confidence_by_class(predictions, true_classess, self.labels)
        self.log_image("confidence_scores", visualise.fig_to_numpy(fig))
        plt.close()

        accuracy = correct / (correct + errors)

        return accuracy, f1_scores

    def _create_sprite_image(self, images):
        """Returns a sprite image consisting of images passed as argument. Images should be [n,h,w]"""

        print(np.min(images), np.max(images), np.mean(images))

        # clip out the negative values
        images[images < 0] = 0

        # looks much better as a negative.
        images = -images

        if isinstance(images, list):
            images = np.array(images)
        img_h = images.shape[1]
        img_w = images.shape[2]
        n_plots = int(np.ceil(np.sqrt(images.shape[0])))

        spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < images.shape[0]:
                    this_img = images[this_filter]
                    spriteimage[i * img_h:(i + 1) * img_h,
                    j * img_w:(j + 1) * img_w] = this_img

        return spriteimage

    def setup_sample_training_data(self, log_dir, writer):

        # get some samples
        segs = [self.datasets.train.sample_segment() for _ in range(1000)]
        sample_X = []
        sample_y = []
        for segment in segs:
            data = self.datasets.train.fetch_segment(segment, augment=False)
            sample_X.append(data)
            sample_y.append(self.labels.index(segment.label))

        X = np.asarray(sample_X, dtype=np.float32)
        y = np.asarray(sample_y, dtype=np.int32)

        data = (X, y, segs)

        sprite_path = os.path.join(log_dir, "examples.png")
        meta_path = os.path.join(log_dir, "examples.tsv")

        config = projector.ProjectorConfig()
        for var_name in ['sample_logits', 'sample_hidden']:
            embedding = config.embeddings.add()
            embedding.tensor_name = var_name
            embedding.metadata_path = meta_path
            embedding.sprite.image_path = sprite_path
            embedding.sprite.single_image_dim.extend([48, 48])

        projector.visualize_embeddings(writer, config)

        # save tsv file containing labels"
        with open(meta_path, 'w') as f:
            f.write("Index\tLabel\tSource\n")
            for index, segment in enumerate(segs):
                f.write("{}\t{}\t{}\n".format(index, segment.label, segment.clip_id))

        # save out image previews
        to_vis = X[:, self.training_segment_frames // 2, 0]
        sprite_image = self._create_sprite_image(to_vis)
        plt.imsave(sprite_path, sprite_image, cmap='gray')

        return data

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

        self.log_id = run_name

        LOG_DIR = os.path.join(self.log_dir, run_name)

        iterations = int(math.ceil(epochs * self.rows / self.batch_size))
        if run_name is None:
            run_name = self.MODEL_NAME
        eval_time = 0
        train_time = 0
        prep_time = 0
        best_step = 0
        examples_since_print = 0
        last_epoch_save = -1
        best_report_acc = 0
        best_val_loss = float('inf')

        # setup writers and run a quick benchmark
        print("Initialising summary writers at {}.".format(LOG_DIR))
        self.setup_summary_writers(run_name)

        # Run the initializer
        init = tf.global_variables_initializer()
        self.session.run(init)

        self.train_samples = self.setup_sample_training_data(LOG_DIR, self.writer_train)

        # setup a saver
        self.saver = tf.train.Saver(max_to_keep=1000)

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
                train_batch = self.datasets.train.next_batch(self.eval_samples, force_no_augmentation=True)

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

                # save the best model if validation score was good
                if val_loss < best_val_loss:
                    print("Saving best validation model.")
                    self.save(os.path.join(CHECKPOINT_FOLDER, "training-best-val.sav"))
                    best_val_loss = val_loss

                # save at epochs
                if int(epoch) > last_epoch_save:

                    # create a training reference set
                    print("Updating example training data")
                    self.update_training_data_examples()
                    self.tune_novelty_detection()

                    print("Epoch report")
                    acc, f1 = self.generate_report()
                    print("results: {:.1f} {}".format(acc*100,["{:.1f}".format(x*100) for x in f1]))
                    print('Save epoch reference')
                    self.eval_score = acc
                    self.save(os.path.join(CHECKPOINT_FOLDER, "training-epoch-{:02d}.sav".format(int(epoch))))
                    last_epoch_save = int(epoch)

                    if acc > best_report_acc:
                        print('Save best epoch tested model.')
                        # saving a copy in the log dir allows tensorboard to access some additional information such
                        # as the current training data varaibles.
                        self.save(os.path.join(CHECKPOINT_FOLDER, "training-best.sav"))
                        try:
                            self.save(os.path.join(LOG_DIR, "training-epoch-{:02d}.sav".format(int(epoch))))
                        except Exception as e:
                            logging.warning("Could not write training checkpoint, probably TensorBoard is open.")
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

        self.log_text('metric/final_score', self.eval_score)

        if self.enable_async_loading:
            self.stop_async()

    def start_async_load(self):
        # make sure the workers load the correct number of frames.
        self.datasets.train.segment_width = self.testing_segment_frames
        self.datasets.validation.segment_width = self.testing_segment_frames
        self.datasets.train.start_async_load(32)
        self.datasets.validation.start_async_load(32)

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

        try:
            self.saver.save(self.session, filename)
        except Exception as e:
            print("*"*60)
            print("Warning, fail saved.  This is usally because the file was open (maybe dropbox was running?)")
            print("*" * 60)
            print(e)

        # save some additional data
        model_stats = {}
        model_stats['name'] = self.MODEL_NAME
        model_stats['description'] = self.MODEL_DESCRIPTION
        model_stats['notes'] = ""
        model_stats['labels'] = self.labels
        model_stats['score'] = self.eval_score
        model_stats['hyperparams'] = self.params
        model_stats['log_id'] = self.log_id
        model_stats['training_date'] = str(time.time())
        model_stats['version'] = self.VERSION

        json.dump(model_stats, open(filename+ ".txt", 'w'), indent=4)

    def load(self, filename):
        """ Loads model and parameters from file. """

        print("Loading model {}".format(filename))

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
        self.attach_nodes()

    def save_params(self, filename):
        """ Saves model parameters. """
        self.saver.save(self.session, filename)

    def restore_params(self, filename):
        """ Restores model parameters. """
        self.saver.restore(self.session, filename)

    def get_tensor(self, name, none_if_not_found=False):
        """
        Returns a reference to tensor by given name.
        :param name: name of tensor
        :param none_if_not_found: if true none is returned if tensor is not found otherwise an exception is thrown.
        :return: the tensor
        """
        try:
            return self.session.graph.get_tensor_by_name(name+":0")
        except Exception as e:
            if none_if_not_found:
                return None
            else:
                raise e

    def attach_nodes(self):
        """ Gets references to key nodes in graph. """

        graph = self.session.graph

        # attach operations
        self.prediction = self.get_tensor("prediction")
        self.accuracy = self.get_tensor("accuracy")
        self.loss = self.get_tensor("loss")
        self.train_op = graph.get_operation_by_name("train_op")

        # novelty
        self.novelty = self.get_tensor('novelty', none_if_not_found=True)
        self.novelty_distance = self.get_tensor('novelty_distance', none_if_not_found=True)

        # attach to IO tensors
        self.X = self.get_tensor('X')
        self.y = self.get_tensor('y')
        self.keep_prob = self.get_tensor('keep_prob')
        self.is_training = self.get_tensor('training')
        self.global_step = self.get_tensor('global_step')

        self.state_out = self.get_tensor('state_out')
        self.state_in = self.get_tensor('state_in')

        self.logits_out = self.get_tensor('logits_out', none_if_not_found=True)
        self.hidden_out = self.get_tensor('hidden_out', none_if_not_found=True)
        self.lstm_out = self.get_tensor('lstm_out')

    def freeze(self):
        """ Freezes graph so that no additional changes can be made. """
        self.session.graph.finalize()

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

        feed_dict = self.get_feed_dict(batch_X, state_in=state)
        pred, state = self.session.run([self.prediction, self.state_out], feed_dict=feed_dict)
        pred = pred[0]

        return pred, state

    def create_summaries(self, name, var):
        """
        Creates TensorFlow summaries for given tensor
        :param name: the namespace for the summaries
        :param var: the tensor
        """
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def save_input_summary(self, input, name, reference_level=None):
        """
        :param input: tensor of shape [B*F, H, W, 1]
        :param name: name of summary
        :param reference_level: if given will add pixels in corners of image to set vmin to 0 and vmax to this level.
        """

        # hard code dims
        W, H = 48, 48

        mean = tf.reduce_mean(input)
        tf.summary.histogram(name, input)
        tf.summary.scalar(name + "/max", tf.reduce_max(input))
        tf.summary.scalar(name + "/min", tf.reduce_min(input))
        tf.summary.scalar(name + "/mean", mean)
        tf.summary.scalar(name + "/std", tf.sqrt(tf.reduce_mean(tf.square(input - mean))))

        if reference_level is not None:
            # this is so silly, we need to create a mask and do some tricks just to modify two pixels...
            # seems to be because tf doesn't really allow for direct updates to tensors.
            levels = np.zeros([1, H, W, 1], dtype=np.float32)
            mask = np.ones([1, H, W, 1], dtype=np.float32)
            for i in range(W):
                levels[0, 0, i, 0] = reference_level * (i/(W-1))
                mask[0, 0, i, 0] = 0

            input = input * mask + levels
            input = tf.abs(input)

        tf.summary.image(name, input[-2:-1], max_outputs=1)

    def create_writable_variable(self, name, shape):
        """ Creates a variable in the model that can be written to. """
        var = tf.get_variable(name=name, initializer=tf.initializers.zeros, dtype=tf.float32, trainable=False,
                              shape=shape)
        input = tf.placeholder(name=name + "_in", dtype=tf.float32, shape=shape)
        assign_op = tf.assign(var, input, name=name + "_assign_op")
        return var

    def update_writeable_variable(self, name, data):
        """ Updates the contents of a writeable variable in the model. """
        assign_op = self.session.graph.get_operation_by_name(name + "_assign_op")
        var_input = self.session.graph.get_tensor_by_name(name + "_in:0")
        self.session.run(assign_op, feed_dict={var_input:data})



