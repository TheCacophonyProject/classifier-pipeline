import time
import tensorflow as tf
import logging
import os
import psutil
import progressbar


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
model = None
loss_fn = None
optimizer = None


def custom_train(m, epochs, train, val, loss, opt, dir):
    history = {"accuracy": [], "loss": [], "val_accuracy": []}

    global model, loss_fn, optimizer
    loss_fn = loss
    model = m
    optimizer = opt
    best_accuracy = None
    same_accuracy = 0
    for epoch in range(epochs):
        logging.info(
            "Start Epoch %s mem %s",
            epoch,
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )
        bar = progressbar.ProgressBar(
            maxval=len(train),
            widgets=[
                "Epoch {}/{} ".format(epoch, epochs),
                progressbar.ETA(),
                progressbar.Bar("=", "[", "]"),
                " ",
                progressbar.Percentage(),
            ],
        )
        bar.start()
        for batch_i in range(len(train)):
            x_batch_train, y_batch_train, weights = train.__getitem__(batch_i)
            loss_value = train_step(x_batch_train, y_batch_train, weights)
            bar.update(batch_i + 1)

            if batch_i % 200 == 0:

                logging.info(
                    "Training loss (for one batch) at step %s %s",
                    batch_i,
                    float(loss_value),
                )
        bar.finish()

        train_acc = float(train_acc_metric.result())
        logging.info("Training acc over epoch: %s" % (float(train_acc),))
        history["accuracy"].append(train_acc)
        history["loss"].append(float(loss_value))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        bar = progressbar.ProgressBar(
            maxval=len(val),
            widgets=[
                "{} - Validating ".format(epoch),
                progressbar.ETA(),
                progressbar.Bar("=", "[", "]"),
                " ",
                progressbar.Percentage(),
            ],
        )
        bar.start()

        # Run a validation loop at the end of each epoch.
        for batch_i in range(len(val)):
            x_batch_val, y_batch_val, weights = val.__getitem__(batch_i)
            val_logits = test_step(x_batch_val, y_batch_val, weights)
            bar.update(batch_i + 1)
            # val_logits = model(x_batch_val, training=False)

            # Update val metrics
            # val_acc_metric.update_state(y_batch_val, val_logits)
        bar.finish()
        val_acc = float(val_acc_metric.result())
        history["val_accuracy"].append(val_acc)
        val_acc_metric.reset_states()
        logging.info("Validation acc: %s" % (float(val_acc),))
        print("Val Accuracy", val_acc)
        val.on_epoch_end()
        train.on_epoch_end()
        if best_accuracy is None:
            best_accuracy = val_acc
            model.save_weights(os.path.join(dir, "val_acc"))
            logging.info(
                "Epoch %s saving best weights accuracy %s %s", epoch, val_acc, dir
            )

        else:
            if round(best_accuracy * 100) == round(val_acc * 100):
                same_accuracy += 1
            elif val_acc > best_accuracy:
                model.save_weights(os.path.join(dir, "val_acc"))
                logging.info(
                    "Epoch %s saving best weights accuracy %s %s", epoch, val_acc, dir
                )
                same_accuracy = 0
                best_accuracy = val_acc
        logging.info(
            "Epoch %s mem %s",
            epoch,
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,
        )
        if same_accuracy == 15:
            # early stop
            logging.info("Done accuray has been the same for 15")
            break
    return history


@tf.function
def train_step(x, y, weights=None):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits, sample_weight=weights)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits, sample_weight=weights)
    return loss_value


@tf.function
def test_step(x, y, weights=None):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits, sample_weight=weights)
