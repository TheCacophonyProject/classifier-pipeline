import time
import tensorflow as tf
import logging
import os
import psutil
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
            loss = self.compiled_loss(y, y_pred,sample_weight=sample_weight, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

def custom_train(model, epochs,train, val,loss_fn,optimizer):
    train_acc_metric = tf.keras.metrics.Accuracy(10)
    val_acc_metric = tf.keras.metrics.Accuracy(10)
    for epoch in range(epochs):
        logging.info("Start Epoch %s mem %s",epoch, psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        for batch_i in range(len(train)):
            x_batch_train,y_batch_train = train.__getitem__(batch_i)
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if batch_i % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (batch_i, float(loss_value))
                )
                print("Seen so far: %d samples" % ((batch_i + 1) * train.batch_size))
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for batch_i in range(len(val)):
            x_batch_val,y_batch_val = val.__getitem__(batch_i)
            val_logits = model(x_batch_val, training=False)

            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        # print("Time taken: %.2fs" % (time.time() - start_time))
        val.on_epoch_end()
        train.on_epoch_end()
        logging.info("Epoch %s mem %s",epoch ,psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        print("Epoch %s mem %s",epoch ,psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
