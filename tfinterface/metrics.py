from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf

def _create_local_variable(name, shape, collections = None, validate_shape = True, dtype = dtypes.float32):
    """
    Creates a new local variable.
    """
    # Make sure local variables are added to
    # tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])
    collections += [ops.GraphKeys.LOCAL_VARIABLES]

    return variables.Variable(
        initial_value = array_ops.zeros(shape, dtype=dtype),
        name = name,
        trainable = False,
        collections = collections,
        validate_shape = validate_shape
    )

def streaming_confusion_matrix(labels, predictions, weights=None, num_classes=None):
    """
    Compute a streaming confusion matrix
    :param labels: True labels
    :param predictions: Predicted labels
    :param weights: (Optional) weights (unused)
    :param num_classes: Number of labels for the confusion matrix
    :return: (percent_confusionMatrix,update_op)
    """
    # Compute a per-batch confusion
    print(labels, predictions)
    batch_confusion = tf.confusion_matrix(labels, predictions, num_classes=num_classes, name='batch_confusion')

    count = _create_local_variable(None,(),dtype=tf.int32)
    confusion = _create_local_variable('streamConfusion', [num_classes, num_classes], dtype=tf.int32)

    # Create the update op for doing a "+=" accumulation on the batch
    count_update = count.assign(count + tf.reduce_sum(batch_confusion))
    confusion_update = confusion.assign(confusion + batch_confusion)

    update_op = tf.group(confusion_update, count_update)

    percent_confusion = 100 * tf.truediv(confusion, count)

    return percent_confusion, update_op

def r2_score(predictions, labels):

    total_error = tf.reduce_sum(tf.square(tf.subtract(labels, tf.reduce_mean(labels))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(labels, predictions)))
    R_squared = 1.0 - unexplained_error / total_error

    return R_squared
