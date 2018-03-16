
import tensorflow as tf
import math
import os

class BestCheckpointSaver(tf.train.SessionRunHook):

    def __init__(self, target, checkpoint_dir, checkpoint_filename = "model.ckpt", minimize = True, **kwargs):

        self.target = target if minimize else -target
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        self.best_value = math.inf
        self.saver = tf.train.Saver(**kwargs)

    def begin(self):
        self.global_step = tf.train.get_or_create_global_step()

    def before_run(self, context):
        return tf.train.SessionRunArgs(
            dict(
                target = self.target,
                step = self.global_step,
            ), 
            None,
            None
        )

    def after_run(self, context, values):

        target = values.results["target"]
        step = values.results["step"]

        if target <= self.best_value:
            self.best_value = target
            tf.logging.info("Found new best model at step {} with value {}".format(step, target))

            self.saver.save(context.session, self.checkpoint_path, global_step = self.global_step)







    