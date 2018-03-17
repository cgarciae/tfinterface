
import tensorflow as tf
import math
import json
import os

####################
## functions
####################

def floyd_metrics_formatter(tags_values):
    """
    This functions formats `tf.train.LoggingTensorHook` according to floyd metric logging systems. Usage:
    ```
    tf.train.LoggingTensorHook(
        ...,
        formatter = ti.estimator.hooks.floyd_metrics_formatter,
    )
    ```
    """
    tags_values = [ dict(metric = metric, value = float(value)) for metric, value in tags_values.items() ]
    tags_values = [ json.dumps(tag) for tag in tags_values ]

    return "\n" + "\n".join(tags_values)


####################
## classes
####################

class BestCheckpointSaver(tf.train.SessionRunHook):

    def __init__(self, target, checkpoint_dir, skip_n_iters = -1, save_steps = 50, checkpoint_filename = "model.ckpt", minimize = True, **kwargs):

        self.target = target if minimize else -target
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        self.best_value = float("inf")
        self.saver = tf.train.Saver(**kwargs)
        self._trigger = False 
        self.save_steps = save_steps
        self.skip_n_iters = skip_n_iters

    def begin(self):
        self.global_step = tf.train.get_or_create_global_step()


    def before_run(self, context):

        run_tensors = dict(
            step = self.global_step,
        )

        if self._trigger:
            run_tensors["target"] = self.target

        return tf.train.SessionRunArgs(
            run_tensors,
            None,
            None
        )

    def after_run(self, context, values):

        step = values.results["step"]

        if self._trigger:
            target = values.results["target"]

            if target <= self.best_value:
                self.best_value = target
                tf.logging.info("Found new best model at step {} with value {}".format(step, target))

                checkpoint_path_base = os.path.basename(self.checkpoint_path)
                if not os.path.exists(checkpoint_path_base):
                    os.makedirs(checkpoint_path_base)

                self.saver.save(context.session, self.checkpoint_path, global_step = self.global_step)

        
        self._trigger = (step % (self.save_steps) == 0) and (step >= self.skip_n_iters)

        







    