.. tfinterface documentation master file, created by
   sphinx-quickstart on Fri Sep  1 18:38:52 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tfinterface's documentation!
=======================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. note:: The use of **Python 3** is *highly* preferred over Python 2. Consider upgrading your applications and infrastructure if you find yourself *still* using Python 2 in production today. If you are using Python 3, congratulations — you are indeed a person of excellent taste.
 —*Kenneth Reitz*

**Behold, the power of tfinterface**::

    >>> import tfinterface as ti
    >>> import tensorflow as tf
    >>> inputs_t = ti.supervised.GeneralSupervisedInputs(
    ...            name='inputs',
    ...            features=dict(shape=(None,2), dtype=tf.float32),
    ...            labels=dict(shape=(None,1), dtype=tf.float32)
    >>> inputs=inputs_t()
    >>> class ModelExample(ti.supervised.SoftmaxClassifier):
    ...            def get_logits(self, ) #TODO
To finnish early.
