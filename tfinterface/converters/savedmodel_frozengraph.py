from __future__ import absolute_import, print_function

import tensorflow as tf
from ..estimator.getters import FolderGetter


class SavedModel2FrozenGraph(FolderGetter):
    """Example Google style docstrings.

    This class 

    Example:
        Examples can be given using either the ``Example`` or ``Examples``
        sections. Sections support any reStructuredText formatting, including
        literal blocks::

            converter = ti.converters.SavedModel2FrozenGraph(saved_model_dir)
            
            converter.dump(frozen_graph_path)

            print("Input Nodes: {}".format(converter.input_nodes))
            print("Output Nodes: {}".format(converter.output_nodes))

    Section breaks are created by resuming unindented text. Section breaks
    are also implicitly created anytime a new section starts.

    Attributes:
        module_level_variable1 (int): Module level variables may be documented in
            either the ``Attributes`` section of the module docstring, or in an
            inline docstring immediately following the variable.

            Either form is acceptable, but the two should not be mixed. Choose
            one convention to document module level variables and be consistent
            with it.

    Todo:
        * For module TODOs
        * You have to also use ``sphinx.ext.todo`` extension
    """


    def __init__(self, saved_model_dir):

        predictor = tf.contrib.predictor.from_saved_model(saved_model_dir)

        graph = predictor.graph
        sess = predictor.session
        output_nodes = [ t.name for t in predictor.fetch_tensors.values() ]
        input_nodes = [ t.name for t in predictor.feed_tensors.values() ]
        

        graph_def = graph.as_graph_def()
        graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, output_nodes)
        graph_def = tf.graph_util.remove_training_nodes(graph_def)


        self.graph_def = graph_def
        self.predictor = predictor
        self.sess = sess
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    
    def dump(self, frozen_graph_path):

        with tf.gfile.GFile(frozen_graph_path, "wb") as f:
            f.write(self.graph_def.SerializeToString())
