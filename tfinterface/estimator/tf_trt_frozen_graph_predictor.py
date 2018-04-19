from __future__ import absolute_import, print_function

from .getters import FileGetter

import tensorflow as tf


class TFTRTFrozenGraphPredictor(FileGetter):

    def __init__(self, frozen_graph_path, input_nodes, output_nodes, trt_ops = {}, input_map_fn = None, engine = {}, **kwargs):

        from tensorflow.contrib import tensorrt as trt

        self.frozen_graph_path = frozen_graph_path

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        n_inputs = len(self.input_nodes)

        # set name to "" to override the default which is "import"
        kwargs.setdefault("name", "")

        graph = tf.Graph()

        with graph.as_default():

            with tf.gfile.GFile(frozen_graph_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            if input_map_fn is not None:
                kwargs["input_map"] = input_map_fn()

            # gpu options
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5, allow_growth = True)
            config = tf.ConfigProto(gpu_options = gpu_options)

            self.sess = tf.Session(graph = graph, config = config)

            
            trt_graph = trt.create_inference_graph(
                input_graph_def = graph_def,
                outputs = self.output_nodes.values(),
                **trt_ops
            )  # Get optimized graph



            tensors = tf.import_graph_def(
                trt_graph,
                return_elements = self.input_nodes.values() + self.output_nodes.values(),
                **kwargs
            )

            input_tensors = tensors[:n_inputs]
            output_tensors = tensors[n_inputs:]

            self.input_nodes = {key: value for key, value in zip(self.input_nodes.keys(), input_tensors) }
            self.output_nodes = {key: value for key, value in zip(self.output_nodes.keys(), output_tensors) }


    def predict(self, **kwargs):

        return self.sess.run(self.output_nodes, feed_dict = {
            self.input_nodes[name]: kwargs[name] 
            for name in self.input_nodes
        })

    def show_graph(self):
        if not hasattr(self, "_writter"):
            log_dir = os.path.basename(self.frozen_graph_path)
            self._writter = tf.summary.FileWriter(log_dir, self.sess.graph)

    def __del__(self):
        if self.sess is not None:
            self.sess.close()