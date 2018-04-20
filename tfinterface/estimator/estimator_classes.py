from __future__ import absolute_import, print_function

import os
import tensorflow as tf
from .getters import FileGetter, FolderGetter


class TRTFrozenGraphPredictor(object):

    def __init__(self, input_nodes, output_nodes, frozen_graph_path, input_map_fn = None, engine = {}, **kwargs):
        import pycuda.driver as cuda
        import pycuda.autoinit
        import tensorrt as trt

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self._engine = None

        # set name to "" to override the default which is "import"
        kwargs.setdefault("name", "")

        graph = tf.Graph()

        with graph.as_default():

            with tf.gfile.GFile(frozen_graph_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            if input_map_fn is not None:
                input_map = input_map_fn()
                kwargs["input_map"] = input_map

            with tf.Session(graph = graph) as sess:
                tf.import_graph_def(graph_def, **kwargs)

                graph_def = graph.as_graph_def()

                graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, self.output_nodes)
                graph_def = tf.graph_util.remove_training_nodes(graph_def)
            

        self._engine = trt.lite.Engine(
            framework = "tf", 
            stream = graph_def,
            input_nodes = input_nodes, 
            output_nodes = output_nodes,
            **engine
        )

    def predict(self, *args):
        return self._engine.infer(*args)

    def __del__(self):
        if self._engine is not None:
            self._engine.destroy()


class FrozenGraphPredictor(FileGetter):

    def __init__(self, frozen_graph_path, input_nodes, output_nodes, input_map_fn = None, sess = None, **kwargs):

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.frozen_graph_path = frozen_graph_path
        self.sess = None

        # set name to "" to override the default which is "import"
        kwargs.setdefault("name", "")

        self.graph = tf.Graph() if sess is None else sess.graph

        with self.graph.as_default():

            with tf.gfile.GFile(self.frozen_graph_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            if input_map_fn is not None:
                input_map = input_map_fn()
                kwargs["input_map"] = input_map

            self.sess = tf.Session(graph = self.graph) if sess is None else sess
            tf.import_graph_def(graph_def, **kwargs)

    def predict(self, **kwargs):

        return self.sess.run(self.output_nodes, feed_dict = {
            self.input_nodes[name]: kwargs[name] 
            for name in self.input_nodes
        })

    def show_graph(self):
        if not hasattr(self, "_writter"):
            log_dir = os.path.dirname(self.frozen_graph_path)
            self._writter = tf.summary.FileWriter(log_dir, self.sess.graph)

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

class CheckpointPredictor(object):

    def __init__(self, input_nodes, output_names, model_dir = None, frozen_graph_path = None, input_map_fn = None, **kwargs):

        self.input_nodes = input_nodes
        self.output_names = output_names
        self.sess = None

        if not (bool(model_dir) ^ bool(frozen_graph_path)):
            raise ValueError("Must pass model path or checkpoint path")
        elif model_dir:
            frozen_graph_path = tf.train.latest_checkpoint(model_dir)

            if frozen_graph_path is None:
                raise ValueError("Checkpoint not found at: {}".format(model_dir))

        meta_path = frozen_graph_path + ".meta"

        self.graph = tf.Graph()

        with self.graph.as_default():

            if input_map_fn is not None:
                input_map = input_map_fn()
                kwargs["input_map"] = input_map

            self.sess = tf.Session(graph = self.graph)
            saver = tf.train.import_meta_graph(meta_path, **kwargs)
            saver.restore(self.sess, frozen_graph_path)


    def predict(self, **kwargs):

        return self.sess.run(self.output_names, feed_dict = {
            self.input_nodes[name]: kwargs[name] 
            for name in self.input_nodes
        })

    def __del__(self):
        if self.sess is not None:
            self.sess.close()




class EstimatorPredictor(object):

    def __init__(self, estimator, serving_input_fn):
        self._predictor = tf.contrib.predictor.from_estimator(estimator, serving_input_fn)

    def predict(self, **kwargs):
        return self._predictor(kwargs) 



class TRTCheckpointPredictor(object):
    """ Inference class to abstract evaluation """
    def __init__(self, serving_input_fn, model_fn, model_dir, params, input_nodes, output_nodes, **kwargs):
        import pycuda.driver as cuda
        import pycuda.autoinit
        import tensorrt as trt
        
        self.serving_input_fn = serving_input_fn
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.params = params
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.graph = tf.Graph()

        
        with self.graph.as_default(), tf.Session(graph = self.graph) as sess:

            inputs = self.serving_input_fn()

            if isinstance(inputs, tf.estimator.export.ServingInputReceiver):
                
                self.features = inputs.features
                self.receiver_tensors = inputs.receiver_tensors
            else:
                self.features = inputs
                self.receiver_tensors = inputs

            spec = self.model_fn(self.features, None, tf.estimator.ModeKeys.PREDICT, deepcopy(self.params or {}))

            self.predictions = spec.predictions
            
            # load model
            saver = tf.train.Saver()
            path = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, path)

            # freeze graph
            model_graph = sess.graph.as_graph_def()

            model_graph = tf.graph_util.convert_variables_to_constants(sess, model_graph, self.output_nodes)
            model_graph = tf.graph_util.remove_training_nodes(model_graph)
            
            # create engine
            self.engine = trt.lite.Engine(
                framework = "tf", 
                stream = model_graph,
                input_nodes = input_nodes, 
                output_nodes = output_nodes,
                **kwargs
            )

    def predict(self, *args):
        return self.engine.infer(*args)

    def __del__(self):
        self._engine.destroy()


class UFFPredictor(object):
    def __init__(self, uff_path, input_nodes, output_nodes, **kwargs):

        import pycuda.driver as cuda
        import pycuda.autoinit
        import tensorrt as trt
        
        self.engine = trt.lite.Engine(
            framework = "uff", 
            path = uff_path, 
            input_nodes = input_nodes, 
            output_nodes = output_nodes,
            **kwargs
        )
    
    def predict(self, *args):
        return self.engine.infer(*args)

class UFFPredictorV2(object):
    def __init__(self, uff_path, input_nodes, output_nodes, severity):

        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import uff

        SEVERITY = {'ERROR' : trt.infer.LogSeverity.ERROR,
                    'INFO'  : trt.infer.LogSeverity.INFO,
                    'INTERNAL_ERROR' : trt.infer.LogSeverity.INTERNAL_ERROR,
                    'WARNING' : trt.infer.LogSeverity.WARNING}

        self.glogger = trt.infer.ConsoleLogger(SEVERITY.get(severity.upper(), SEVERITY['INFO']))

        self.uff_model = open(uff_path, 'rb').read()

        parser = trt.parsers.uffparser.create_uff_parser()
        for i, (node, dims) in enumerate(input_nodes.items()):
            parser.register_input(node, tuple(dims), i)
        for node in output_nodes:
            parser.register_output(node)

        self.engine = trt.utils.uff_to_trt_engine(logger=self.glogger,
                                                  stream=self.uff_model,
                                                  parser=parser,
                                                  max_batch_size=1,
                                                  max_workspace_size= 1 << 30,
                                                  datatype=trt.infer.DataType.FLOAT
                                                  )

        self.runtime = trt.infer.create_infer_runtime(self.glogger)
        self.context = self.engine.create_execution_context()


    def infer(self, input_data, output_shape):

        import numpy as np
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import uff


        output = np.empty(output_shape, dtype = np.float32)

        d_input  =  cuda.mem_alloc(1 * input_data.size * input_data.dtype.itemsize)
        d_output =  cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()

        cuda.memcpy_htod_async(d_input, input_data, stream)
        self.context.enqueue(1, bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        return output

    def __del__(self):
        self.context.destroy()
        self.engine.destroy()
        self.runtime.destroy()

