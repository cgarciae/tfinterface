



class UFFGenerator(object):

    def __init__(self, serving_input_fn, model_fn, model_dir, params, sess = None):
        self.sess = sess
        self.serving_input_fn = serving_input_fn
        self.model_fn = model_fn
        self.model_dir = model_dir
        self.params = params

        self.graph = tf.Graph()

        with self.graph.as_default():

            inputs = self.serving_input_fn()

            if isinstance(inputs, tf.estimator.export.ServingInputReceiver):
                
                self.features = inputs.features
                self.receiver_tensors = inputs.receiver_tensors
            else:
                self.features = inputs
                self.receiver_tensors = inputs

            spec = self.model_fn(self.features, None, tf.estimator.ModeKeys.PREDICT, self.params)

            self.predictions = spec.predictions

            print("PREDICTIONS UFF")
            print(self.predictions)



    def dump(self, uff_path, model_outputs):

        import uff

        with self.graph.as_default(), tf.Session() as sess:
            saver = tf.train.Saver()
            path = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, path)

            model_graph = sess.graph.as_graph_def()

            model_graph = tf.graph_util.convert_variables_to_constants(sess, model_graph, model_outputs)
            model_graph = tf.graph_util.remove_training_nodes(model_graph)

            for elem in model_graph.node:
                print(elem.name)

            uff_model = uff.from_tensorflow(model_graph, model_outputs)

            assert(uff_model)

            uff_folder = os.path.dirname(uff_path)

            if uff_folder and not os.path.exists(uff_folder):
                os.makedirs(uff_folder)

            # define mode based on python version
            if sys.version_info.major > 2:
                mode = "wb"
            else:
                mode = "w"
            
            with open(uff_path, mode) as f:
                f.write(uff_model)

    