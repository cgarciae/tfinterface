import tensorflow as tf
sess = tf.InteractiveSession()

with tf.device("cpu:0"):
    a = tf.placeholder(tf.float32, [])


sess.run(tf.convert_to_tensor(2.0), {a: 100.0})
