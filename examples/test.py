import tensorflow as tf
sess = tf.InteractiveSession()

with tf.device("cpu:0"):
    a = tf.placeholder(tf.float32, [2])
    b = tf.placeholder(tf.float32, [2])

    iff = tf.where(a < b, tf.constant(1.0), tf.constant(-1.0))


sess.run(iff, {a: 100.0, b: 10.0})
