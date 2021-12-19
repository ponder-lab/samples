import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(tf.float32, [None, 1], 'x')
is_training = tf.placeholder(tf.bool)

#y = tf.layers.batch_normalization(x, center=False, scale=False,
#                                  training=is_training)
bn = tf.layers.BatchNormalization(center=False, scale=False,
                                  trainable=True)
y = bn(x, training=is_training)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

sess = tf.compat.v1.InteractiveSession()
tf.global_variables_initializer().run()


for i in range(100):
    idata = np.random.normal(100, 50, size=(100, 1))
    res = sess.run([y, update_ops], feed_dict={x: idata, is_training: True})
    print(np.mean(np.squeeze(res[0])), np.std(np.squeeze(res[0])))
    res = sess.run(y, feed_dict={x: idata, is_training: False})
    print(np.mean(np.squeeze(res)), np.std(np.squeeze(res)))
#print("IN ", np.squeeze(idata))
#print("GT ", (np.squeeze(idata)-100)/50)
#print("RES", np.squeeze(res[0]))

sess.close()
