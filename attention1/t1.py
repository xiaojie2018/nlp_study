import tensorflow as tf

a = tf.zeros((10, 10))
b = a + 1
c = a + 2

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
