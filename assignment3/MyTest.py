import tensorflow as tf

X = tf.Variable(initial_value=10.0, trainable=True)

loss = 2*X*X - 5*X + 4

op = tf.train.AdamOptimizer(0.03).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(op)

    print(sess.run([X, loss]))

