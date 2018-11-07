import tensorflow as tf

# Cria um objeto tensor no TensorFlow
hello_constant = tf.constant('Ola, Mundo!')

with tf.Session() as sess:
    # Roda a operação tf.constant na sessão
    output = sess.run(hello_constant)
    print(output)
