# inference_private.py (được gọi từ Flask server)
import numpy as np
import tf_encrypted as tfe
import tensorflow as tf

def load_model():
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.keras.backend.set_session(sess)
            model = tf.keras.models.load_model('model/mnist_cnn.h5')
            return model

def predict_private(image):
    # image: numpy array shape (28, 28, 1)
    model = load_model()

    # Chuyển ảnh thành batch (1 ảnh)
    x = np.expand_dims(image, axis=0)  # (1, 28, 28, 1)

    # Wrap mô hình với tf-encrypted
    with tfe.protocol.SecureNN():
        x_private = tfe.define_private_input('input-provider', lambda: tf.convert_to_tensor(x, dtype=tf.float32))
        model_output = model(x_private)
        prediction = tfe.define_output('output-receiver', model_output)

        with tfe.Session() as sess:
            result = sess.run(prediction)
            return int(np.argmax(result))
