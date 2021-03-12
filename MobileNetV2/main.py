import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import matplotlib.pyplot as plt

file_name = 'image2.jpg'
image = image.load_img(file_name, target_size=(224,224))
plt.imshow(image)
plt.show()

mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
resize_image = tf.keras.preprocessing.image.img_to_array(image)
final_image = np.expand_dims(resize_image, axis=0)
final_image = tf.keras.applications.mobilenet_v2.preprocess_input(final_image)

predictions = mobile.predict(final_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)

