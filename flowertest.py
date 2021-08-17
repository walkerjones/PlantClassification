
import tensorflow as tf
from tensorflow import keras

image_size = (180, 180) 
model = keras.models.load_model('flowers_98.h5')

img = keras.preprocessing.image.load_img(
    "flowertest/dmucha.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis



predictions = model.predict(img_array)
score = predictions
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
for i in range(5):
    print('%12s' % class_names[i],":", ('{:.1%}'.format(score[0,i])))