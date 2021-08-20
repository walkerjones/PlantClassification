
import tensorflow as tf
from tensorflow import keras
import os

model = keras.models.load_model('h5/fruits_50.h5')

txtfile = open("fruits/classes.txt")
class_names = txtfile.read()
class_names = class_names.split(",")
classes=len(class_names)

significant_floor = 1/classes

folder_path="fruitstest"
for fname in os.listdir(folder_path):
    fpath = os.path.join(folder_path, fname)
    image_size = (180, 180) 
    img = keras.preprocessing.image.load_img(fpath, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions   

    print(" Inference on: ", fpath)

    for i in range(classes):
        if score[0,i] > significant_floor:
            print('%12s'%class_names[i],":", ('{:.1%}'.format(score[0,i])))