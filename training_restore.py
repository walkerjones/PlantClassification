import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from pathlib import Path
import numpy as np
#import plotting
import os


dataset_directory = "flower299" #flowers/fruits/flower299
model_variant = "advanced" #basic/tuned/advanced
epochs = 20
validation_split = 0.2
num_classes=299
image_size = (180, 180)
batch_size = 32

def load_train_ds():
    ##Load training part from dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join("datasets", dataset_directory),
        validation_split=validation_split,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    ##Prefetch
    train_ds = train_ds.prefetch(buffer_size=32)
    return train_ds
    
def load_val_ds():
    ##Load validation part from dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join("datasets", dataset_directory),
        validation_split=validation_split,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    ) 
    ##Prefetch
    val_ds = val_ds.prefetch(buffer_size=32)
    return val_ds

train_ds=load_train_ds()
val_ds=load_val_ds()

filepath = os.path.join("saves", dataset_directory, "best.h5")

model = load_model(filepath)
callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join("saves",dataset_directory, "continue_{epoch}.h5")),
]
history = model.fit(
    train_ds, 
    epochs=epochs, 
    callbacks=callbacks, 
    validation_data=val_ds,
)
filepath=os.path.join("saves", dataset_directory, model_variant)
Path(filepath).mkdir(parents=True, exist_ok=True)
np.save(os.path.join(filepath,"continue_history.npy"),history.history)