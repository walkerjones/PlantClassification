import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy
import os

dataset_directory = "flowers"
image_size = (180, 180)
batch_size = 32
validation_split = 0.2
epochs = 2
num_classes=5


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

def make_model(num_classes,input_shape):
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
     
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

def fit_model(model,train_ds,val_ds,epochs):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join("saves",dataset_directory, "save_{epoch}.h5")),
    ]
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        callbacks=callbacks, 
        validation_data=val_ds,
    )
    return history


def plot_history(history):
    ## list all data in history
    ##print(history.history.keys())
    print("Saving history")

    ## summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join("saves", "graphics", dataset_directory, "advanced_acc.png"))
    plt.close()

    ## summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join("saves", "graphics", dataset_directory, "advanced_loss.png"))

def draw_model(model):
    keras.utils.plot_model(
    model, 
    to_file="model_advanced.png", 
    show_shapes = True,
    show_dtype = False,
    show_layer_names = True ,
    rankdir = "TB",
    expand_nested= False ,
    dpi = 96,
    layer_range = None,)

train_ds=load_train_ds()
val_ds=load_val_ds()
model = make_model(num_classes,input_shape=image_size + (3,))
compile_model(model)
draw_model(model)
history=fit_model(model,train_ds,val_ds,epochs)
plot_history(history)