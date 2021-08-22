import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from pathlib import Path
import numpy as np
import plotting
import os

dataset_directory = "flowers" #flowers/fruits/flower299
model_variant = "basic" #basic/tuned/advanced
validation_split = 0.2
epochs = 5
num_classes=5
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

def make_model_basic(num_classes,input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

def make_model_advanced(num_classes,input_shape):    
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
    filepath=os.path.join("saves", dataset_directory, model_variant)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(filepath,"history"+str(epochs)+".npy"),history.history)

def model_choice(choice):
    if choice == "basic":
        model = make_model_basic(num_classes,input_shape=image_size + (3,))
    elif choice == "tuned":
        print("Tuned model")
    elif choice == "advanced":
        model = make_model_advanced(num_classes,input_shape=image_size + (3,))
    return model

train_ds=load_train_ds()
val_ds=load_val_ds()
model = model_choice(model_variant)
compile_model(model)
fit_model(model,train_ds,val_ds,epochs)


history=plotting.load_history(dataset_directory,model_variant,epochs)
plotting.plot_history(history,dataset_directory,model_variant,epochs)

plotting.draw_model(model_variant,model)

