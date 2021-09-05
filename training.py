import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from pathlib import Path
import numpy as np
import os


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

def make_model_xception(num_classes,input_shape):
    model = tf.keras.applications.Xception(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes,
    classifier_activation="softmax",)
    return model

def make_model_vgg19(num_classes,input_shape):
    model = tf.keras.applications.VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes,
    classifier_activation="softmax")
    return model

def make_model_resnet50v2(num_classes,input_shape):
    model = tf.keras.applications.ResNet50V2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes,
    classifier_activation="softmax")
    return model

def make_model_mobilev2(num_classes,input_shape):
    model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    alpha=1.0,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=num_classes,
    classifier_activation="softmax")
    return model

def make_model_dense201(num_classes,input_shape):
    model = tf.keras.applications.DenseNet201(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes)
    return model
    
def make_model_efficientB5(num_classes,input_shape):
    model = tf.keras.applications.EfficientNetB5(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes,
    classifier_activation="softmax")
    return model

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
            os.path.join("saves", dataset_directory, model_variant, "save_{epoch}.h5")),
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
    elif choice == "xception":
        model = make_model_xception(num_classes,input_shape=image_size + (3,))
    elif choice == "vgg19":
        model = make_model_vgg19(num_classes,input_shape=image_size + (3,))
    elif choice == "resnet50v2":
        model = make_model_resnet50v2(num_classes,input_shape=image_size + (3,))
    elif choice == "mobilev2":
        model = make_model_mobilev2(num_classes,input_shape=image_size + (3,))
    elif choice == "dense201":
        model = make_model_dense201(num_classes,input_shape=image_size + (3,))
    elif choice == "efficientB5":
        model = make_model_efficientB5(num_classes,input_shape=image_size + (3,))
    return model

""""""
#models = ["basic", "xception", "vgg19", "resnet50v2" , "mobilev2", "dense201", "efficientB5"]
models = ["xception"]
for type in models:



    dataset_directory = "flower299" #flowers/fruits/flower299
    num_classes=299              #5      /23    /299

    model_variant = type #"resnet50v2"
    validation_split = 0.2
    epochs = 50
    image_size = (160, 160)
    batch_size = 32

    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    print("Type: ", model_variant)
    train_ds=load_train_ds()
    val_ds=load_val_ds()
    model = model_choice(model_variant)
    compile_model(model)
    fit_model(model,train_ds,val_ds,epochs)