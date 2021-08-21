import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

def load_history(dataset_directory, model_variant):
    filepath = os.path.join("saves", dataset_directory, model_variant, "history.npy")
    history = np.load(filepath,allow_pickle='TRUE').item()
    return history

def load_model(dataset_directory,model_choice):
    model = keras.models.load_model(os.path.join("saves", dataset_directory, model_choice))
    return model

def plot_history(history, dataset_directory, model_variant):
    print("Drawing history")
    plt.tight_layout(pad=2, w_pad=2, h_pad=2)

    ## summarize history for accuracy
    plt.figure(figsize=(3.3,3.3))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    Path(os.path.join("saves", "graphics", 
        dataset_directory)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join("saves", "graphics", 
        dataset_directory, model_variant+"_acc.png"), dpi=200)
    plt.close()

    ## summarize history for loss
    plt.figure(figsize=(3.3,3.3))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    Path(os.path.join("saves", "graphics", 
        dataset_directory)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join("saves", "graphics", 
        dataset_directory, model_variant+"_loss.png"), dpi=200)
    plt.close()

def draw_model(model_variant,model):
    print("Drawing model")
    Path(os.path.join("saves", "graphics",)).mkdir(parents=True, exist_ok=True)
    keras.utils.plot_model(
        model, 
        to_file=os.path.join("saves", "graphics", model_variant+".png"), 
        show_shapes = True,
        show_dtype = False,
        show_layer_names = True ,
        rankdir = "TB",
        expand_nested= False ,
        dpi = 96,
        layer_range = None,)

"""
dataset_directory = "flowers"
model_variant = "basic"
model_choice = "save_2.h5"

history=load_history(dataset_directory,model_variant)
plot_history(history,dataset_directory,model_variant)

model=load_model(dataset_directory, model_choice)
draw_model(model_variant,model)
"""