import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

def load_history(dataset_directory, model_variant, epochs):
    filepath = os.path.join("saves", dataset_directory, 
        model_variant, "history"+str(epochs)+".npy")
    history = np.load(filepath,allow_pickle='TRUE').item()
    return history

def load_model(dataset_directory, model_variant, model_choice):
    model = keras.models.load_model(os.path.join("saves", 
        dataset_directory, model_variant, model_choice))
    return model

def plot_history(history, dataset_directory, model_variant, epochs):
    directory = os.path.join("saves", "graphics", dataset_directory)
    print("Drawing history")
    

    ## summarize history for accuracy
    ## size optimized for word document
    plt.figure(figsize=(7.0,3))
    plt.tight_layout()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    #plt.title('model accuracy')
    plt.ylabel('dokładność')
    plt.xlabel('epoka')
    plt.legend(['uczenie', 'walidacja'], loc='upper left')
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(directory,model_variant+str(epochs))+ "_acc.png",
        bbox_inches='tight',transparent = True, dpi=600)
    plt.close()

    ## summarize history for loss
    plt.figure(figsize=(7.0,3))
    plt.tight_layout()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    #plt.title('model loss')
    plt.ylabel('strata')
    plt.xlabel('epoka')
    plt.legend(['uczenie', 'walidacja'], loc='upper right')
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(directory,model_variant+str(epochs))+"_loss.png",
        bbox_inches='tight',transparent = True, dpi=600)
    plt.close()

def draw_model(model_variant,model):
    model.summary()
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
        dpi = 100,
        layer_range = None,)


dataset_directory = "flower299"
model_variant = "advanced"
model_choice = "save_100.h5"
epochs = 100

history=load_history(dataset_directory,model_variant, epochs)
plot_history(history,dataset_directory,model_variant, epochs)

model=load_model(dataset_directory, model_variant, model_choice)
draw_model(model_variant,model)
