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

def plot_history(history, dataset_directory, model_variant, epochs, unique_name, time_factor):
    directory = os.path.join("saves", "graphics", dataset_directory, unique_name)
    print("Drawing history")
    
    #shift x label and values by one or by time-factor
    xpoints = range(1,epochs+1)
    xname = "epoka"
    if time_factor != 0:
        xpoints = np.multiply(xpoints, time_factor)
        xname = "czas serii [ms] "

    ## summarize history for accuracy
    ## size optimized for word document
    plt.figure(figsize=(7.0,3))
    plt.tight_layout()
    plt.plot(xpoints,history['accuracy'])
    plt.plot(xpoints,history['val_accuracy'])
    plt.grid()
    #plt.title('model accuracy')
    plt.ylabel('dokładność')
    plt.xlabel(xname)
    plt.legend(['uczenie', 'walidacja'], loc='lower right')
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(directory,model_variant+str(epochs))+ "_acc.png",
        bbox_inches='tight',transparent = True, dpi=600)
    plt.close()

    ## summarize history for loss
    plt.figure(figsize=(7.0,3))
    plt.tight_layout()
    plt.plot(xpoints,history['loss'])
    plt.plot(xpoints,history['val_loss'])
    plt.grid()
    #plt.title('model loss')
    plt.ylabel('strata')
    plt.xlabel(xname)
    plt.legend(['uczenie', 'walidacja'], loc='upper right')
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(directory,model_variant+str(epochs))+"_loss.png",
        bbox_inches='tight',transparent = True, dpi=600)
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
        dpi = 100,
        layer_range = None,)

    with open(os.path.join("saves", "graphics", model_variant+".txt"), "w") as text_file:
        model.summary(print_fn=lambda x: text_file.write(x+ '\n'))

""""""
#models = ["basic", "xception", "vgg19", "resnet50v2" , "mobilev2", "dense201", "efficientB5"]
models = ["basic", "xception", "resnet50v2" , "mobilev2", "dense201", "efficientB5"]
timetable =[83, 223, 191, 111, 390, 713]
for i in range(len(timetable)):



    dataset_directory = "fruits"
    model_variant = models[i]
    time_factor = timetable[i]
    model_choice = "save_50.h5"
    epochs = 50
    unique_name = model_variant + ""

    history=load_history(dataset_directory,model_variant, epochs)
    plot_history(history,dataset_directory,model_variant, epochs, unique_name, time_factor)

    model=load_model(dataset_directory, model_variant, model_choice)
    draw_model(model_variant,model)
