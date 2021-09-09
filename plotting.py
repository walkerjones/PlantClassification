import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import numpy as np
import os

def load_history(dataset_directory, model_variant, epochs):
    filepath = os.path.join("saves", dataset_directory, 
        model_variant, "history_best.npy") #"history"+str(epochs)+".npy"
    history = np.load(filepath,allow_pickle='TRUE').item()
    return history

def load_model(dataset_directory, model_variant, model_choice):
    model = keras.models.load_model(os.path.join("saves", 
        dataset_directory, model_variant, model_choice))
    return model

def plot_history(history, dataset_directory, model_variant, epochs, unique_name):
    directory = os.path.join("saves", "graphics", dataset_directory, model_variant)
    print("Drawing history")
    
    
    #shift x label and values by one or by time-factor
    xpoints = range(1,epochs+1)
    xname = "epoka"
    ## summarize history for accuracy
    ## size optimized for word document
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(7.0,2.5))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 1])
    ax0 = fig.add_subplot(spec[0])
    ax0.plot(xpoints,history['accuracy'])
    ax0.plot(xpoints,history['val_accuracy'])
    ax0.grid()
    ax0.set_title('a) dokładność')
    ax0.set_ylabel('dokładność')
    ax0.set_xlabel(xname)
    #ax0.set_xlim([0, 10])
    ax0.legend(['uczenie', 'walidacja'])
    plt.yscale('linear')
    
    ## summarize history for loss
  
    ax1 = fig.add_subplot(spec[1])
    ax1.plot(xpoints,history['loss'])
    ax1.plot(xpoints,history['val_loss'])
    ax1.grid()
    ax1.set_title('b) strata')
    ax1.set_ylabel('strata')
    ax1.set_xlabel(xname)
    #ax1.set_xlim([0, 10])
    ax1.legend(['uczenie', 'walidacja'])
    plt.yscale('log')
    plt.setp(ax1.get_yticklabels(), fontsize=7)
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(directory,unique_name+"_"+str(epochs))+".png",
        bbox_inches='tight',transparent = True, dpi=600)
    plt.close()

def compare_history(dataset_directory, epochs, models, legends, timetable, timed):
    #models = ["basic", "xception", "resnet50v2" , "mobilev2", "dense201", "efficientB5"]
    #timetable =[83, 223, 191, 111, 390, 713]
    if dataset_directory == "flower299":
        timetable=np.multiply(timetable,2.899/60)
    elif dataset_directory == "fruits":
        timetable=np.multiply(timetable,0.599/60)
    elif dataset_directory == "flowers":
        timetable=np.multiply(timetable,0.108/60)

    topics = ["accuracy", "val_accuracy", "loss", "val_loss"]
    topics_label = ["dokładność w uczeniu", "dokładność w walidacji", "strata w uczeniu", "strata w walidacji"]


    directory = os.path.join("saves", "graphics", dataset_directory)
    Path(directory).mkdir(parents=True, exist_ok=True)

    for topic, topic_label in zip(topics, topics_label):
        print("Drawing comparison of: ",topic)
        plt.figure(figsize=(7.0,3.5))
        plt.tight_layout()
        

        if ((topic == 'loss') or (topic == 'val_loss')):
            plt.yscale('log')            
        else:
            plt.yscale('linear')            
                    
        xpoints = range(1,epochs+1)
        xname = "epoka"

        for time_factor, model in zip(timetable, models):
            
            xpoints = range(1,epochs+1)
            xname = "epoka"
            if timed:
                xpoints = np.multiply(xpoints,time_factor)
                xname = "czas [min]"
            
            history=load_history(dataset_directory, model, epochs)
            plt.plot(xpoints,history[topic])
            
        if not timed:
            plt.xticks(range(0,55,5))
        else:
            plt.xticks(range(0,240,30))
        plt.ylabel(topic_label)
        
        plt.xlabel(xname)
        plt.legend(legends)
        plt.grid()
        plt.savefig(os.path.join(directory,"compare_"+topic+"_"+str(timed)+".png"),
            bbox_inches='tight',transparent = True, dpi = 600)
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
#models = ["basic", "xception", "resnet50v2" , "mobilev2", "dense201", "efficientB5"]
models = ["basic", "xception", "resnet50v2" , "mobilev2", "dense201", "efficientB5"]
legends = ["Basic", "Xception", "ResNet50V2" , "MobileNetV2", "DenseNet201", "EfficientNetB5"]
timetable =[83, 223, 191, 111, 390, 713]

dataset_directory = "fruits"
model_choice = "save_best.h5"
epochs = 50


for i in range(len(models)):
    model_variant = models[i]
    unique_name = model_variant + ""
    history=load_history(dataset_directory,model_variant, epochs)
    plot_history(history,dataset_directory,model_variant, epochs, unique_name)

    model=load_model(dataset_directory, model_variant, model_choice)
    draw_model(model_variant,model)


compare_history(dataset_directory, epochs, models, legends, timetable, 0)
compare_history(dataset_directory, epochs, models, legends, timetable, 1)