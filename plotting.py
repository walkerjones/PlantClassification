from ast import NameConstant
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as path_effects
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
            #correction for shorter history
            if model == "efficientB5":
                epochs = 25
            else:
                epochs = 50
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
            plt.xticks(range(0,40,5))
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

def compare_evaluation():
    datasets = ["flowers", "fruits", "flower299"]
    directory = os.path.join("saves", "graphics")
    Path(directory).mkdir(parents=True, exist_ok=True)
    names = ["Xception", "ResNet50V2" , "MobileNetV2", "DenseNet201", "EfficientNetB5"]
    for dataset in datasets:
        if dataset == "flower299":
            amount = 5
            accs=[0.633, 0.600, 0.466, 0.5, 0.466]
            losses=[2.193, 2.65, 1.744, 2.53, 1.617]
        elif dataset == "fruits":
            amount = 5
            accs=[0.923, 0.807, 0.807, 0.9615, 0.846]#, 0.134]
            losses=[0.2355, 0.6239,  1.107, 0.092, 0.553]#, 14.78]
        else:
            amount = 5
            accs=[0.888, 0.814, 0.703, 0.925, 0.666]#, 0.259]
            losses=[0.249, 0.923, 1.44, 0.286, 2.950]#, 10.8]
        plt.rcParams.update({'font.size': 7})
        plt.figure(figsize=(7.0,2))
        #spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2,2])
        #ax0 = fig.add_subplot(spec[0])
        #ax1 = fig.add_subplot(spec[1])
        #plt.ylim([0, 110])
        plt.xticks(range(5),names)
        X=np.arange(amount)
        for xrift, values in zip([-0.2, 0.2],[accs, losses]):
            plt.bar(X+xrift, values, width = 0.4)

        for i in range(amount):
            if(accs[i] <0.1):
                plt.text(-0.2+i-0.12, accs[i]*1.1, ('{:.1%}'.format(accs[i])), size=7)
            elif(accs[i] < 0.8):
                plt.text(-0.2+i-0.17, accs[i]*1.1, ('{:.1%}'.format(accs[i])), size=7)
            elif(accs[i] < 1):
                tekst = plt.text(-0.2+i-0.15, accs[i]*0.77, ('{:.1%}'.format(accs[i])), 
                    color='white', fontweight='semibold', size=7)
                tekst.set_path_effects([path_effects.withStroke(linewidth=1, foreground='gray')])
            else:
                tekst = plt.text(-0.2+i-0.20, accs[i]*0.85, ('{:.1%}'.format(accs[i])), 
                    color='white', fontweight='semibold', size=7)
                tekst.set_path_effects([path_effects.withStroke(linewidth=1, foreground='gray')])
        for i in range(amount):
            if(losses[i] <0.1):
                plt.text(0.2+i-0.11, losses[i]*1.2, ('{:.1}'.format(losses[i])), size=8)
            elif(losses[i] < 0.8):
                plt.text(0.2+i-0.09, losses[i]*1.1, ('{:.1}'.format(losses[i])), size=8)
            elif(losses[i] < 1):
                tekst = plt.text(0.2+i-0.08, losses[i]*0.7, ('{:.1}'.format(losses[i])), 
                    color='white', fontweight='semibold', size=8,)
                tekst.set_path_effects([path_effects.withStroke(linewidth=1, foreground='gray')])
            else:
                tekst = plt.text(0.2+i-0.09, losses[i]*0.85, ('{:.2}'.format(losses[i])), 
                    color='white', fontweight='semibold', size=8)
                tekst.set_path_effects([path_effects.withStroke(linewidth=1, foreground='gray')])
                        
        plt.legend(['dokładność', 'strata'])
        plt.savefig(os.path.join(directory,dataset,"evaluations.png"),
            bbox_inches='tight',transparent = True, dpi=600)
        plt.close()

models = ["xception", "resnet50v2" , "mobilev2", "dense201", "efficientB5", "basic"]
legends = ["Xception", "ResNet50V2" , "MobileNetV2", "DenseNet201", "EfficientNetB5", "Basic"]
timetable =[223, 191, 111, 390, 713, 83]

dataset_directory = "flowers"
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
#compare_evaluation()