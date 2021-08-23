
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as path_effects
from pathlib import Path
import numpy as np
import os

def load_model(dataset,model_variant,model_choice):
    model = keras.models.load_model(os.path.join("saves", dataset, model_variant, model_choice))
    return model

def inference(dataset,model_variant,model_choice,unique): 
    model = load_model(dataset,model_variant,model_choice)
    txtfile = open(os.path.join("datasets", dataset, "classes.txt"))
    class_names = txtfile.read()
    class_names = class_names.split(",")
    classes = len(class_names)
    indices = np.arange(classes)

    significant_floor = 1/classes

    folder_path=os.path.join("test_images",dataset)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        image_size = (180, 180) 
        img = keras.preprocessing.image.load_img(fpath, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions
        score = np.vstack([score, indices])
        if(classes > 5): 
            #sorting from most likely class if there is >5 classes
            #for clarity when plotting
            score = score[:, score[0, :].argsort()[::-1][:classes]]
        names = list()
        values = list()

        print(" Inference on: ", fpath)
        for i in range(5):
            print('%12s'%class_names[int(score[1,i])],":", ('{:.1%}'.format(score[0,i])))
            names.append(class_names[int(score[1,i])][0:14])
            values.append(100*score[0,i])
        #Drawing inference
        directory = os.path.join("saves", "graphics", dataset, "inference", unique)
        Path(directory).mkdir(parents=True, exist_ok=True)
        print("Drawing plots")          

        plt.rcParams.update({'font.size': 7})
        fig = plt.figure(figsize=(7.0,1.5))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1.5, 5.5])
        ax0 = fig.add_subplot(spec[0])
        ax0.imshow(img)
        end=len(fname)-4
        if end > 12:
            end=12
        ax0.set_title(fname[0:end])
        ax0.axes.xaxis.set_ticks([])
        ax0.axes.yaxis.set_ticks([])
        ax1 = fig.add_subplot(spec[1])
        ax1.set_title("Dopasowanie wg klas [%]")
        #ax1.grid()
        ax1.set_ylim([0, 110])
        ax1.bar(names, values)
        for i in range(5):
            if(values[i] <10):
                ax1.text(i-0.17, values[i]+5, ('{:.1%}'.format(values[i]/100)), size=8)
            elif(values[i] <80):
                ax1.text(i-0.22, values[i]+5, ('{:.1%}'.format(values[i]/100)), size=8)
            else:
                ax1.text(i-0.25, values[i]-15, ('{:.1%}'.format(values[i]/100)), 
                    color='white', fontweight='semibold', size=8)
                       
        plt.savefig(os.path.join(directory,model_variant+"_"+fname[0:12])+".png",
            bbox_inches='tight',transparent = True, dpi=600)
        plt.close()

inference("flower299","advanced","save_30.h5","testowe2")