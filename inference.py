
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import numpy as np
import os

def load_model(dataset,model_variant,model_choice):
    model = keras.models.load_model(os.path.join("saves", dataset, model_variant, model_choice))
    return model

def classify(dataset,model_variant,model_choice,unique,verbosity): 
    model = load_model(dataset,model_variant,model_choice)
    txtfile = open(os.path.join("datasets", dataset, "classes_pl.txt"), encoding="utf-8")
    class_names = txtfile.read()

    class_names = class_names.split(",")
    classes = len(class_names)
    indices = np.arange(classes)

    folder_path=os.path.join("test_images",dataset)
    for fname in os.listdir(folder_path):
        for filename in os.listdir(os.path.join(folder_path, fname)):
            fpath = os.path.join(folder_path, fname, filename)
            
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
                if(verbosity!="silent"):
                    print('%12s'%class_names[int(score[1,i])],":", ('{:.1%}'.format(score[0,i])))
                names.append(class_names[int(score[1,i])][0:20])
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
            end=name_end=len(filename)-4
            if end > 16:
                name_end=16
            ax0.set_title(filename[0:name_end])
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
                elif(values[i] <99.95):
                    ax1.text(i-0.25, values[i]-15, ('{:.1%}'.format(values[i]/100)), 
                        color='white', fontweight='semibold', size=8)
                else:
                    ax1.text(i-0.3, values[i]-15, ('{:.1%}'.format(values[i]/100)), 
                        color='white', fontweight='semibold', size=8)

                        
            plt.savefig(os.path.join(directory,model_variant+"_"+filename[0:end])+".png",
                bbox_inches='tight',transparent = True, dpi=600)
            plt.close()

def evaluation(dataset,model_variant,model_choice,unique): 
    model = load_model(dataset,model_variant,model_choice)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join("test_images", dataset),
        image_size=image_size,
        batch_size=32)
    test_ds = test_ds.prefetch(buffer_size=32)
    score = model.evaluate(test_ds)
    textt=str(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    print(textt)
    directory = os.path.join("saves", "graphics", dataset, "inference", unique)
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(directory,"eval.txt"), "w") as text_file:
        text_file.write(textt) 


image_size = (192, 192) 
verbosity ="silent"
models = ["basic", "xception", "resnet50v2" , "mobilev2", "dense201", "efficientB5"]
dataset = "flowers"
model_choice = "save_best.h5"
epochs = 50

for i in range(len(models)):
    model_variant = models[i]
    unique_name = model_variant + ""
    classify(dataset, model_variant, model_choice, unique_name, verbosity)
    evaluation(dataset, model_variant, model_choice, unique_name)