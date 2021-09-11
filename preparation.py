import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

def cleaning(choice):
    """
    ## Load the data
    ## Filter out corrupted images
    """
    txtfile = open(os.path.join("datasets", choice, "classes.txt"))
    class_names = txtfile.read()
    class_names = class_names.split(",")

    num_skipped = 0
    for folder_name in class_names:
        folder_path = os.path.join("datasets", choice, folder_name)
        num_images = 0
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
            elif is_jfif:
                num_images += 1
        print("Found %d" %num_images, "images in '%s' class" %folder_name)
    print("Deleted %d images" % num_skipped)

def get_dataset_label_numbers(path):
    all_label_paths = get_all_subfolders(path)
    result_array = []
    for path in all_label_paths:
        result_array.append(get_number_of_files(path=path))
    return np.array(result_array)

def get_number_of_files(path=None):
    if path:
        return len([f for f in os.scandir(path) if f.is_file()])
    return None

def get_all_subfolders(path=None):
    if path:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    return None

def get_dataset_dimensions(path=None, label_numbers=None):
    all_label_paths = get_all_subfolders(path)
    dimensions = []
    for index, label_path in enumerate(all_label_paths):
        print(label_path)
        label_width = 0.0
        label_height = 0.0
        for image_file in os.scandir(label_path):
            if image_file.is_file():
                specimen_image = Image.open(image_file.path)
                label_width += specimen_image.size[1]
                label_height += specimen_image.size[0]
                specimen_image.close()
        dimensions.append([label_height / label_numbers[index], label_width / label_numbers[index]])
    return np.array(dimensions)

def statistics(choice):
    """
    Produces statistics for the dataset (avgs, means, stds etc)
    """
    paths = os.path.join("datasets", choice)
    nr_specimens = get_dataset_label_numbers(paths)
    print(np.sum(nr_specimens))
    label_width_x_height_values = get_dataset_dimensions(paths, label_numbers=nr_specimens)
    label_height_values = np.array([x[0] for x in label_width_x_height_values])
    label_width_values = np.array([x[1] for x in label_width_x_height_values])
    textt = str("Liczba klas: "+ str(len(nr_specimens))+
        "\nŚrednia liczba zdjęć na klasę: "+ str(np.average(nr_specimens))+
        "\nMediana liczby zdjęć na klasę: "+ str(np.median(nr_specimens))+
        "\nOdchylenie std zdjęć na klasę: "+ str(np.std(nr_specimens))+
        "\nŚrednia szerokość obrazu: "+ str(np.average(label_width_values))+
        "\nMediana szerokości obrazu: "+ str(np.median(label_width_values))+
        "\nOdchylenie std szerokości zdjęć: "+ str(np.std(label_width_values))+
        "\nŚrednia wysokość obrazu: "+ str(np.average(label_height_values))+
        "\nMediana wysokości obrazu: "+ str(np.median(label_height_values))+
        "\nOdchylenie std wysokości zdjęć: "+ str(np.std(label_height_values)))
    directory = os.path.join("saves", "graphics", choice)
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(directory,"stats.txt"), "w") as text_file:
        text_file.write(textt) 
    

def load_ds():
    ##Load dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join("datasets", choice),
        image_size=(224, 224),
        batch_size=32,
    )
    ##Prefetch
    train_ds = train_ds.prefetch(buffer_size=32)
    return train_ds

def examples(choice):
    txtfile = open(os.path.join("datasets", choice, "classes_pl.txt"), encoding="utf-8")
    class_names = txtfile.read()
    class_names = class_names.split(",")    
    train_ds=load_ds()
    fig = plt.figure(figsize=(7,3.5))
    plt.rcParams.update({'font.size': 7})
    for images, labels in train_ds.take(1):
        for i in range(1, 9):
            fig.add_subplot(2, 4, i)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    directory = os.path.join("saves", "graphics",choice)
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(directory,"examples_"+choice+".png"),
        bbox_inches='tight',transparent = True, dpi=600)
    plt.close()


choices= ["fruits", "flowers", "flower299"]  

for choice in choices:
    cleaning(choice)
    statistics(choice)
    examples(choice)

