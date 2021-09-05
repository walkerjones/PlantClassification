import tensorflow as tf
import numpy as np
from PIL import Image
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

def labels_under_value(labels_numbers, value):
    result = []
    for el in labels_numbers:
        if el < value:
            result.append(el)
    if not result:
        result = [0.0]
    return np.array(result)

def statistics(choice):
    """
    Produces statistics for the dataset (avgs, means, stds etc)
    """
    choice = os.path.join("datasets", choice)
    nr_specimens = get_dataset_label_numbers(choice)
    print(np.sum(nr_specimens))
    label_width_x_height_values = get_dataset_dimensions(choice, label_numbers=nr_specimens)
    label_height_values = np.array([x[0] for x in label_width_x_height_values])
    label_width_values = np.array([x[1] for x in label_width_x_height_values])
    print("Liczba klas: ", len(nr_specimens))
    print("Średnia liczba zdjęć na klasę: ", np.average(nr_specimens))
    print("Mediana liczby zdjęć na klasę: ", np.median(nr_specimens))
    print("Odchylenie std zdjęć na klasę: ", np.std(nr_specimens))
    print("Średnia szerokość obrazu: ", np.average(label_width_values))
    print("Mediana szerokości obrazu: ", np.median(label_width_values))
    print("Odchylenie std szerokości zdjęć: ", np.std(label_width_values))
    print("Średnia wysokość obrazu: ", np.average(label_height_values))
    print("Mediana wysokości obrazu: ", np.median(label_height_values))
    print("Odchylenie std wysokości zdjęć: ", np.std(label_height_values))

##choice: fruits/flowers/flower299
choice = "fruits"

cleaning(choice)
statistics(choice)