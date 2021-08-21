import tensorflow as tf
import numpy as np
from PIL import Image
import os

def cleaning(choice):
    """
    ## Load the data
    ## Filter out corrupted images
    """
    txtfile = open(os.path.join(choice, "classes.txt"))
    class_names = txtfile.read()
    class_names = class_names.split(",")

    num_skipped = 0
    for folder_name in class_names:
        folder_path = os.path.join(choice, folder_name)
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

def statistics(choice, goal_nr_per_label):
    """
    Produces statistics for the dataset (avgs, means, stds etc)
    """
    nr_specimens_per_fruit = get_dataset_label_numbers(choice)
    print(np.sum(nr_specimens_per_fruit))
    label_width_x_height_values = get_dataset_dimensions(choice, label_numbers=nr_specimens_per_fruit)
    label_height_values = np.array([x[0] for x in label_width_x_height_values])
    label_width_values = np.array([x[1] for x in label_width_x_height_values])
    labels_under = labels_under_value(nr_specimens_per_fruit, goal_nr_per_label)
    print("Total number of labels: ", len(nr_specimens_per_fruit))
    print("Target image number for label: ", goal_nr_per_label)
    print("Labels Under target value: ", len(labels_under) - 1 * (labels_under[0] == 0.0))
    print("Average missing images: ", np.average(labels_under))
    print("Median missing images: ", np.median(labels_under))
    print("Standard deviation of missing images: ", np.std(labels_under))
    print("Average nr of images per label: ", np.average(nr_specimens_per_fruit))
    print("Median nr of images per label: ", np.median(nr_specimens_per_fruit))
    print("Standard deviation of images per label: ", np.std(nr_specimens_per_fruit))
    print("Average width of image: ", np.average(label_width_values))
    print("Median width of image: ", np.median(label_width_values))
    print("Standard deviation of width of image: ", np.std(label_width_values))
    print("Average height of image: ", np.average(label_height_values))
    print("Median height of image: ", np.median(label_height_values))
    print("Standard deviation of height of image: ", np.std(label_height_values))

##choice: fruits/flowers/flower299

##cleaning("fruits")
statistics("fruits",1000)