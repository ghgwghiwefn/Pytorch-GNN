import glob
import os
import multiprocessing as mp

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from skimage.io import imread
from torch_geometric.utils import from_networkx

import Graph_preprocessing_functions
import HyperParameters
import Utils as U
from Utils import training_folders, testing_folders
import random
import matplotlib.pyplot as plt

#Check test image 1066
def show_img(img, label):
  plt.imshow(img.permute(1, 2, 0))
  plt.title(HyperParameters.CLASSES[label])
  plt.show()

def load_and_preprocess_images(folders, target_size=HyperParameters.target_size, extensions=("jpg", "jpeg", "png", "gif")):
    images = []
    labels = []
    for label, folder in enumerate(folders):
        print(folder)
        print(label)
        num = 0
        for ext in extensions:
            for image_path in glob.glob(os.path.join(folder, f"*.{ext}")):
                try:
                    img = Image.open(image_path).convert("RGB").resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    # Get the height, width, and number of channels (if it's a color image)
                    '''height, width = img_array.shape[:2]
                    print(height, width)'''
                    images.append(img_array)
                    labels.append(label)
                    num+=1

                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")
        print(f'Amount: {num}')
    return np.array(images), np.array(labels)

def process_images_to_graphs(images, labels):
    processed_graphs = []
    stops = sorted(random.sample(range(len(images)), min(20, len(images)))) #create random stops for visualization
    print(stops)
    for i, image in enumerate(images):
        #Make the graph
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(image)
        #Add graph to graph list
        processed_graphs.append(graph)
        #Update progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} out of {len(images)} images")
        if i in stops and HyperParameters.show_visualization_stops:
            #Visualize random Graphs
            segments = slic(image, n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
            Graph_preprocessing_functions.show_comparison(image, labels[i], segments)
            Graph_preprocessing_functions.draw_graph(graph)
    return processed_graphs

def clean_data():
    classes = HyperParameters.CLASSES
    print(training_folders[0])
    print()

    print("Loading and preprocessing images...")
    training_data, training_labels = load_and_preprocess_images(training_folders)
    testing_data, testing_labels = load_and_preprocess_images(testing_folders)

    print("Processing training data to graphs...")
    processed_training_graphs = process_images_to_graphs(training_data, training_labels)

    print("Processing testing data to graphs...")
    processed_testing_graphs = process_images_to_graphs(testing_data, testing_labels)

    print("Saving processed graphs...")
    training_tensor = [from_networkx(G) for G in processed_training_graphs]  # Convert to PyTorch Geometric Data objects
    testing_tensor =  [from_networkx(G) for G in processed_testing_graphs]  # Convert to PyTorch Geometric Data objects
    print(training_tensor[0])
    torch.save(training_tensor, (U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    torch.save(testing_tensor, (U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    
    print("Saving labels...")
    np.save((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve(), training_labels)
    np.save((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve(), testing_labels)

    print("Data cleanup completed successfully.")

if __name__ == "__main__":
    clean_data()
