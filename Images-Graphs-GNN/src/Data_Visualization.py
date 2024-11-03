import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

import Graph_preprocessing_functions
import HyperParameters
import Utils as U
from Utils import training_folders, testing_folders
import random
import Data_cleanup

def visualize_data():
    training_data, training_labels = Data_cleanup.load_and_preprocess_images(training_folders)
    testing_data, testing_labels = Data_cleanup.load_and_preprocess_images(testing_folders)
    sample_size = 64
    view_size = 5
    training_indexes = random.sample(range(len(training_data)), min(sample_size, len(training_data)))
    testing_indexes = random.sample(range(len(testing_data)), min(sample_size, len(testing_data)))

    processed_training_graphs = []
    processed_testing_graphs = []
    processed_training_labels = []
    processed_testing_labels = []

    for i, x in enumerate(training_indexes):
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(training_data[x])
        if i < view_size:
            segments = slic(training_data[x], n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
            Graph_preprocessing_functions.show_comparison(training_data[x], training_labels[x], segments)
            Graph_preprocessing_functions.draw_graph(graph)
        processed_training_graphs.append(graph)
        processed_training_labels.append(training_labels[x])

    for i, x in enumerate(testing_indexes):
        graph = Graph_preprocessing_functions.make_graph_for_image_slic(testing_data[x])
        if i < view_size:
            segments = slic(testing_data[x], n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
            Graph_preprocessing_functions.show_comparison(testing_data[x], testing_labels[x], segments)
            Graph_preprocessing_functions.draw_graph(graph)
        processed_testing_graphs.append(graph)
        processed_testing_labels.append(testing_labels[x])
    
    training_tensor = [from_networkx(G) for G in processed_training_graphs]  # Convert to PyTorch Geometric Data objects
    testing_tensor =  [from_networkx(G) for G in processed_testing_graphs]  # Convert to PyTorch Geometric Data objects
    print(training_tensor)
    print(training_tensor[0])

    print("Saving processed graphs...")
    torch.save(training_tensor, (U.TEST_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    torch.save(testing_tensor, (U.TEST_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    
    print("Saving labels...")
    np.save((U.TEST_DATA_FOLDER / 'training_labels.npy').resolve(), processed_training_labels)
    np.save((U.TEST_DATA_FOLDER / 'testing_labels.npy').resolve(), processed_testing_labels)
    print(f'Numebr of test data: {len(training_tensor)}')
    return training_tensor, testing_tensor


if __name__ == "__main__":
    visualize_data()