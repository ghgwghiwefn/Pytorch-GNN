import random
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import find_objects
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import GCNConv
from torchvision.transforms import ToTensor
from torch_geometric.data import Batch

import Data_cleanup
import Graph_preprocessing_functions
import HyperParameters
import Utils as U
from Dataset import SLICDataset
import Data_Visualization
from GNN_Model import GNN

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    training_data = torch.load((U.TEST_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.TEST_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.TEST_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.TEST_DATA_FOLDER / 'testing_labels.npy').resolve())

    print(training_data)

    # Extract the images, graphs, and edges
    '''normalized_training_images = training_data['images']  # Images (already tensors)
    training_edge_indices = training_data['edge_indices']  # Edge indices
    training_node_features = training_data['node_features']  # Node features

    normalized_testing_images = testing_data['images']  # Images (already tensors)
    testing_edge_indices = testing_data['edge_indices']  # Edge indices
    testing_node_features = testing_data['node_features']  # Node features'''

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    Data_Visualization.visualize_data()
    training_data = torch.load((U.TEST_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.TEST_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.TEST_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.TEST_DATA_FOLDER / 'testing_labels.npy').resolve())

    # Further preprocessing (assuming you generate node features and edges during cleanup)
    # Example: create_edge_index_from_slic(segments) and compute_node_features(image, segments)

###Finish loading data###
print("Training data and labels")
print(len(training_data))
print(len(training_labels))
print("------------------")

#Make sure graphs have loaded properly
try:
    G = to_networkx(training_data[0], node_attrs=['color', 'eccentricity', 'aspect_ratio'])
    Graph_preprocessing_functions.draw_graph(G)
except Exception as e:
    print("Exception occurred:", e)
    G = to_networkx(training_data[0], node_attrs=['color', 'eccentricity', 'aspect_ratio'])
    print(G)

### HYPER PARAMETERS ###
CLASSES = HyperParameters.CLASSES
BATCH_SIZE = HyperParameters.BATCH_SIZE
HIDDEN_UNITS = HyperParameters.HIDDEN_UNITS
OUTPUT_SHAPE = len(CLASSES)
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.EPOCHS

#group the graphs and labels together for the DataLoader:
training_group = []
testing_group = []

for graph, label in zip(training_data, training_labels):
    data = Data(
        x=torch.cat([graph.color, graph.eccentricity.unsqueeze(1), graph.aspect_ratio.unsqueeze(1)], dim=1),
        edge_index=graph.edge_index,  # Edge indices
        y=torch.tensor([label]),  # Ensure label is in tensor format
        color=graph.color,  # Node colors
        eccentricity=graph.eccentricity,  # Eccentricity of the nodes
        aspect_ratio=graph.aspect_ratio,  # Aspect ratios of the nodes
        num_nodes=graph.num_nodes,  # Total number of nodes
        num_edges=graph.num_edges  # Total number of edges
    )
    training_group.append(data)
print(f"View a graph data object: {training_group[0]}")

for graph, label in zip(testing_data, testing_labels):
    data = Data(
        x=torch.cat([graph.color, graph.eccentricity.unsqueeze(1), graph.aspect_ratio.unsqueeze(1)], dim=1),
        edge_index=graph.edge_index,  # Edge indices
        y=torch.tensor([label]),  # Ensure label is in tensor format
        color=graph.color,  # Node colors
        eccentricity=graph.eccentricity,  # Eccentricity of the nodes
        aspect_ratio=graph.aspect_ratio,  # Aspect ratios of the nodes
        num_nodes=graph.num_nodes,  # Total number of nodes
        num_edges=graph.num_edges  # Total number of edges
    )
    testing_group.append(data)

#Load the data into training batches.
training_batches = DataLoader(training_group, batch_size=BATCH_SIZE, shuffle=True)
testing_batches = DataLoader(testing_group, batch_size=BATCH_SIZE, shuffle=False)

print("Batch Lengths")
print(len(training_batches))
'''
for batch in training_batches:
    print(f'\n\nTraining Batch:')
    print(f'Number of graphs in batch: {batch.num_graphs}')
    print(f'Batch node features: {batch.x}')  # Node features (if this contains other features)
    print(f'Batch edge indices: {batch.edge_index}')  # Edge indices
    print(f'Batch labels: {batch.y}')  # Labels (if present)
    print(f'Batch size: {batch.batch.size()}')  # Total number of nodes in the batch
    
    # Assuming your features are stored in the Data object:
    print(f'Batch color features: {batch.color}')  # Color feature
    print(f'Batch eccentricity features: {batch.eccentricity}')  # Eccentricity feature
    print(f'Batch aspect ratio features: {batch.aspect_ratio}')  # Aspect ratio feature

    break  # Remove this line if you want to see all batches
'''
#DECLARE MODEL INSTANCE WITH INPUT DIMENSION
Test_Model = GNN(input_dim=3+1+1) # -3 color(R,G,B) + 1 Eccentricity + 1 Aspect_ratio
#Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Test_Model.parameters(), lr=LEARNING_RATE)

#Make Accuracy function
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

'''
Training Loop
#1. Forward Pass
#2. Calculate the loss on the model's predictions
#3. Optimizer
#4. Back Propagation using loss
#5. Optimizer step
'''
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}\n---------")
    Test_Model.train()
    training_loss = 0
    for batch_idx, batch_graphs in enumerate(training_batches):
        #Get the batch of features to send to the model
        x = batch_graphs.x
        y = batch_graphs.y #Get the labels in y
        edge_index = batch_graphs.edge_index
        batch = batch_graphs.batch
        #1: Get predictions from the model
        y_pred = Test_Model(x, edge_index, batch)
        
        #2: Calculate the loss on the model's predictions
        loss = loss_fn(y_pred, y) 
        training_loss += loss.item() #Keep track of each batch's loss

        #3: optimizer zero grad
        optimizer.zero_grad()

        #4: loss back prop
        loss.backward()

        #5: optimizer step:
        optimizer.step()
    #Finish training batch and calculate the average loss:
    training_loss /= len(training_batches)

    #Move to testing on the testing data
    print("Testing the Model...")
    testing_loss, test_acc = 0, 0 #Metrics to test how well the model is doing
    Test_Model.eval()
    with torch.inference_mode():
        for batch_idx, batch_graphs in enumerate(testing_batches):
            #Get the batch features to send to the model again
            x = batch_graphs.x
            y = batch_graphs.y #Get the labels in y
            edge_index = batch_graphs.edge_index
            batch = batch_graphs.batch
            #1: Model Prediction
            y_pred = Test_Model(x, edge_index, batch)

            #2: Calculate loss
            loss = loss_fn(y_pred, y)
            testing_loss += loss.item()
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    print(f"Train loss: {training_loss:.4f} | Test loss: {testing_loss:.4f} | Test acc: {test_acc:.4f}%")

