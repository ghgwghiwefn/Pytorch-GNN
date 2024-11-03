import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import Data_cleanup
import HyperParameters
import Utils as U
from GNN_Model import GNN
from Graph_preprocessing_functions import is_grey, draw_graph, convert_to_data
device = HyperParameters.device
print(device)

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve())

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
    Data_cleanup.clean_data()
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_training_graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'processed_testing_graphs.pt').resolve())
    training_labels = np.load((U.CLEAN_DATA_FOLDER / 'training_labels.npy').resolve())
    testing_labels = np.load((U.CLEAN_DATA_FOLDER / 'testing_labels.npy').resolve())

    # Further preprocessing (assuming you generate node features and edges during cleanup)
    # Example: create_edge_index_from_slic(segments) and compute_node_features(image, segments)

#LABELS
###Finish loading data###

def show_image(x, y):
    fig = plt.figure("Superpixels -- %d segments" % (50))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x)
    plt.title(y)
    plt.axis("off")
    plt.show()

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
    data = convert_to_data(graph, label)
    training_group.append(data)

'''for graph in training_group:
    for label in set(training_labels):'''


print(f"View a graph data object: {training_group[0]}")

for graph, label in zip(testing_data, testing_labels):
    data = convert_to_data(graph, label)
    testing_group.append(data)

#Load the data into training batches.
training_batches = DataLoader(training_group, batch_size=BATCH_SIZE, shuffle=True)
testing_batches = DataLoader(testing_group, batch_size=BATCH_SIZE, shuffle=False)

print("Batch Lengths")
print(len(training_batches))

#DECLARE MODEL INSTANCE WITH INPUT DIMENSION
# Before the model call
Model_0 = GNN(input_dim=HyperParameters.input_dim) # -3 color(R,G,B) + 1 Eccentricity + 1 Aspect_ratio + 1 solidity
Model_0.to(device)
#Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model_0.parameters(), lr=LEARNING_RATE)

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

# Lists to store loss values
train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize best validation loss as infinity
patience = HyperParameters.PATIENCE  
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch: {epoch}\n---------")
    Model_0.train()
    training_loss = 0
    for batch_idx, batch_graphs in enumerate(training_batches):
        #Get the batch of features to send to the model
        batch_graphs = batch_graphs.to(device)
        x = batch_graphs.x
        y = batch_graphs.y.to(device).long()  #Get the labels in y
        edge_index = batch_graphs.edge_index
        batch = batch_graphs.batch
        #1: Get predictions from the model
        y_pred = Model_0(x, edge_index, batch)
        
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
    train_losses.append(training_loss)

    #Move to testing on the testing data
    print("Testing the Model...")
    testing_loss, test_acc = 0, 0 #Metrics to test how well the model is doing
    Model_0.eval()
    with torch.inference_mode():
        for batch_idx, batch_graphs in enumerate(testing_batches):
            #Get the batch features to send to the model again
            batch_graphs = batch_graphs.to(device)
            x = batch_graphs.x
            y = batch_graphs.y.to(device).long()  #Get the labels in y
            edge_index = batch_graphs.edge_index
            batch = batch_graphs.batch
            #1: Model Prediction
            y_pred = Model_0(x, edge_index, batch)

            #2: Calculate loss
            loss = loss_fn(y_pred, y)
            testing_loss += loss.item()
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    testing_loss /= len(testing_batches)
    test_acc /= len(testing_batches)
    val_losses.append(testing_loss)
    print(f"Train loss: {training_loss:.4f} | Test loss: {testing_loss:.4f} | Test acc: {test_acc:.4f}%")

    # Check if current validation loss is the best so far
    if testing_loss < best_val_loss:
        best_val_loss = testing_loss
        # Save the model's parameters (state_dict) to a file
        torch.save(Model_0.state_dict(), (U.MODEL_FOLDER / 'Model_0.pth').resolve())
        print(f'Saved best model with validation loss: {best_val_loss:.4f}')
        epochs_no_improve = 0  # Reset counter if improvement
    else:
        epochs_no_improve += 1
        print(f'Num epochs since improvement: {epochs_no_improve}')
        #stop training if overfitting starts to happen
        if epochs_no_improve >= patience:
            print("Early stopping")
            break

# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
