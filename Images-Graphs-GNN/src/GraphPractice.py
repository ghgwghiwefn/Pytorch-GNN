import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage import color
from skimage.measure import regionprops, label
from HyperParameters import *

# Original image
def show_comparison(image, segments):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Superpixel image with boundaries
    ax2.imshow(mark_boundaries(image, segments))
    ax2.set_title('SLIC Segmentation')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

# Function to set node positions with a minimum distance
def set_node_positions(G, min_distance=2):
    pos = {}
    angle_step = 2 * np.pi / len(G.nodes())
    
    for i, node in enumerate(G.nodes()):
        angle = i * angle_step
        pos[node] = (min_distance * np.cos(angle), min_distance * np.sin(angle))
    
    return pos

def find_neighbors(segments, superpixel_id):
    """
    Find neighboring superpixels for a given superpixel ID.

    Parameters:
        segments (ndarray): 2D array where each element is the ID of the superpixel.
        superpixel_id (int): The ID of the superpixel to find neighbors for.

    Returns:
        list: A list of neighboring superpixel IDs.
    """
    # Get the indices of the given superpixel
    superpixel_indices = np.argwhere(segments == superpixel_id)
    
    # Initialize a set to hold neighboring superpixels
    neighbors = set()
    
    # Check for neighbors in the 8 surrounding pixels
    for idx in superpixel_indices:
        x, y = idx[0], idx[1]
        
        # Loop through neighboring coordinates
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx == 0 and dy == 0):  # Skip the center pixel
                    continue
                neighbor_x, neighbor_y = x + dx, y + dy
                
                # Check if the neighbor is within bounds
                if (0 <= neighbor_x < segments.shape[0] and
                    0 <= neighbor_y < segments.shape[1]):
                    neighbor_id = segments[neighbor_x, neighbor_y]
                    if neighbor_id != superpixel_id:  # Avoid adding the same superpixel
                        neighbors.add(neighbor_id)
    
    return list(neighbors)

def average_color_of_superpixel(image, segments, segment_id):
    # Get the mask for the superpixel
    mask = (segments == segment_id)

    # Calculate the average color
    average_color = image[mask].mean(axis=0)

    return average_color

def calculate_shape_features(image, segments, segment_id):
    """
    Calculate shape features for a given superpixel mask.

    Parameters:
        mask (ndarray): Binary mask of the superpixel (1s for superpixel pixels, 0s otherwise).

    Returns:
        dict: A dictionary with shape features.
    """
    # Label the mask to identify connected components
    mask = (segments == segment_id)
    labeled_mask = label(mask)
    
    # Calculate properties
    properties = regionprops(labeled_mask)

    # Assuming the mask represents a single superpixel
    if properties:
        region = properties[0]
        features = {
            'area': region.area,                 # Number of pixels in the superpixel
            'perimeter': region.perimeter,       # Perimeter of the superpixel
            'eccentricity': region.eccentricity, # Measure of how much the shape deviates from being circular
            'aspect_ratio': region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0,
            'solidity': region.solidity           # Proportion of the convex hull that is occupied by the shape
        }
        return features
    else:
        return None
    
def calculate_eccentricity(segments, segment_id):
    mask = (segments == segment_id)
    labeled_superpixel = label(mask)
    properties = regionprops(labeled_superpixel)
    region = properties[0]
    return region.eccentricity


def load_and_preprocess_images(folders, target_size=(128, 128), extensions=("jpg", "jpeg", "png", "gif")):
    images = []
    labels = []
    for label, folder in enumerate(folders):
        for ext in extensions:
            i = 0
            for image_path in glob.glob(os.path.join(folder, f"*.{ext}")):
                try:
                    img = Image.open(image_path).convert("RGB").resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(label)
                    i+=1
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")
                if i > 4:
                    break
            break
        break
    return np.array(images), np.array(labels)

project_folder = 'DataScienceProject/'
training_folders = [os.path.join(project_folder, "GNN_Dataset/seg_train/seg_train", class_name) 
                        for class_name in ["buildings", "forest", "glacier", "mountain", "sea", "street"]]

training_data, training_labels = load_and_preprocess_images(training_folders)

for x in range(len(training_data)):
    segments = slic(training_data[x], n_segments=50, sigma=sigma)

    show_comparison(training_data[x], segments)
    
    # Create the graph
    G = nx.Graph()

    # Step 2: Loop over each segment
    for segment_id in np.unique(segments):
        average_color = average_color_of_superpixel(training_data[x], segments, segment_id)
        eccentricity = calculate_eccentricity(segments, segment_id)
        neighbors = find_neighbors(segments, segment_id)

        #print(f'Eccentricity: {eccentricity}')
        G.add_node(segment_id, color=average_color, eccentricity=eccentricity)
        #print(neighbors)
        for neighbor in neighbors:
            G.add_edge(segment_id, neighbor)

    print(G)

# Extract node colors and weights
'''node_colors = [data['color'] for _, data in G.nodes(data=True)]
node_labels = {node: data['label'] for node, data in G.nodes(data=True)}'''

# Draw the graph with the correct node colors and labels
'''nx.draw(G, node_color=node_colors, node_size=1000, with_labels=True, labels=node_labels)

plt.margins(.2)
plt.show()'''

# Example of accessing a custom attribute
'''for node, data in G.nodes(data=True):
    print(f"Node {node} has color {data['color']}, weight {data['weight']}, label {data['label']}")'''