import torch
from Graph_preprocessing_functions import make_graph_for_image_slic, draw_graph, show_comparison, show_comparison_no_label, convert_to_data
from Data_cleanup import load_and_preprocess_images
import Utils as U
from GNN_Model import GNN
import HyperParameters
from skimage.segmentation import slic
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
import random
from PIL import Image
import requests
from io import BytesIO
import numpy as np

device = HyperParameters.device

Model_0 = GNN(input_dim=HyperParameters.input_dim)
try:
    Model_0.load_state_dict(torch.load((U.MODEL_FOLDER / 'Model_0.pth').resolve()))
except:
    Model_0.load_state_dict(torch.load((U.MODEL_FOLDER / 'Model_0.pth').resolve(), map_location=torch.device('cpu')))
Model_0.to(device)

def model_on_test_data():
    num_correct = 0
    amount = 10
    images, labels = load_and_preprocess_images(U.testing_folders)
    for x in range(amount):
        img_index = random.randint(0, len(images)-1)
        label = labels[img_index]
        img = images[img_index]
        input('press enter to continue...')

    print(f"Percentage correct: {num_correct/amount*100:2f}%")

def model_on_new_images():
    # Replace this with your image URL
    while True:
        image_url = input("Enter image URL: ")
        try:
            # Fetch the image from the URL
            response = requests.get(image_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Open the image using Pillow and convert to RGB
                img = Image.open(BytesIO(response.content)).convert("RGB").resize(HyperParameters.target_size)
                
                # Convert the image to a numpy array and normalize pixel values
                img_array = np.array(img, dtype=np.float32) / 255.0
                segments = slic(img_array, n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
                show_comparison_no_label(img_array, segments)
                make_prediction(img_array)

        except: 
            print("Invalid Image")

def make_prediction(img):
    img_graph = make_graph_for_image_slic(img)
    draw_graph(img_graph)

    graph = from_networkx(img_graph)
    data = convert_to_data(graph)
    data = data.to(device)

    Model_0.eval()
    with torch.no_grad():
        x = data.x
        edge_index = data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long)  # All nodes belong to the same graph, so all batch indices are 0
        batch = batch.to(device)
        prediction = F.softmax(Model_0.forward(x, edge_index, batch))
        predicted_class = prediction.argmax(dim=1)
        print(f"\nPredicted class: {HyperParameters.CLASSES[predicted_class]}; Confidence: {prediction[0][predicted_class].item()*100:.2f}%")
        return predicted_class

if __name__ == "__main__":
    model_on_new_images()