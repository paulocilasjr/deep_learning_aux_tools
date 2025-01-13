import argparse
import os
import zipfile
import csv
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

# Function to load the model based on user input
def load_model(model_name):
    if model_name == "resnet":
        model = models.resnet50(pretrained=True)
    elif model_name == "vgg":
        model = models.vgg16(pretrained=True)
    elif model_name == "foundation":
        model = models.densenet121(pretrained=True)  # Example: DenseNet for foundation
    else:
        raise ValueError("Invalid model name. Choose from 'resnet', 'vgg', or 'foundation'.")
    model.eval()
    return model

# Function to process images and generate embeddings
def generate_embeddings(zip_file, model_name, output_file):
    # Load the model
    model = load_model(model_name)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification head

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract zip file and process images in order
    with zipfile.ZipFile(zip_file, 'r') as z:
        image_files = sorted(z.namelist())  # Ensure order is preserved
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(["Filename"] + [f"Dim_{i}" for i in range(2048)])  # Adjust dimensions if needed
            
            for image_file in image_files:
                try:
                    # Open image from zip archive
                    with z.open(image_file) as file:
                        image = Image.open(file).convert("RGB")
                        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

                        # Generate embedding
                        with torch.no_grad():
                            embedding = model(image_tensor).flatten().numpy()

                        # Write to CSV
                        csv_writer.writerow([image_file] + embedding.tolist())

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from images in a zip file.")
    parser.add_argument("--zip", required=True, help="Input zip file containing images.")
    parser.add_argument("--model", required=True, choices=["resnet", "vgg", "foundation"], help="Model to use for embedding generation.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    args = parser.parse_args()

    # Generate embeddings and save to CSV
    generate_embeddings(args.zip, args.model, args.output)
