import torch
from meshnet_gn import enMesh_checkpoint
from meshnet2tfjs import meshnet2tfjs
import os

# Define model parameters based on the benchmark script
in_channels = 1    # Input channels (typically 1 for MRI data)
n_classes = 2      # Output classes (2 classes in the benchmark script)
channels = 15      # Channels per layer (15 in the benchmark script)
config_file = "mindgrabAE.json"  # Using the available config file

# Load the model architecture
model = enMesh_checkpoint(in_channels, n_classes, channels, config_file)

# Load weights from the state dict
model.load_state_dict(torch.load("mindgrab.pth", map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Define the output directory
output_dir = "mindgrab_tfjs"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert the model to TFJS format
meshnet2tfjs(model, output_dir)

print(f"Conversion complete. Model files saved to {output_dir}/")