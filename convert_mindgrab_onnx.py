import torch
from meshnet_gn import enMesh_checkpoint
import os

# Define model parameters based on the benchmark script
in_channels = 1    # Input channels (typically 1 for MRI data)
n_classes = 2      # Output classes (2 classes in the benchmark script)
channels = 15      # Channels per layer (15 in the benchmark script)
config_file = "mindgrabAE.json"  # Using the available config file
cube_size = 256    # Input cube size

# Load the model architecture
model = enMesh_checkpoint(in_channels, n_classes, channels, config_file)

# Load weights from the state dict
model.load_state_dict(torch.load("mindgrab.pth", map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor for ONNX export
dummy_input = torch.randn(1, 1, cube_size, cube_size, cube_size)

# Define the output file
output_file = "mindgrab.onnx"

# Export the model to ONNX format
torch.onnx.export(
    model,                      # model being run
    dummy_input,                # model input (or a tuple for multiple inputs)
    output_file,                # where to save the model
    export_params=True,         # store the trained parameter weights inside the model file
    opset_version=12,           # the ONNX version to export the model to
    do_constant_folding=True,   # whether to execute constant folding for optimization
    input_names=['input'],      # the model's input names
    output_names=['output'],    # the model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},    # variable length axes
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported to {output_file}")