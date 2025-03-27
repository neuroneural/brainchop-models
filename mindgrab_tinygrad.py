import torch
import numpy as np
from tiny_meshnet import enMesh_checkpoint, torch_to_tinygrad
from meshnet_gn import enMesh_checkpoint as PyTorchMeshNet
from tinygrad import Tensor, nn

def main():
    # Define model parameters
    in_channels = 1    # Input channels (typically 1 for MRI data)
    n_classes = 2      # Output classes (2 classes in benchmark script)
    channels = 15      # Channels per layer (15 in benchmark script)
    config_file = "mindgrabAE.json"  # Config file

    print("Creating tinygrad MeshNet model...")
    # Create the tinygrad model
    tinygrad_model = enMesh_checkpoint(in_channels, n_classes, channels, config_file)
    
    print("Creating PyTorch MeshNet model...")
    # Create the PyTorch model
    pytorch_model = PyTorchMeshNet(in_channels, n_classes, channels, config_file)
    
    print("Loading PyTorch weights...")
    # Load PyTorch weights
    pytorch_state_dict = torch.load("mindgrab.pth", map_location=torch.device('cpu'))
    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()
    
    print("Converting PyTorch weights to tinygrad...")
    # Convert PyTorch weights to tinygrad
    tinygrad_model = torch_to_tinygrad(tinygrad_model, pytorch_state_dict)
    
    # Create a test input tensor (using a smaller size for quick testing)
    print("Creating test input tensor...")
    test_size = 32
    np_input = np.random.randn(1, 1, test_size, test_size, test_size).astype(np.float32)
    
    # Create PyTorch tensor from numpy
    pytorch_input = torch.tensor(np_input)
    # Create tinygrad tensor from numpy
    tinygrad_input = Tensor(np_input)
    
    # Run inference on both models
    print("Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = pytorch_model(pytorch_input).numpy()
    
    print("Running tinygrad inference...")
    tinygrad_output = tinygrad_model.forward(tinygrad_input).numpy()
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"Tinygrad output shape: {tinygrad_output.shape}")
    
    # Compare outputs
    output_diff = np.abs(pytorch_output - tinygrad_output)
    max_diff = np.max(output_diff)
    mean_diff = np.mean(output_diff)
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check if outputs are close enough
    if max_diff < 1e-4:
        print("✅ Test passed! Outputs are nearly identical.")
    else:
        print("❌ Test failed! Outputs differ significantly.")
    
    return tinygrad_model, pytorch_model

if __name__ == "__main__":
    main()