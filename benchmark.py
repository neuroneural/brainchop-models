import gc
import torch
import time
import json
import argparse
import csv
import os
from meshnet_gn import enMesh_checkpoint
from synthstrip import StripModel
from default_meshnet import enMesh_checkpoint as DefaultMeshnet

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Neural network model benchmarking')
    parser.add_argument('--mindgrab', action='store_true', help='Use MindGrab model')
    parser.add_argument('--synthstrip', action='store_true', help='Use SynthStrip model')
    parser.add_argument('--meshnet', action='store_true', help='Use original MeshNet model with meshnet_config.json')
    parser.add_argument('--default-meshnet', action='store_true', help='Use Default MeshNet model with default_meshnet_config.json')
    parser.add_argument('--short-meshnet', action='store_true', help='Use Short MeshNet model with modelAEinv2.json')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--size', type=int, default=256, help='Input tensor size (cubic)')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='CSV output file')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16'], 
                        help='Data type for model inference (fp32, fp16)')
    # New arguments
    parser.add_argument('--autocast', action='store_true', help='Use torch.cuda.amp.autocast for mixed precision')
    parser.add_argument('--half-model', action='store_true', help='Convert model weights & buffers to FP16')
    
    args = parser.parse_args()
    
    # Check if any model is selected
    if not (args.mindgrab or args.synthstrip or args.meshnet or args.default_meshnet or args.short_meshnet):
        parser.error("At least one model must be specified using --mindgrab, --synthstrip, --meshnet, --default-meshnet, or --short-meshnet")
    
    # Simple device info
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"Device: {device_name}, Count: {device_count}")
    
    # Model selection and name
    model_name = ""
    if args.mindgrab:
        model_name = "MindGrab"
        print(f"Using {model_name} model")
        model = enMesh_checkpoint(1, 2, 15, "models/meshnet_config.json")
        # Load weights if needed
        # model.load_state_dict(torch.load("models/mindgrab.pth"))
    elif args.synthstrip:
        model_name = "SynthStrip"
        print(f"Using {model_name} model")
        model = StripModel()
        # Load weights if needed
        # model.load_state_dict(torch.load("models/synthstrip.pth"))
    elif args.meshnet:
        model_name = "MeshNet"
        print(f"Using {model_name} model")
        model = DefaultMeshnet(1, 2, 15, "models/meshnet_config.json")
    elif args.default_meshnet:
        model_name = "DefaultMeshNet"
        print(f"Using {model_name} model")
        model = DefaultMeshnet(1, 2, 5, "models/default_meshnet_config.json")
    elif args.short_meshnet:
        model_name = "ShortMeshNet"
        print(f"Using {model_name} model")
        model = DefaultMeshnet(1, 2, 15, "models/modelAEinv2.json")
    else:
        print("Error: No model selected")
        return
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Set data type based on argument
    dtype = None
    precision_mode = args.dtype
    if args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'fp16':
        dtype = torch.float16
    
    # Handle half-model if specified (or fp16 mode)
    if args.half_model or args.dtype == 'fp16':
        # First use model.half() to convert most parameters
        model = model.half()
        
        # Manually convert any remaining bias parameters to half precision
        for module in model.modules():
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = module.bias.data.half()
        
        precision_mode = "half_model" if args.half_model else "fp16"
        print(f"Using {'model.half()' if args.half_model else 'fp16'} precision with manual bias conversion")
    
    # Set parameters to not require gradients
    layers = [p for p in model.parameters()]
    for p in layers:
        p.grad = None
        p.requires_grad = False
    
    # Create input tensor based on specified size and dtype
    input_tensor = torch.randn(1, 1, args.size, args.size, args.size, device=device, dtype=dtype)
    
    # Handle autocast mode
    if args.autocast:
        precision_mode = "autocast"
        print("Using autocast mixed precision")
    else:
        print(f"Using {precision_mode} precision")
    
    # Memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated()
    
    # Benchmark setup
    num_runs = args.runs
    times = []
    
    # Define inference function with option to use autocast
    @torch.compile
    def loop(input_tensor):
        torch.cuda.synchronize()
        if args.autocast:
            with torch.cuda.amp.autocast():
                output = model(input_tensor)
        else:
            output = model(input_tensor)
        torch.cuda.synchronize()
        return output
    
    # Warm-up run
    print("Running warm-up pass...")
    with torch.no_grad():
        output = loop(input_tensor)
        del output
        torch.cuda.empty_cache()
    
    # Benchmark loop
    print(f"Starting benchmark with {num_runs} runs...")
    with torch.no_grad():
        for i in range(num_runs):
            # Clear cache before each run
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()
            output = loop(input_tensor)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Run {i+1}: {times[-1] * 1000:.2f} ms")
            del output
    
    # Calculate metrics
    average_time = sum(times) / len(times) * 1000
    peak_mem = torch.cuda.max_memory_allocated() - start_mem
    peak_mem_mb = peak_mem / 1024 / 1024
    
    print(f"\nResults:")
    print(f"Model: {model_name}")
    print(f"Device: {device_name}")
    print(f"Precision: {precision_mode}")
    print(f"Average inference time: {average_time:.2f} ms")
    print(f"Peak memory usage: {peak_mem_mb:.2f} MB")
    
    # Write results to CSV
    file_exists = os.path.isfile(args.output)
    
    # Prepare CSV row data
    row_data = {
        'model': model_name,
        'device_type': device_type,
        'device_name': device_name,
        'device_count': device_count,
        'precision': precision_mode,
        'avg_time_ms': f"{average_time:.2f}",
        'peak_memory_mb': f"{peak_mem_mb:.2f}",
        'tensor_size': args.size
    }
    
    # Add individual run times
    for i, t in enumerate(times):
        row_data[f'run_{i+1}_ms'] = f"{t * 1000:.2f}"
    
    # Write to CSV
    with open(args.output, 'a', newline='') as csvfile:
        # Define columns
        fieldnames = ['model', 'device_type', 'device_name', 'device_count', 'precision', 'tensor_size']
        for i in range(1, num_runs + 1):
            fieldnames.append(f'run_{i}_ms')
        fieldnames.extend(['avg_time_ms', 'peak_memory_mb'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if new file
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
