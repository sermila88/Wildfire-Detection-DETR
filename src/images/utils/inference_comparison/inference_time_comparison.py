# Inference time compare

import glob
import os
import numpy as np
import json
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import supervision as sv
import time
import psutil
import torch
import gc
import platform, psutil
import sys
import subprocess

# Add path for eval functions
sys.path.append('/vol/bitbucket/si324/rf-detr-wildfire/src/images')

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPUs

# Configuration
OUTPUT_BASE_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/inference_time_comparison"
TEST_IMAGES_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img/images/test"

def load_models(model_type, model_path):
    """Load model once - FORCED TO CPU"""
    if model_type == "YOLO":
        from ultralytics import YOLO
        print(f"Loading YOLO from: {model_path}")
        model = YOLO(model_path)
        model.to('cpu')  # Force CPU
        return model
    elif model_type == "RF-DETR":
        from rfdetr import RFDETRBase
        print(f"Loading RF-DETR from: {model_path}")
        model = RFDETRBase(pretrain_weights=model_path, num_classes=1)
        if hasattr(model, 'to'):
            model.to('cpu')
        return model
    elif model_type == "RT-DETR":
        from ultralytics import RTDETR
        print(f"Loading RT-DETR from: {model_path}")
        model = RTDETR(model_path)
        model.to('cpu')  # Force CPU
        return model
    return None

def generate_rfdetr_predictions(model, image_path, conf_threshold):
    """Generate RF-DETR predictions - exact copy from eval script"""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        predictions = model.predict(img_rgb, threshold=conf_threshold)
    return predictions

def generate_rtdetr_predictions(model, image_path, conf_threshold):
    """Generate RT-DETR predictions - exact copy from eval script"""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        results = model.predict(source=img_rgb, conf=conf_threshold, verbose=False, device='cpu')
        predictions = sv.Detections.from_ultralytics(results[0])
    return predictions

def run_single_inference(model, model_type, image_path):
    """Run a single inference - using exact same logic as eval script"""
    if model_type == "YOLO":
        _ = model.predict(image_path, conf=0.01, iou=0.01, verbose=False, device='cpu')
    elif model_type == "RF-DETR":
        _ = generate_rfdetr_predictions(model, image_path, 0.01)
    elif model_type == "RT-DETR":
        _ = generate_rtdetr_predictions(model, image_path, 0.01)

def benchmark_model(model, model_type, test_images, num_warmup=10):
    """Latency and FPS"""
    
    print(f"  Running on CPU with {torch.get_num_threads()} threads")
    
    # Warmup runs
    print(f"  Warming up with {num_warmup} images...")
    for i in range(min(num_warmup, len(test_images))):
        run_single_inference(model, model_type, test_images[i])
    
    # Actual benchmark
    print(f"  Benchmarking all {len(test_images)} test images...")
    latencies = []
    
    for image_path in tqdm(test_images, desc="  Progress"):
        start_time = time.perf_counter()
        run_single_inference(model, model_type, image_path)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    mean_latency = np.mean(latencies)
    fps = 1000 / mean_latency

    std_latency = np.std(latencies)
    return {
        'mean_latency_ms': float(mean_latency),
        'std_latency_ms': float(std_latency),  
        'throughput_fps': float(fps)
    }

if __name__ == "__main__":
    # Model configurations
    models = {
        "YOLO_baseline": {
            "type": "YOLO",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/YOLO_baseline/training_outputs/eager-flower-1/weights/best.pt"
        },
        "RF-DETR": {
            "type": "RF-DETR",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RF-DETR_initial_training/checkpoints/checkpoint_best_ema.pth"
        },
        "RT-DETR": {
            "type": "RT-DETR",
            "path": "/vol/bitbucket/si324/rf-detr-wildfire/src/images/outputs/RT-DETR_hyperparameter_tuning/trial_009/checkpoints/weights/best.pt"
        }
    }

    # Add system info
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    #CPU info
    try:
        cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
        cpu_model = cpu_info.split(':')[1].strip() if ':' in cpu_info else "Unknown CPU"
    except:
        cpu_model = platform.processor()
    
    print(f"CPU: {cpu_model}")
    
    # Get test images
    test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")))
    if not test_images:
        test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    
    print(f"\nFound {len(test_images)} test images")
    print(f"Running on CPU (GPUs disabled)")
    print("="*60)
    
    # Benchmark each model
    all_results = {}
    
    for model_name, config in models.items():
        print(f"\nBenchmarking {model_name}")
        print("-"*40)
        
        # Load model
        model = load_models(config['type'], config['path'])
        
        if model is None:
            print(f"  Failed to load {model_name}")
            continue
        
        # Quick sanity check
        try:
            test_img = test_images[0] if test_images else None
            if test_img:
                run_single_inference(model, config['type'], test_img)
                print(f"  ✓ Model loaded and inference test passed")
        except Exception as e:
            print(f"  ⚠ Warning: Test inference failed: {e}")
        
        # Run benchmark
        results = benchmark_model(
            model, 
            config['type'],
            test_images,
            num_warmup=10
        )
        
        all_results[model_name] = results
        
        # Print results 
        print(f"\n  Results for {model_name}:")
        print(f"    Latency: {results['mean_latency_ms']:.1f} ms")
        print(f"    FPS:     {results['throughput_fps']:.1f}")
        
        # Clear model from memory
        del model
        gc.collect()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_BASE_DIR, f"benchmark_results_cpu_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'device': 'CPU',
            'num_threads': torch.get_num_threads(),
            'system_info': {
                'cpu': platform.processor(),
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_threads': psutil.cpu_count(logical=True)
            },
            'results': all_results
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    # Print comparison table
    if all_results:
        print(f"\n{'Model':<20} {'Latency (ms)':<15} {'FPS':<10}")
        print("-"*45)
        for model_name, results in all_results.items():
            print(f"{model_name:<20} {results['mean_latency_ms']:<15.1f} {results['throughput_fps']:<10.1f}")
    print(f"\nResults saved to: {output_file}")

    # Save the table to a text file
    output_dir = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/Comparative_eval/CPU"
    os.makedirs(output_dir, exist_ok=True)
    
    table_file = os.path.join(output_dir, "inference_time_comparison_CPU.txt")
    with open(table_file, 'w') as f:
        f.write("Inference Time Comparison - CPU\n")
        f.write("="*50 + "\n")
        f.write(f"Hardware: {cpu_model}\n")
        f.write(f"Physical cores: {psutil.cpu_count(logical=False)}\n")
        f.write(f"Threads used: {torch.get_num_threads()}\n")
        f.write(f"Test images: {len(test_images)}\n")
        f.write("="*50 + "\n\n")
        f.write(f"{'Model':<20} {'Latency (ms)':<15} {'FPS':<10}\n")
        f.write("-"*45 + "\n")
        for model_name, results in all_results.items():
            f.write(f"{model_name:<20} {results['mean_latency_ms']:<15.1f} {results['throughput_fps']:<10.1f}\n")

    print(f"Table saved to: {table_file}")

    
