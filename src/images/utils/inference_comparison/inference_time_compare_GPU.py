# Inference time compare on the GPU

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

# Add path for eval functions
sys.path.append('/vol/bitbucket/si324/rf-detr-wildfire/src/images')


# Configuration
OUTPUT_BASE_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/inference_time_comparison"
TEST_IMAGES_DIR = "/vol/bitbucket/si324/rf-detr-wildfire/src/images/data/pyro25img/images/test"

def load_models(model_type, model_path):
    """Load model once - FOR GPU"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    if model_type == "YOLO":
        from ultralytics import YOLO
        print(f"Loading YOLO from: {model_path}")
        model = YOLO(model_path)
        # YOLO auto-detects GPU
        return model
    elif model_type == "RF-DETR":
        from rfdetr import RFDETRBase
        print(f"Loading RF-DETR from: {model_path}")
        model = RFDETRBase(pretrain_weights=model_path, num_classes=1)
        if hasattr(model, 'to'):
            model.to(device)
        return model
    elif model_type == "RT-DETR":
        from ultralytics import RTDETR
        print(f"Loading RT-DETR from: {model_path}")
        model = RTDETR(model_path)
        # RT-DETR auto-detects GPU
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
        results = model.predict(source=img_rgb, conf=conf_threshold, verbose=False) 
        predictions = sv.Detections.from_ultralytics(results[0])
    return predictions

def run_single_inference(model, model_type, image_path):
    """Run a single inference """
    if model_type == "YOLO":
        _ = model.predict(image_path, conf=0.01, iou=0.01, verbose=False)
    elif model_type == "RF-DETR":
        _ = generate_rfdetr_predictions(model, image_path, 0.01)
    elif model_type == "RT-DETR":
        _ = generate_rtdetr_predictions(model, image_path, 0.01)

def benchmark_model(model, model_type, test_images, num_warmup=5, num_test=100):
    """Benchmark a model's inference performance"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Running on {device.upper()}")
    
    # Get process for memory monitoring
    process = psutil.Process()
    
    # Warmup runs
    print(f"  Warming up with {num_warmup} images...")
    for i in range(min(num_warmup, len(test_images))):
        run_single_inference(model, model_type, test_images[i])
    
    # Actual benchmark
    print(f"  Benchmarking {num_test} images...")
    latencies = []
    memory_usage = []
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Progress bar
    for i in tqdm(range(min(num_test, len(test_images))), desc="  Progress"):
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024

        # Add GPU synchronization BEFORE timing starts
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time the inference
        start_time = time.perf_counter()
        run_single_inference(model, model_type, test_images[i])

        # Add GPU synchronization AFTER inference, BEFORE timing ends
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        
        # Record metrics
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_usage.append(mem_after - initial_memory)
    
    # Calculate statistics
    latencies = np.array(latencies)
    memory_usage = np.array(memory_usage)
    
    results = {
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'throughput_fps': float(1000 / np.mean(latencies)),
        'total_time_seconds': float(np.sum(latencies) / 1000),
        'peak_memory_mb': float(np.max(memory_usage)) if len(memory_usage) > 0 else 0,
        'avg_memory_mb': float(np.mean(memory_usage)) if len(memory_usage) > 0 else 0,
        'num_images_tested': min(num_test, len(test_images))
    }
    
    return results

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
    print(f"CPU: {platform.processor()}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical Threads: {psutil.cpu_count(logical=True)}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*60)
    
    # Get test images
    test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")))
    if not test_images:
        test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    
    print(f"\nFound {len(test_images)} test images")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device.upper()}")
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
            num_warmup=5,
            num_test=536  # All test images
        )
        
        all_results[model_name] = results
        
        # Print results 
        print(f"\n  Results for {model_name}:")
        print(f"    Mean latency:     {results['mean_latency_ms']:.2f} ms")
        print(f"    Std latency:      {results['std_latency_ms']:.2f} ms")
        print(f"    Median latency:   {results['median_latency_ms']:.2f} ms")
        print(f"    P95 latency:      {results['p95_latency_ms']:.2f} ms")
        print(f"    P99 latency:      {results['p99_latency_ms']:.2f} ms")
        print(f"    Throughput:       {results['throughput_fps']:.2f} FPS")
        print(f"    Total time:       {results['total_time_seconds']:.2f} seconds")
        print(f"    Peak memory:      {results['peak_memory_mb']:.2f} MB")
        
        # Clear model from memory
        del model
        gc.collect()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_BASE_DIR, f"benchmark_results_gpu_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'device': 'GPU',
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
        print(f"\n{'Model':<20} {'Mean (ms)':<12} {'FPS':<10} {'Memory (MB)':<12}")
        print("-"*54)
        for model_name, results in all_results.items():
            print(f"{model_name:<20} {results['mean_latency_ms']:<12.2f} {results['throughput_fps']:<10.2f} {results['peak_memory_mb']:<12.2f}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Performance ranking
    if len(all_results) > 1:
        print("\nPerformance Ranking (fastest to slowest):")
        # Sort by latency
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['mean_latency_ms'])
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            print(f"  {rank}. {model_name}: {results['mean_latency_ms']:.2f} ms ({results['throughput_fps']:.2f} FPS)")