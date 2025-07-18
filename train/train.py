from rfdetr import RFDETRBase
import torch
import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["LOCAL_RANK"] = "0"

# Set dataset path (COCO format)
dataset_path = "/vol/bitbucket/si324/rf-detr-wildfire/pyro25img/images"

# Set output directory 
output_dir = "./output/img_data_train_output"

# Initialize model
model = RFDETRBase()

# Train the model
model.train(
    dataset_dir=dataset_path,
    epochs=50,
    batch_size=4,           
    grad_accum_steps=4,     
    lr=1e-4,
    device="cuda",      
    ddp=False,
    output_dir=output_dir
)
