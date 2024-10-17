import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
from PIL import Image
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import cv2
import torchvision
from model import UNET 
from torchvision.utils import save_image
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_IMG_DIR = 'data\\train_images'
TRAIN_MASK_DIR = 'data\\train_masks'
VAL_IMG_DIR = 'data\\val_images'
VAL_MASK_DIR = 'data\\val_masks'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        #forward
        with torch.amp.autocast('cuda'):
            predictions =  model(data)
            loss = loss_fn(predictions, targets)
            
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  # Adjust based on your dataset
                std=[1.0, 1.0, 1.0],   # Adjust based on your dataset
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2()
        ],
    )
    
    model = UNET(input_channels=3, output_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        
        check_accuracy(val_loader, model, device=DEVICE)
        
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

def test_single_image(model, image_path, transform, device, save_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    
    augmented = transform(image=image)
    image = augmented["image"].unsqueeze(0)  # Add batch dimension
    
    image = image.to(device)
    
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(model(image))
        prediction = (prediction > 0.5).float()  # Apply thresholding
    
    # Remove batch dimension and save the prediction
    torchvision.utils.save_image(prediction, save_path)
    print(f"Prediction saved at {save_path}")
    
def testing(test_image_path, save_path):
    model = UNET(input_channels=3, output_channels=1).to(DEVICE)  # Initialize model
    
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  # Adjust based on your dataset
                std=[1.0, 1.0, 1.0],   # Adjust based on your dataset
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2()
        ],
    )
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_accuracy(val_loader, model, device=DEVICE)

    # Testing on a single image

    test_single_image(model, test_image_path, train_transform, DEVICE, save_path)
           
if __name__ == "__main__":
    test_image_path = "data\\train_images\\0cdf5b5d0ce1_01.jpg"
    save_path = "saved_predictions/test_prediction.png"
    
    testing(test_image_path, save_path)