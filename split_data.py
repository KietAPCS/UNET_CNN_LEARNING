import os
import shutil
from sklearn.model_selection import train_test_split

train_images_dir = 'data\\train_images'
train_masks_dir = 'data\\train_masks'
val_images_dir = 'data\\val_images'
val_masks_dir = 'data\\val_masks'

os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

image_files = sorted(os.listdir(train_images_dir))
mask_files = sorted(os.listdir(train_masks_dir))

# Ensure the lists match (same filenames for images and masks)
print(f"Image files'length: {len(image_files)}")
print(f"Mask files'length: {len(mask_files)}")
assert len(image_files) == len(mask_files)

for img_file, mask_file in zip(image_files, mask_files):
    file_name, file_ext = os.path.splitext(img_file)
    file_mask_name = f"{file_name}_mask.gif"
    assert file_mask_name == mask_file  
    
train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)

for img, mask in zip(val_imgs, val_masks):
    shutil.move(os.path.join(train_images_dir, img), os.path.join(val_images_dir, img))
    shutil.move(os.path.join(train_masks_dir, mask), os.path.join(val_masks_dir, mask))

print("Validation data split and moved successfully.")
