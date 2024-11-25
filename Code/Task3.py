import cv2
import numpy as np
import os

# Stitch together using cv2 Built-In function
def stich_images(images):
    # Start CV2 stitching
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images)
    
    # Save panorama
    cv2.imwrite(output, panorama)
    
    # Success/Failure message
    if status == cv2.Stitcher_OK:
        print(f"Panorama created: {output}")
    else:
        print(f"Failed to create image: {status}")

def main():
    # Get file names
    image_files = [f for f in os.listdir(folder)]
    images = []

    # Read each file
    for file in image_files:
        img = cv2.imread(f"{folder}/{file}")
        
        # Error Handlding
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {file}")
    
    # Start Stitching
    stich_images(images)

if __name__ == "__main__":
    # Settings
    folder = "Panorama"
    output = "Panorama/Panorama.jpg"

    main()