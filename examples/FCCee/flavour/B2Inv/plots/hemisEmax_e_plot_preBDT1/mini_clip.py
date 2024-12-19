import cv2
import os
import sys
import numpy as np



def create_video_from_images(image_folder, output_video, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # reverse=True would ensure the images are in the correct order

    # Read the first image to get the frame size
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' for .avi format
    out = cv2.VideoWriter(output_video, fourcc, fps, size)

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)  # Write out frame to video

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {output_video}")

# Usage
image_folder = '/r01/lhcb/ejnw2/fcc/FCCAnalyses/examples/FCCee/flavour/B2Inv/hemisEmax_e_plot_preBDT1/pngs'
output_video = 'output_video.mp4'
create_video_from_images(image_folder, output_video)
