import cv2
from PIL import Image
from torchvision import transforms
import numpy as np

def load_and_resize_image(image_path, new_size):
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Define the resize transformation
    resize_transform = transforms.Resize(new_size)
    
    # Apply the transformation to the image
    resized_image = resize_transform(image)
    
    # Convert the PIL image to an OpenCV-compatible format (numpy array)
    # Note: PIL uses RGB by default and OpenCV uses BGR
    resized_image_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    
    return resized_image_cv

def display_image(cv_image):
    # Display the image using OpenCV
    cv2.imwrite('input_videos/image_resized.png', cv_image)
    cv2.imshow('Resized Image', cv_image)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()


# Path to your image file
image_path = 'input_videos/image.png'  # Update this path to your image file
new_size = (384, 640)  # Set the desired new size (width, height)

# Load and resize the image
resized_image = load_and_resize_image(image_path, new_size)

# Display the resized image
display_image(resized_image)
