import numpy as np
from io import BytesIO
from PIL import Image

TARGET_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess_image(image_bytes):
    """
    Standard preprocessing for ResNet/EfficientNet models.
    Converts image bytes to a normalized tensor-like numpy array.
    """
    # Load image from bytes
    img = Image.open(BytesIO(image_bytes))
    img = img.convert('RGB')
    
    # Resize
    img = img.resize(TARGET_SIZE)
    
    # Convert to numpy and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet normalization
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
