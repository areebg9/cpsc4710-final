from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

"""
def vis_image(pixels):
    pixels_np = pixels.detach().cpu().numpy()
    pixels_np = pixels_np.transpose(1, 2, 0)
    
    plt.imshow(pixels_np)
    plt.axis('off')
    plt.show()
"""

def tensor_to_pil(tensor, processor):
    mean = processor.image_mean
    std = processor.image_std
    
    img = tensor.detach().cpu()
    
    # Denormalize
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    
    # Convert to [0, 255] range and PIL
    img = (img * 255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).numpy()
    
    return Image.fromarray(img)

def vis_image_denormalized(inputs, processor):
    pixels = inputs['pixel_values'][0]
    pixels = pixels.cpu()
    mean = torch.tensor(processor.image_mean).view(3, 1, 1).to('cpu')
    std = torch.tensor(processor.image_std).view(3, 1, 1).to('cpu')
    
    # Denormalize: pixel = (normalized * std) + mean
    pixels_denorm = pixels * std + mean

    pixels_np = pixels_denorm.detach().cpu().numpy()
    pixels_np = pixels_np.transpose(1, 2, 0)
    
    pixels_np = pixels_np.clip(0, 1)
    
    plt.imshow(pixels_np)
    plt.axis('off')
    plt.show()