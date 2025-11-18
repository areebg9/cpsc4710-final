import transformers
from PIL import Image
import torchvision
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def generate_loss(model, image, label, processor):
    inputs = processor(images=image, return_tensors="pt").to(device)
    inputs['pixel_values'].requires_grad_()
    inputs['labels'] = torch.tensor([label]).to(device)
    
    outputs = model(**inputs)
    return outputs.loss, inputs

def fgsm_attack(inputs, loss, epsilon):
    loss.backward()
    pert = epsilon * inputs['pixel_values'].grad.sign()
    
    perturbed_input = {key: value.clone() for key, value in inputs.items()}
    perturbed_input['pixel_values'] = inputs['pixel_values'] + pert
    
    return perturbed_input

def get_classification(model, inputs):
    outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx], predicted_class_idx