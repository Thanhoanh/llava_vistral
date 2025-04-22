
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch

class CLIPVisionEncoder:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0]
