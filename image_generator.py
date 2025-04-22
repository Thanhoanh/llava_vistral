
from diffusers import StableDiffusionPipeline
import torch

class ImageGenerator:
    def __init__(self, config_path):
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt):
        image = self.pipe(prompt).images[0]
        return image
