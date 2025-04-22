
import gradio as gr
from PIL import Image
from models.image_generator import ImageGenerator
from models.vision_encoder import CLIPVisionEncoder
from models.projector import ProjectorMLP
from models.llm_vistral import load_llm
import torch

vision_encoder = CLIPVisionEncoder("openai/clip-vit-base-patch32")
projector = ProjectorMLP(input_dim=1024, output_dim=4096)
llm = load_llm("checkpoints/vistral-mm")
image_generator = ImageGenerator("configs/diffusion_config.yaml")

def analyze_image(image):
    features = vision_encoder(image)
    projected = projector(features.unsqueeze(0))
    prompt = "Ảnh này có đặc điểm gì?"
    input_embeds = projected
    response = llm.generate_from_image_features(input_embeds, prompt)
    return response

def generate_from_prompt(prompt):
    image = image_generator.generate(prompt)
    return image

with gr.Blocks() as demo:
    gr.Markdown("# 📈 Vistral-Multimodal: Vietnamese AI Assistant")

    with gr.Tab("Image → Text"):
        image_input = gr.Image(type="pil")
        caption_output = gr.Textbox()
        btn1 = gr.Button("Phân tích ảnh")
        btn1.click(analyze_image, inputs=image_input, outputs=caption_output)

    with gr.Tab("Text → Image"):
        prompt_input = gr.Textbox(label="Nhập mô tả y tế")
        image_output = gr.Image()
        btn2 = gr.Button("Sinh ảnh")
        btn2.click(generate_from_prompt, inputs=prompt_input, outputs=image_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
