import gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.vision_encoder import CLIPVisionEncoder
from models.projector import ProjectorMLP
from models.image_generator import ImageGenerator
import torch

# Load mô hình ngôn ngữ
llm = AutoModelForCausalLM.from_pretrained(
    "vistral-mm",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).eval()
tokenizer = AutoTokenizer.from_pretrained("vistral-mm")

# Load mô hình thị giác
vision_encoder = CLIPVisionEncoder("openai/clip-vit-base-patch32")
projector = ProjectorMLP(input_dim=512, output_dim=llm.config.hidden_size)

# Load mô hình sinh ảnh
image_generator = ImageGenerator("configs/diffusion_config.yaml")

def analyze_image_and_question(image, question):
    features = vision_encoder(image)
    if features.dim() == 1:
        features = features.unsqueeze(0)
    image_embed = projector(features)

    # Sinh câu trả lời từ câu hỏi
    prompt = question
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_image(prompt):
    image = image_generator.generate(prompt)
    return image

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Vistral-Multimodal: Trợ lý Y tế Tiếng Việt")

    with gr.Tab("🖼️ Ảnh + Văn bản → Trả lời"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Ảnh y tế")
            text_input = gr.Textbox(label="Câu hỏi về ảnh", placeholder="VD: Ảnh này biểu thị điều gì?")
        output = gr.Textbox(label="📋 Trả lời từ mô hình")
        btn = gr.Button("Phân tích")
        btn.click(analyze_image_and_question, inputs=[image_input, text_input], outputs=output)

    with gr.Tab("📝 Văn bản → 🖼️ Ảnh"):
        prompt_input = gr.Textbox(label="Nhập mô tả triệu chứng", placeholder="VD: Ban đỏ hình cánh bướm trên mặt")
        image_output = gr.Image(label="Ảnh được sinh")
        btn2 = gr.Button("Sinh ảnh")
        btn2.click(generate_image, inputs=prompt_input, outputs=image_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
