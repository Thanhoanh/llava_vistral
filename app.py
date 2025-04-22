import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load mô hình và tokenizer từ thư mục đã huấn luyện
model = AutoModelForCausalLM.from_pretrained(
    "vistral-mm",  # thư mục chứa model
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("vistral-mm")

# Hàm xử lý câu hỏi
def chat_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Trợ lý tiếng Việt Vistral-MM (Text → Text)")

    prompt_input = gr.Textbox(label="📝 Nhập câu hỏi hoặc mô tả", placeholder="Ví dụ: Triệu chứng của sốt xuất huyết?")
    output_text = gr.Textbox(label="📋 Câu trả lời từ mô hình")

    btn = gr.Button("💬 Gửi")
    btn.click(chat_with_model, inputs=prompt_input, outputs=output_text)

# Khởi chạy server
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
