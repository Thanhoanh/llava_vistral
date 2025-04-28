import os
import torch
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from models.vision_encoder import CLIPVisionEncoder
from models.projector import ProjectorMLP
from models.llm_vistral import load_llm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import json

from torchvision import transforms
from PIL import Image

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, vision_encoder, projector):
        self.data = [json.loads(line) for line in open(data_path)]
        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image"]
        instruction = sample["instruction"]
        response = sample["response"]

        # Load và transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # Bây giờ image là Tensor rồi

        input_ids = self.tokenizer(instruction, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids.squeeze(0)
        labels = self.tokenizer(response, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids.squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image_tensor": image
        }

if __name__ == "__main__":
    print("🚀 Khởi tạo...")

    with open("configs/vistral_config.json") as f:
        cfg = json.load(f)
    with open("configs/lora_config.json") as f:
        lora_cfg = json.load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    model_wrapper = load_llm(cfg["model_name_or_path"])
    model = get_peft_model(model_wrapper.model, LoraConfig(**lora_cfg))
    tokenizer = model_wrapper.tokenizer

    vision_encoder = CLIPVisionEncoder("openai/clip-vit-base-patch32")
    projector = ProjectorMLP(input_dim=512, output_dim=model.config.hidden_size)

    dataset = MultimodalDataset("dataset/prompt/vi_multimodal.jsonl", tokenizer, vision_encoder, projector)
    dataloader = DataLoader(dataset, batch_size=cfg["per_device_train_batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.cuda.amp.GradScaler('cuda')  # mixed precision

    print("✅ Bắt đầu huấn luyện...")
    for epoch in range(cfg["num_train_epochs"]):
        pbar = tqdm(dataloader)
        for batch in pbar:
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            images = batch["image_tensor"].to(model.device)

            # Encode ảnh
            image_feat = vision_encoder(images)
            image_embed = projector(image_feat)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": loss.item()})

    print("📦 Đang lưu mô hình vào:", cfg["output_dir"])
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("✅ Đã lưu mô hình.")
