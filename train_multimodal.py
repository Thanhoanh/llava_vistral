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

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, vision_encoder, projector):
        self.data = [json.loads(line) for line in open(data_path)]
        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder
        self.projector = projector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image"]
        instruction = sample["instruction"]
        response = sample["response"]

        # Load image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        # Encode text
        input_ids = self.tokenizer(instruction, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0]
        labels = self.tokenizer(response, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0]

        # Encode image
        image_feat = self.vision_encoder(image)
        if image_feat.dim() == 1:
            image_feat = image_feat.unsqueeze(0)
        
        image_embed = self.projector(image_feat)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image_embed": image_embed
        }

if __name__ == "__main__":
    print("🚀 Đang khởi tạo...")

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
    dataloader = DataLoader(dataset, batch_size=cfg["per_device_train_batch_size"], shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    print("✅ Bắt đầu huấn luyện...")
    for epoch in range(cfg["num_train_epochs"]):
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("📦 Đang lưu mô hình vào:", cfg["output_dir"])
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("✅ Đã lưu mô hình.")
