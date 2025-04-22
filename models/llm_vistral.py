
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return VistralLLM(model, tokenizer)

class VistralLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_from_image_features(self, image_embedding, prompt=""):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
