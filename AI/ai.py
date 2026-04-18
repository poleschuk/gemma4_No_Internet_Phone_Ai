import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class AI:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        dtype=torch.bfloat16,
                        device_map=self.device
                    )
    
    def generate(self, messages):
        processor = AutoProcessor.from_pretrained(self.model_path)
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        processor.parse_response(response)
        
        return response
        
        