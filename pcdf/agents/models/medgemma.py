import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class MedGemma:
    def __init__(self, model_id="google/medgemma-4b-it"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, system_instruction, user_instruction, image_path):

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_instruction}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_instruction},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=64,
            cache_implementation="static"
        )

        decoded = self.processor.decode(output[0], skip_special_tokens=True)

        if "model" in decoded:
            decoded = decoded.split("model", 1)[1]

        decoded = decoded.replace("\n", "").strip()

        return decoded
