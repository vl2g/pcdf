import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class InternVL3:
    def __init__(self, model_id="OpenGVLab/InternVL3-2B-hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, system_instruction, user_instruction, image_path, max_tokens=64):
        print()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"{system_instruction} {user_instruction}"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, torch.bfloat16)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=151645,
            cache_implementation="static"
        )

        generated_text = self.processor.decode(
            output_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return generated_text