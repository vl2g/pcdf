import torch
from PIL import Image
from transformers import AutoTokenizer, AutoConfig, AutoModel

class mPLUGOwl3():
    def __init__(self, model_id = 'mPLUG/mPLUG-Owl3-7B-240728'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(self.config, 'vision_config') and self.config.vision_config is not None:
            self.config.vision_config._attn_implementation = "sdpa"
        if hasattr(self.config, 'text_config') and self.config.text_config is not None:
            self.config.text_config._attn_implementation = "sdpa"
        self.config._attn_implementation = "sdpa"

        self.model = AutoModel.from_pretrained(
            model_id,
            attn_implementation='sdpa',
            torch_dtype=torch.half,
            trust_remote_code=True
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = self.model.init_processor(self.tokenizer)

    def run(self, system_instruction, user_instruction, image_path):

        image = Image.open(image_path).convert("RGB")
        if image.size[0] > 400:
            scale = 400 / image.size[0]
            image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
        
        messages = [
            {"role": "user", "content": f"<|image|>\n{system_instruction + ' ' + user_instruction}"},
            {"role": "assistant", "content": ""}
        ]

        inputs = self.processor(messages, images=[image], videos=None)
        inputs.to(self.device)
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': 64,
            'decode_text': True,
        })

        with torch.no_grad():
            output = self.model.generate(**inputs)

        return output[0]