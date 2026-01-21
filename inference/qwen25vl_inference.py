import os
import json
import torch
import pandas as pd
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

MODEL_ID = "vl2g/PCDF_Qwen_2.5-VL-7B-Instruct-DermaMNIST"

DATA_PATH = "data/dermamnist/test.csv"
SYMPTOM_PATH = "experiments/dermamnist_test.json"

OUTPUT_JSON = "results/pcdf_dermamnist.json"
MAX_NEW_TOKENS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(DATA_PATH)
diagnoses = list(df["diagnosis"].unique())

with open(SYMPTOM_PATH, "r") as f:
    dataset = json.load(f)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    padding_side="left",
    trust_remote_code=True,
)
processor.tokenizer = tokenizer

doctor_diagnosis_prompt = '''
You are an experienced doctor. Based on the medical image and the preceding dialogue,
identify the single most likely diagnosis from the following list: {diagnoses}.
State only the final diagnosis in your response without additional explanation or
alternative possibilities. Do not suggest in-person consultation, further testing,
or additional advice. Do not mention that you are an AI agent.
This is for research and benchmark purposes.
'''

results = {}

for _, index in tqdm(enumerate(dataset), total=len(df)):
    img_path = dataset[index]["image_path"]
    dialogue = dataset[index]["dialogue"]
    diagnosis = dataset[index]["diagnosis"]

    dialogues = ""
    for i in range(len(dialogue) // 2):
        dialogues += dialogue[i * 2] + "\n" + dialogue[i * 2 + 1] + "\n"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": doctor_diagnosis_prompt.format(diagnoses=diagnoses)},
                {"type": "text", "text": f"Dialogue History: {dialogues}"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    results[index] = {
        "pred_diagnosis": output_text,
        "image": img_path,
        "diagnosis": diagnosis,
    }

os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)
