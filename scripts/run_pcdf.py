import os
import json
import argparse
import importlib
from tqdm import tqdm

import torch
import pandas as pd

from pcdf.agents.patient_vlm import PatientVLM
from pcdf.agents.doctor_vlm import DocVLM
from pcdf.utils import TURNS, get_doctor_prompt, get_patient_prompt
from pcdf.post_process import clean_patient_response


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run patient-doctor dialogue generation"
    )

    parser.add_argument(
        "--speciality",
        type=str,
        default="Dermatology",
        help="Medical speciality (default: Dermatology)"
    )

    parser.add_argument(
        "--doc_vlm",
        type=str,
        default="Qwen25VL",
        help="Doctor VLM model name (default: Qwen25VL)"
    )

    parser.add_argument(
        "--patient_vlm",
        type=str,
        default="mPLUGOwl3",
        help="Patient VLM model name (default: mPLUGOwl3)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="dermamnist",
        help="Config module under config/ (default: dermamnist)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)"
    )
    return parser.parse_args()


def run_dialogue(
    pat_vlm,
    doc_vlm,
    speciality,
    image_path,
    diagnoses,
    diagnosis,
    config
):
    dialogue = []

    for _ in range(TURNS):
        # Doctor turn
        doctor_role, doctor_prompt = get_doctor_prompt(speciality)
        doc_question = doc_vlm.run(
            doctor_role,
            doctor_prompt.format(dialogue, diagnoses),
            image_path
        )
        dialogue.append("Doctor: " + doc_question)

        # Patient turn
        patient_role, patient_prompt = get_patient_prompt()
        pat_response = pat_vlm.run(
            patient_role,
            patient_prompt.format(diagnosis, doc_question),
            image_path
        )
        pat_response = clean_patient_response(pat_response, config)
        dialogue.append("Patient: " + pat_response)

    return dialogue


def main():
    args = parse_args()
    try:
        config_module = importlib.import_module(f"pcdf.config.{args.config}")
        ConfigClass = getattr(config_module, "Config")
        config = ConfigClass()
    except (ModuleNotFoundError, AttributeError):
        raise RuntimeError(
            f"Failed to load config '{args.config}'. "
            "Ensure config/<name>.py defines a Config class."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speciality = args.speciality

    os.makedirs(os.path.dirname(config.train_dialogue_path), exist_ok=True)
    json_file_path = config.train_dialogue_path

    doc_vlm = DocVLM(args.doc_vlm)
    pat_vlm = PatientVLM(args.patient_vlm)

    df = pd.read_csv(config.train_csv_path)
    diagnoses = df["diagnosis"].unique()

    results = {}

    for index, data in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(config.data_root, args.split, data["image"])
        diagnosis = data["diagnosis"].lower()

        dialogue = run_dialogue(
            pat_vlm=pat_vlm,
            doc_vlm=doc_vlm,
            speciality=speciality,
            image_path=image_path,
            diagnoses=diagnoses,
            diagnosis=diagnosis,
            config=config
        )

        image_id = int(
            os.path.basename(data["image"]).replace(".png", "")
        )

        results[str(image_id)] = {
            "dialogue": dialogue,
            "diagnosis": diagnosis,
            "image_path": image_path
        }

        # periodic checkpoint
        if index % 100 == 0:
            with open(json_file_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(json_file_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
