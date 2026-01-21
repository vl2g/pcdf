import os
import json
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def evaluate(json_path):
    with open(json_path, "r") as f:
        dataset = json.load(f)

    y_pred = []
    y_true = []

    for _, data in dataset.items():
        pred = data.get("pred_diagnosis", "").strip().lower()
        true = data.get("diagnosis", "").strip().lower()

        if pred == "" or true == "":
            continue

        y_pred.append(pred)
        y_true.append(true)

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average="macro") * 100

    print(f"Total samples evaluated: {len(y_true)}")
    print(f"Accuracy: {accuracy:.1f}")
    print(f"Precision: {precision:.1f}")
    print(f"Recall: {recall:.1f}")
    print(f"Macro F1 Score: {f1:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON file"
    )
    args = parser.parse_args()
    evaluate(args.json_path)
