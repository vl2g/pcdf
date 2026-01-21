# PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)

A novel framework that simulates pre-consultation dialogues between two Vision-Language Models (VLMs) - PatientVLM and DocVLM - to enhance medical diagnosis efficiency. This work was presented at AAAI 2026.

---

## Table of Contents
- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Data Preparation](#data-preparation)
- [Pre-Consultation Dialogue Generation](#pre-consultation-dialogue-generation)
- [DocVLM Inference and Evaluation](#docvlm-inference-and-evaluation)
- [Prerequisites](#prerequisites)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

This repository implements a Pre-Consultation Dialogue Framework (PCDF) where:
- **PatientVLM**: Simulates patient perspective, describing symptoms, patient history, and medical images
- **DocVLM**: Acts as a clinical expert, asking diagnostic questions and making assessments

The dialogue-based approach improves diagnostic accuracy by leveraging the complementary strengths of different VLMs.

---

## Installation Guide

### 1. Clone the Repository
```sh
git clone https://github.com/vl2g/pcdf.git
cd pcdf
```

### 2. Setup Conda Environment and Install Dependencies
```sh
conda create --name pcdf python=3.11.14 -y
conda activate pcdf
pip install -r requirements.txt
```

---

## Data Preparation

### Prepare MedMNIST Dataset

Process the MedMNIST dataset for your desired medical specialty:
```sh
python scripts/process_medmnist.py --dataset dermamnist
```

**Supported Datasets:**
- `dermamnist` - Dermatology images
- `pathmnist` - Pathology images  
- `retinamnist` - Retinal fundus images
- `pneumoniamnist` - Chest X-rays (Radiology/Pneumonia)

---

## Pre-Consultation Dialogue Generation

Generate dialogues between PatientVLM and DocVLM using the PCDF framework.

### Default Command
```sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

```sh
python scripts/run_pcdf.py \
    --speciality Dermatology \
    --doc_vlm Qwen25VL \
    --patient_vlm mPLUGOwl3 \
    --config dermamnist \
    --split train
```


### Arguments

| Argument | Description | Default | Other Options |
|----------|-------------|---------|---------------|
| `--speciality` | Medical specialty domain | `Dermatology` | `Pathology`, `Ophthalmology`, `Radiology` |
| `--doc_vlm` | Vision-Language Model for doctor role | `Qwen25VL` | `Gemma3`, `MedGemma`, `InternVL3`, `mPLUGOwl3` |
| `--patient_vlm` | Vision-Language Model for patient role | `mPLUGOwl3` | `Gemma3`, `MedGemma`, `InternVL3`, `Qwen25VL` |
| `--config` | Dataset configuration file | `dermamnist` | `pathmnist`, `retinamnist`, `pneumoniamnist` |
| `--split` | Dataset split to process | `train` | `test`, `val` |

### Supported Vision-Language Models

- **Gemma3**: Google's Gemma 3 vision-language model
- **MedGemma**: Medical domain-adapted version of Gemma
- **InternVL3**: InternVL version 3 multimodal model
- **Qwen25VL**: Qwen 2.5 Vision-Language model
- **mPLUGOwl3**: Alibaba's mPLUG-Owl 3 model

### Specialty-Dataset Mapping

| Specialty | Recommended Config | Description |
|-----------|-------------------|-------------|
| `Dermatology` | `dermamnist` | Skin lesion classification |
| `Pathology` | `pathmnist` | Histopathology image analysis |
| `Ophthalmology` | `retinamnist` | Retinal fundus imaging |
| `Radiology` | `pneumoniamnist` | Pneumonia detection from chest X-rays |

---

## DocVLM Inference and Evaluation

After finetuning the DocVLM on pre-consultation dialogues, evaluate the DocVLM's diagnostic performance.

### 1. Setup Inference Environment

Create a separate environment for inference (requires Python 3.10):
```sh
conda create --name docvlm python=3.10.19 -y
conda activate docvlm
pip install -r inference/requirements.txt
```

### 2. Run DocVLM Inference

Execute inference using the trained DocVLM:
```sh
python inference/qwen25vl_inference.py
```

This script processes the generated PCDF dialogues and produces diagnostic predictions.

### 3. Evaluate Results
```sh
python inference/evaluate.py --json_path results/pcdf_dermamnist.json
```

**Arguments:**
- `--json_path`: Path to the inference results JSON file

---

## Citation

If you find this work useful in your research, please consider citing:
```bibtex
@inproceedings{lokesh2026patientvlm,
  title     = {PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis},
  author    = {Lokesh, K and Penamakuri, Abhirama Subramanyam and Agarwal, Uday and Challa, Apoorva and Gowda, Shreya K and Gupta, Somesh and Mishra, Anand},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  publisher = {AAAI Press}
}
```

---