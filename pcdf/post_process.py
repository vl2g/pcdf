import re
import pandas as pd

REMOVE_WORDS = {
    "normal", "no", "diabetic", "like", "lesion", "lesions", "associated",
    "background", "smooth", "muscle", "cell", "cancer"
}

def clean_diagnosis_terms(diagnosis):
    words = re.findall(r'\b\w+\b', diagnosis.lower())
    cleaned_words = [w for w in words if w not in REMOVE_WORDS]
    return cleaned_words

def load_medical_keywords(config):
    df = pd.read_csv(config.test_csv_path)
    keywords = set()

    if 'diagnosis' in df.columns:
        for diagnosis in df['diagnosis'].dropna().unique():
            cleaned_words = clean_diagnosis_terms(diagnosis)
            for w in cleaned_words:
                keywords.add(w)
            keywords.add(diagnosis.lower())

    return keywords

def clean_patient_response(response, config):
    medical_keywords = load_medical_keywords(config)
    response_lower = response.lower()
    matched_keywords = [kw for kw in medical_keywords if kw in response_lower]

    if not matched_keywords:
        return response
    sentences = re.split(r'(?<=\.)\s+', response.strip())

    if len(sentences) > 1:
        cleaned_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if not any(kw in sentence_lower for kw in matched_keywords):
                cleaned_sentences.append(sentence)
        cleaned_response = " ".join(cleaned_sentences).strip()
        return cleaned_response if cleaned_response else ""
    else:
        cleaned = response
        for kw in matched_keywords:
            cleaned = re.sub(re.escape(kw), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
