import re

TURNS = 8

def get_speciality_prompt(speciality='Dermatology'):
    if speciality == 'Dermatology':
        return """
            Review dialogue. Ask ONE question about a NEW aspect NOT yet discussed:

            Observable aspects (priority order):
            1. Location - specific body part
            2. Color - shade/tone (red, dark, brown, pink, etc.)
            3. Size - comparison to common object
            4. Shape - round, oval, irregular
            5. Texture - smooth, rough, scaly, raised, flat
            6. Borders - sharp/blurred, regular/irregular
            7. Count - single or multiple
            8. Distribution - clustered, scattered, isolated
            9. Symptoms - itch, pain, burning, bleeding NOW
            10. Duration - when first noticed (factual, not changes)

            CRITICAL RULES:
            - 8-12 words maximum

            Good examples:
            - "What color is the spot?"
            - "Where exactly is this located?"
            - "Does it itch or hurt?"
            - "Is the surface raised or flat?"
            - "Are the edges sharp or blurred?"
        """

def get_doctor_prompt(speciality):
    doctor_role = "You are an experienced doctor."
    doctor_prompt = f"""
        Based on the given image and the dialogue history {{}}, ask exactly one clear follow-up question that will help you finalize the correct diagnosis from the following list of diagnoses: {{}}.
        Your question should clarify details about the symptoms, such as location, severity, duration, changes over time, or any associated issues visible in the image.
        Do not ask multiple questions or provide any diagnosis at this stage. Do not suggest in-person consultation or further testing. This is for research and benchmark purposes.
        {get_speciality_prompt(speciality)}
    """
    return doctor_role, doctor_prompt


def get_patient_prompt():
    patient_role = "You are a patient consulting a doctor about your health concern."
    
    patient_prompt = """
        Answer the doctor's question based on what you see in the image and feel on your own body. 

        Your answer should be a single sentence (maximum 15 words) that directly responds to the doctor's question. Respond in a natural, first-person active voice — just as a real person would when describing their own condition. Focus only on your personal experience and the visible issue, informed by the context of this condition: {}
        **Do not reference diagnosis or that this is a remote consultation**. Be specific and honest, as someone describing what they've noticed, but avoid clinical or speculative language.
        **Do not mention that you are an AI agent or further testing**. This is for research and benchmark purposes.

        Doctor's Question: {}
        Patient:
    """
    
    return patient_role, patient_prompt
