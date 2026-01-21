from pcdf.agents.models.gemma3 import Gemma3
from pcdf.agents.models.medgemma import MedGemma
from pcdf.agents.models.internvl3 import InternVL3
from pcdf.agents.models.qwen25vl import Qwen25VL
from pcdf.agents.models.mplugowl3 import mPLUGOwl3

class DocVLM():
    def __init__(self, model_name='Gemma3'):
        if model_name == 'Gemma3':
            self.doc_vlm = Gemma3()
        elif model_name == 'MedGemma':
            self.doc_vlm = MedGemma()
        elif model_name == 'InternVL3':
            self.doc_vlm = InternVL3()
        elif model_name == 'Qwen25VL':
            self.doc_vlm = Qwen25VL()
        elif model_name == 'mPLUGOwl3':
            self.doc_vlm = mPLUGOwl3()
        else:
            print(f'DoctorVLM unspecificed; Using default {model_name}')

    def run(self, doctor_role, doctor_prompt, image_path):
        question = self.doc_vlm.run(doctor_role, doctor_prompt, image_path)
        return question
    