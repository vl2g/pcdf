class Config:
    def __init__(self):
        self.data_root = "MedMNIST/DermaMNIST"

        self.train_csv_path = "data/dermamnist/train.csv"
        self.test_csv_path = "data/dermamnist/test.csv"
        self.val_csv_path = "data/dermamnist/val.csv"
        
        self.train_dialogue_path = "experiments/dermamnist_train.json"
        self.test_dialogue_path = "experiments/dermamnist_test.json"
        self.val_dialogue_path = "experiments/dermamnist_val.json"

        self.label_map = {
            'vascular lesions': 0,
            'dermatofibroma': 1,
            'melanocytic nevi': 2,
            'actinic keratoses': 3,
            'benign keratosis-like lesions': 4,
            'basal cell carcinoma': 5,
            'melanoma': 6
        }