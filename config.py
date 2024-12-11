# import os

# class Config:
#     SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
#     MODEL_PATH = "model\model_balanced_f1_0.8857_20241210_165107.pt"
#     TOKENIZER_PATH = 'bert-base-uncased'


import os

class Config:
    # Menggunakan SECRET_KEY dari environment variable atau fallback ke default
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    
    # Path model yang diatur secara dinamis berdasarkan lokasi file ini
    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'model', 
        'model_balanced_f1_0.8857_20241210_165107.pt'
    )
    
    # Path tokenizer tetap menggunakan nama tokenizer standar
    TOKENIZER_PATH = 'bert-base-uncased'  # Standard tokenizer name