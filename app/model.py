import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

class CRISPRBertModel(nn.Module):
    def _init_(self, bert_model):
        super()._init_()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return self.sigmoid(logits)

class CRISPRPredictor:
    def _init_(self, model_path=None, tokenizer_path=None):  # Make parameters optional with default None
        if model_path is None:
            model_path = 'path/to/your/model.pt'  # Set your default path
        if tokenizer_path is None:
            tokenizer_path = 'bert-base-uncased'  # Default BERT tokenizer
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and BERT model
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            bert_model = BertModel.from_pretrained(tokenizer_path)
            
            # Initialize CRISPR model
            self.model = CRISPRBertModel(bert_model).to(self.device)
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def preprocess_sequence(self, seq):
        """Preprocess DNA sequence"""
        seq = seq.replace('_', '[MASK]')
        if seq.startswith('-'):
            seq = seq[1:] + '-'
        return seq

    def predict(self, target_seq, offtarget_seq):
        """Predict off-target risk"""
        try:
            # Preprocess sequences
            target_seq = self.preprocess_sequence(target_seq)
            offtarget_seq = self.preprocess_sequence(offtarget_seq)
            
            # Combine sequences
            combined_seq = f"{target_seq} [SEP] {offtarget_seq}"
            
            # Tokenize
            encoding = self.tokenizer(
                combined_seq,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Prepare input
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_ids, attention_mask)
                prob = output.squeeze().item()
                return max(0, min(1, prob))  # Ensure output is between 0 and 1
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return 0.0  # Return default value on error