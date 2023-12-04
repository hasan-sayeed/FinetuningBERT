import torch
import contextlib
import pandas as pd
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('hidden_rte.csv')

# Create a list of tuples from the 'text1' and 'text2' columns
tuples_list = list(zip(df['text1'], df['text2']))


tokenizer_mini = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
tokenized_inp_tiny = tokenizer_mini(tuples_list, max_length=512, padding=True, truncation=True, return_tensors="pt")



input_ids = tokenized_inp_tiny['input_ids'].to(device)
token_type_ids = tokenized_inp_tiny['token_type_ids'].to(device)
attention_mask = tokenized_inp_tiny['attention_mask'].to(device)



# Define the model

class BertClassifier(nn.Module):
    def __init__(self, pre_trained_model_name, n_classes, freeze_bert=True):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.freeze_bert = freeze_bert

        # Optionally freeze the BERT layers
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # Forward pass through BERT without gradient computation if freeze_bert=True
        with torch.no_grad() if getattr(self, 'freeze_bert', False) else contextlib.nullcontext():
            out = self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            pooled_output = out.pooler_output

        # Forward pass through the classifier with gradient computation
        return self.out(pooled_output)
    

model = BertClassifier('prajjwal1/bert-tiny', 2)

filename = 'best_model_tiny_epoch_10_lr_1e-06.pth'
model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
model.to(device)


# Perform inference
output = model(input_ids, token_type_ids, attention_mask)



# Apply softmax to get probabilities
probabilities = F.softmax(output, dim=1)

# Get predicted class indices
predicted_classes = torch.argmax(probabilities, dim=1)

# Extract probabilities for class 0 and class 1
probability_class_0 = probabilities[:, 0]
probability_class_1 = probabilities[:, 1]

# Convert the tensors to lists
predicted_classes_list = predicted_classes.tolist()
probability_class_0_list = probability_class_0.tolist()
probability_class_1_list = probability_class_1.tolist()

print("Predicted Classes:", predicted_classes_list)
print("Probabilities for Class 0:", probability_class_0_list)
print("Probabilities for Class 1:", probability_class_1_list)



df['prediction'] = predicted_classes_list
df['probab_0'] = probability_class_0_list
df['probab_1'] = probability_class_1_list


df.to_csv('hidden_rte_predictions.csv', index=False)