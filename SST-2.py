import time
import torch
import random
import contextlib
import numpy as np
from torch import nn, optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Some parameters
n_classes = 2
batch_size = 16
max_epochs = 10

# Some hyperparameters
learning_rates = [1e-4, 1e-5, 1e-6]


# Load Dataset

train_dataset = load_dataset("gpt3mix/sst2", split="train")
val_dataset = load_dataset("gpt3mix/sst2", split="validation")
test_dataset = load_dataset("gpt3mix/sst2", split="test")

# Extracting 'text1' and 'text2' columns into a list of lists

train_text_data = list(train_dataset['text'])
val_text_data = list(val_dataset['text'])
test_text_data = list(test_dataset['text'])

# Extracting the 'label' column into a list

train_label = train_dataset['label']
val_label = val_dataset['label']
test_label = test_dataset['label']



# Tokenize the text data

tokenizer_tiny = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
tokenizer_mini = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

# Define a custom PyTorch Dataset
class CreateDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=512):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.tokenized_inputs = self.tokenize_sentences()

    def tokenize_sentences(self):
        return self.tokenizer(
            self.sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_inputs['input_ids'][idx],
            'attention_mask': self.tokenized_inputs['attention_mask'][idx],
            'token_type_ids': self.tokenized_inputs.get('token_type_ids', None)[idx],
            'label': torch.tensor(self.labels[idx])
        }

    def create_dataloader(self, batch_size=64, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    

# Create an instance of the custom dataset for tiny bert
dataset_tiny_train = CreateDataset(train_text_data, train_label, tokenizer_tiny)
dataset_tiny_val = CreateDataset(val_text_data, val_label, tokenizer_tiny)
dataset_tiny_test = CreateDataset(test_text_data, test_label, tokenizer_tiny)

# Create dataloaders for tiny bert
dataloader_tiny_train = dataset_tiny_train.create_dataloader(batch_size=64, shuffle=False)
dataloader_tiny_val = dataset_tiny_val.create_dataloader(batch_size=64, shuffle=False)
dataloader_tiny_test = dataset_tiny_test.create_dataloader(batch_size=64, shuffle=False)

# Create an instance of the custom dataset for mini bert
dataset_mini_train = CreateDataset(train_text_data, train_label, tokenizer_mini)
dataset_mini_val = CreateDataset(val_text_data, val_label, tokenizer_mini)
dataset_mini_test = CreateDataset(test_text_data, test_label, tokenizer_mini)

# Create dataloaders for mini bert
dataloader_mini_train = dataset_mini_train.create_dataloader(batch_size=64, shuffle=False)
dataloader_mini_val = dataset_mini_val.create_dataloader(batch_size=64, shuffle=False)
dataloader_mini_test = dataset_mini_test.create_dataloader(batch_size=64, shuffle=False)



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
    

# Training Models



# tiny bert fine-tuning
start_time_1 = time.time()

print("1. Tiny Bert with fine-tuning")

# Initialize some variables for keeping track of the best model
best_model = None
best_epoch = None
best_learning_rate = None
best_val_accuracy = 0

for lr in learning_rates:
    print(f'Learning rate: {lr}')


    model_tiny = BertClassifier('prajjwal1/bert-tiny', n_classes, freeze_bert=False)
    model_tiny.to(device)

    # Using cross entropy loss for loss computation
    loss_fn = nn.CrossEntropyLoss()

    # Using Adam optimizer for optimization
    optimizer = optim.Adam(model_tiny.parameters(), lr=lr)

    for ep in range(1, max_epochs+1):
        print(f'epoch: {ep}')
        train_loss = []

        for batch in dataloader_tiny_train:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_tiny(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        print(f'average training loss: {np.mean(train_loss)}')

        val_loss = []
        val_acc = []

        for batch in dataloader_tiny_val:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_tiny(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            val_loss.append(loss.item())

            _, preds = torch.max(outputs, dim=1)

            acc = torch.sum(preds == labels).item() / len(labels)

            val_acc.append(acc)

        print(f'average validation loss: {np.mean(val_loss)}')
        print(f'average validation accuracy: {np.mean(val_acc)}')

        # Save the best model based on validation accuracy
        if best_val_accuracy < np.mean(val_acc):
            best_val_accuracy = np.mean(val_acc)
            best_model = model_tiny.state_dict()
            best_epoch = ep
            best_learning_rate = lr

# Save the best model
torch.save(best_model, f'SST_best_model_tiny_fine-tuned_epoch_{best_epoch}_lr_{best_learning_rate}.pth')

# After the training loop completes, report the best learning rate and lowest validation loss
print(f'Best epoch: {best_epoch}')
print(f'Best learning rate: {best_learning_rate}')
print(f'Best validation accuracy: {best_val_accuracy}')

# Calculate the test accuracy
test_acc = []
with torch.no_grad():
    for batch in dataloader_tiny_test:
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model_tiny(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)

        acc = torch.sum(preds == labels).item() / len(labels)

        test_acc.append(acc)

print(f'average test accuracy: {np.mean(test_acc)}')

print(f"Time taken for tiny bert fine-tune: {time.time() - start_time_1:.5f} seconds")



# mini bert fine-tuning
start_time_2 = time.time()

print("\n")
print("\n")
print("\n")
print("\n")
print("2. Mini Bert with fine-tuning")

# Initialize some variables for keeping track of the best model
best_model = None
best_epoch = None
best_learning_rate = None
best_val_accuracy = 0

for lr in learning_rates:
    print(f'Learning rate: {lr}')


    model_mini = BertClassifier('prajjwal1/bert-mini', n_classes, freeze_bert=False)
    model_mini.to(device)

    # Using cross entropy loss for loss computation
    loss_fn = nn.CrossEntropyLoss()

    # Using Adam optimizer for optimization
    optimizer = optim.Adam(model_mini.parameters(), lr=lr)

    for ep in range(1, max_epochs+1):
        print(f'epoch: {ep}')
        train_loss = []

        for batch in dataloader_mini_train:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_mini(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        print(f'average training loss: {np.mean(train_loss)}')

        val_loss = []
        val_acc = []

        for batch in dataloader_mini_val:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_mini(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            val_loss.append(loss.item())

            _, preds = torch.max(outputs, dim=1)

            acc = torch.sum(preds == labels).item() / len(labels)

            val_acc.append(acc)

        print(f'average validation loss: {np.mean(val_loss)}')
        print(f'average validation accuracy: {np.mean(val_acc)}')

        # Save the best model based on validation accuracy
        if best_val_accuracy < np.mean(val_acc):
            best_val_accuracy = np.mean(val_acc)
            best_model = model_mini.state_dict()
            best_epoch = ep
            best_learning_rate = lr

# Save the best model
torch.save(best_model, f'SST_best_model_mini_fine-tuned_epoch_{best_epoch}_lr_{best_learning_rate}.pth')

# After the training loop completes, report the best learning rate and lowest validation loss
print(f'Best epoch: {best_epoch}')
print(f'Best learning rate: {best_learning_rate}')
print(f'Best validation accuracy: {best_val_accuracy}')

# Calculate the test accuracy
test_acc = []
with torch.no_grad():
    for batch in dataloader_mini_test:
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model_mini(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)

        acc = torch.sum(preds == labels).item() / len(labels)

        test_acc.append(acc)

print(f'average test accuracy: {np.mean(test_acc)}')

print(f"Time taken for mini bert fine-tune: {time.time() - start_time_2:.5f} seconds")



# tiny bert without fine-tuning
start_time_3 = time.time()

print("\n")
print("\n")
print("\n")
print("\n")
print("3. Tiny Bert without fine-tuning")

# Initialize some variables for keeping track of the best model
best_model = None
best_epoch = None
best_learning_rate = None
best_val_accuracy = 0

for lr in learning_rates:
    print(f'Learning rate: {lr}')


    model_tiny = BertClassifier('prajjwal1/bert-tiny', n_classes, freeze_bert=True)
    model_tiny.to(device)

    # Using cross entropy loss for loss computation
    loss_fn = nn.CrossEntropyLoss()

    # Using Adam optimizer for optimization
    optimizer = optim.Adam(model_tiny.parameters(), lr=lr)

    for ep in range(1, max_epochs+1):
        print(f'epoch: {ep}')
        train_loss = []

        for batch in dataloader_tiny_train:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_tiny(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        print(f'average training loss: {np.mean(train_loss)}')

        val_loss = []
        val_acc = []

        for batch in dataloader_tiny_val:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_tiny(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            val_loss.append(loss.item())

            _, preds = torch.max(outputs, dim=1)

            acc = torch.sum(preds == labels).item() / len(labels)

            val_acc.append(acc)

        print(f'average validation loss: {np.mean(val_loss)}')
        print(f'average validation accuracy: {np.mean(val_acc)}')

        # Save the best model based on validation accuracy
        if best_val_accuracy < np.mean(val_acc):
            best_val_accuracy = np.mean(val_acc)
            best_model = model_tiny.state_dict()
            best_epoch = ep
            best_learning_rate = lr

# Save the best model
torch.save(best_model, f'SST_best_model_tiny_without_fine-tuning_epoch_{best_epoch}_lr_{best_learning_rate}.pth')

# After the training loop completes, report the best learning rate and lowest validation loss
print(f'Best epoch: {best_epoch}')
print(f'Best learning rate: {best_learning_rate}')
print(f'Best validation accuracy: {best_val_accuracy}')

# Calculate the test accuracy
test_acc = []
with torch.no_grad():
    for batch in dataloader_tiny_test:
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model_tiny(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)

        acc = torch.sum(preds == labels).item() / len(labels)

        test_acc.append(acc)

print(f'average test accuracy: {np.mean(test_acc)}')

print(f"Time taken for tiny bert without fine-tune: {time.time() - start_time_3:.5f} seconds")



# mini bert without fine-tuning
start_time_4 = time.time()

print("\n")
print("\n")
print("\n")
print("\n")
print("4. Mini Bert without fine-tuning")

# Initialize some variables for keeping track of the best model
best_model = None
best_epoch = None
best_learning_rate = None
best_val_accuracy = 0

for lr in learning_rates:
    print(f'Learning rate: {lr}')
    

    model_mini = BertClassifier('prajjwal1/bert-mini', n_classes, freeze_bert=True)
    model_mini.to(device)

    # Using cross entropy loss for loss computation
    loss_fn = nn.CrossEntropyLoss()

    # Using Adam optimizer for optimization
    optimizer = optim.Adam(model_mini.parameters(), lr=lr)

    for ep in range(1, max_epochs+1):
        print(f'epoch: {ep}')
        train_loss = []

        for batch in dataloader_mini_train:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_mini(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        print(f'average training loss: {np.mean(train_loss)}')

        val_loss = []
        val_acc = []

        for batch in dataloader_mini_val:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model_mini(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            val_loss.append(loss.item())

            _, preds = torch.max(outputs, dim=1)

            acc = torch.sum(preds == labels).item() / len(labels)

            val_acc.append(acc)

        print(f'average validation loss: {np.mean(val_loss)}')
        print(f'average validation accuracy: {np.mean(val_acc)}')

        # Save the best model based on validation accuracy
        if best_val_accuracy < np.mean(val_acc):
            best_val_accuracy = np.mean(val_acc)
            best_model = model_mini.state_dict()
            best_epoch = ep
            best_learning_rate = lr

# Save the best model
torch.save(best_model, f'SST_best_model_mini_without_fine-tuning_epoch_{best_epoch}_lr_{best_learning_rate}.pth')

# After the training loop completes, report the best learning rate and lowest validation loss
print(f'Best epoch: {best_epoch}')
print(f'Best learning rate: {best_learning_rate}')
print(f'Best validation accuracy: {best_val_accuracy}')

# Calculate the test accuracy
test_acc = []
with torch.no_grad():
    for batch in dataloader_mini_test:
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model_mini(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)

        acc = torch.sum(preds == labels).item() / len(labels)

        test_acc.append(acc)

print(f'average test accuracy: {np.mean(test_acc)}')

print(f"Time taken for mini bert without fine-tune: {time.time() - start_time_4:.5f} seconds")




# Function to generate random predictions
def random_classifier_predictions(num_samples):
    return [random.choice([0, 1]) for _ in range(num_samples)]

# Generate random predictions for the test set
random_preds = random_classifier_predictions(len(dataloader_tiny_test.dataset))

# Calculate random classifier test accuracy
random_test_acc = sum(pred == label for pred, label in zip(random_preds, dataloader_tiny_test.dataset.labels)) / len(random_preds)

print("\n")
print("\n")
print("\n")
print("\n")
print(f'Random classifier test accuracy: {random_test_acc}')