import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import os
from utils import custom_transform, tokenize_function

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

dataset = load_dataset("imdb")

original_train = dataset["train"]
sampled = original_train.shuffle(seed=42).select(range(5000))
transformed = sampled.map(custom_transform, load_from_cache_file=False)

tokenized_original = original_train.map(tokenize_function, batched=True, load_from_cache_file=False)
tokenized_transformed = transformed.map(tokenize_function, batched=True, load_from_cache_file=False)

combined_dataset = concatenate_datasets([tokenized_original, tokenized_transformed])
combined_dataset = combined_dataset.remove_columns(["text"])
combined_dataset = combined_dataset.rename_column("label", "labels")
combined_dataset.set_format("torch")

train_dataloader = DataLoader(combined_dataset, shuffle=True, batch_size=8)

#create eval dataloader from test set
test_dataset = dataset["test"].map(tokenize_function, batched=True, load_from_cache_file=False)
test_dataset = test_dataset.remove_columns(["text"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch")
eval_dataloader = DataLoader(test_dataset, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained("out", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * 1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

model.train()
for batch in train_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

#save model
os.makedirs("out_augmented", exist_ok=True)
model.save_pretrained("out_augmented")
tokenizer.save_pretrained("out_augmented")
print("✓ Model saved to out_augmented/")

#eval and save predictions
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels) * 100

with open("out_augmented_original.txt", 'w') as f:
    for pred, label in zip(all_predictions, all_labels):
        f.write(f"{pred} {label}\n")

print(f"Augmented Model Test Accuracy: {accuracy:.2f}%")
print(f"Predictions saved to out_augmented_original.txt")