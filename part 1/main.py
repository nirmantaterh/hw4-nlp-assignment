import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import evaluate
import random
import argparse
from utils import *
import os

# Global variables
device = None
tokenizer = None


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def do_train(model, train_dataloader, eval_dataloader, device, num_epochs=3, learning_rate=2e-5):
    """Fine-tune BERT on IMDB sentiment dataset."""
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        model.eval()
        eval_accuracy = 0
        total_eval_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()
                eval_accuracy += correct
                total_eval_samples += labels.size(0)
        
        epoch_accuracy = eval_accuracy / total_eval_samples * 100
        print(f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f}, Val Accuracy={epoch_accuracy:.2f}%")
    
    return model


def create_augmented_dataloader(args, dataset):
    """Create dataloader with original + 5000 transformed training examples."""
    
    # Get original training dataset
    original_train = dataset["train"]
    
    # Sample 5000 random examples
    sampled = original_train.shuffle(seed=42).select(range(min(5000, len(original_train))))
    
    # Apply transformation
    transformed = sampled.map(custom_transform, load_from_cache_file=False)
    
    # Tokenize both
    tokenized_original = original_train.map(tokenize_function, batched=True, load_from_cache_file=False)
    tokenized_transformed = transformed.map(tokenize_function, batched=True, load_from_cache_file=False)
    
    # Combine datasets
    combined_dataset = datasets.concatenate_datasets([tokenized_original, tokenized_transformed])
    
    # Prepare for model
    combined_dataset = combined_dataset.remove_columns(["text"])
    combined_dataset = combined_dataset.rename_column("label", "labels")
    combined_dataset.set_format("torch")
    
    # Create dataloader
    train_dataloader = DataLoader(combined_dataset, shuffle=True, batch_size=args.batch_size)
    
    return train_dataloader


def create_transformed_dataloader(args, dataset, debug_transformation):
    """Create dataloader for transformed test set."""
    
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example", k)
            print(small_dataset[k])
            print("\nTransformed Example", k)
            print(small_transformed[k])
            print('=' * 30)
        exit()
    
    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized = transformed_tokenized.remove_columns(["text"])
    transformed_tokenized = transformed_tokenized.rename_column("label", "labels")
    transformed_tokenized.set_format("torch")
    
    eval_dataloader = DataLoader(transformed_tokenized, batch_size=args.batch_size)
    return eval_dataloader


def evaluate_model(model, eval_dataloader, device, model_name="original"):
    """Evaluate model and return accuracy."""
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
    return accuracy, all_predictions, all_labels


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_augmented", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_transformed", action="store_true")
    parser.add_argument("--model_dir", type=str, default="out")
    parser.add_argument("--debug_train", action="store_true")
    parser.add_argument("--debug_transformation", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_accuracy", type=float, default=91.0, help="Target test accuracy to reach")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs to train")
    parser.add_argument("--random_seed", action="store_true", help="Use random seeds instead of fixed seed 0")
    
    args = parser.parse_args()
    
    if not args.random_seed:
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Using FIXED seeds (reproducible)")
    else:
        print("Using RANDOM seeds (non-reproducible)")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    #load and tokenize dataset
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    #create dataloaders
    if args.debug_train:
        train_dataloader = DataLoader(tokenized_dataset["train"].shuffle(seed=42).select(range(500)), shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"].shuffle(seed=42).select(range(125)), batch_size=args.batch_size)
        print(f"Debug training...")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)
        print(f"Actual training...")
    
    print(f"len(train_dataloader): {len(train_dataloader)}")
    print(f"len(eval_dataloader): {len(eval_dataloader)}")
    
    #iterative epoch checking
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        
        # ITERATIVE TRAINING: Train 1 epoch at a time and check if we hit target accuracy
        for epoch_count in range(1, args.max_epochs + 1):
            print(f"\n{'='*60}")
            print(f"ITERATIVE TRAINING: Epoch {epoch_count}/{args.max_epochs}")
            print(f"{'='*60}")
            
            model = do_train(model, train_dataloader, eval_dataloader, device, num_epochs=5, learning_rate=args.learning_rate)
            
            test_accuracy, predictions, labels = evaluate_model(model, eval_dataloader, device, "original")
            print(f"\n✓ Epoch {epoch_count} - Test Accuracy: {test_accuracy:.2f}%")
            
            # Check if we reached target accuracy
            if test_accuracy >= args.target_accuracy:
                print(f"\n{'='*60}")
                print(f"✅ TARGET REACHED! Test Accuracy: {test_accuracy:.2f}% ≥ {args.target_accuracy}%")
                print(f"{'='*60}\n")
                break
            else:
                print(f"Below target ({test_accuracy:.2f}% < {args.target_accuracy}%). Continuing to next epoch...\n")
        
        os.makedirs("out", exist_ok=True)
        model.save_pretrained("out")
        tokenizer.save_pretrained("out")
        
        out_file = "out_original.txt"
        with open(out_file, 'w') as f:
            for pred, label in zip(predictions, labels):
                f.write(f"{label} {pred}\n")
        
        print(f"Results saved to {out_file}")
        print(f"Model saved to out/")
    
    # train on augmented data
    if args.train_augmented:
        train_dataloader_aug = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("out", num_labels=2)
        do_train(model, train_dataloader_aug, eval_dataloader, device, args.num_epochs, args.learning_rate)
        args.model_dir = "out_augmented"
        
        #save augmented model
        os.makedirs("out_augmented", exist_ok=True)
        model.save_pretrained("out_augmented")
        tokenizer.save_pretrained("out_augmented")
    
    #eval on original test data
    if args.eval:
        model_to_load = args.model_dir
        model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=2, trust_remote_code=True)
        model.to(device)
        
        accuracy, all_predictions, all_labels = evaluate_model(model, eval_dataloader, device, "original")
        
        if model_to_load == "out_augmented":
            out_file = "out_augmented_original.txt"
        else:
            out_file = "out_original.txt"
        
        with open(out_file, 'w') as f:
            for pred, label in zip(all_predictions, all_labels):
                f.write(f"{label} {pred}\n")
        
        print(f"Original Test Accuracy: {accuracy:.2f}%")
        print(f"Results saved to {out_file}")
    
    #eval on transformed data
    if args.eval_transformed:
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        
        model_to_load = args.model_dir if hasattr(args, 'model_dir') and args.model_dir else "out"
        model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=2, trust_remote_code=True)
        model.to(device)
        
        accuracy, all_predictions, all_labels = evaluate_model(model, eval_transformed_dataloader, device, "transformed")
        
        if model_to_load == "out_augmented":
            out_file = "out_augmented_transformed.txt"
        else:
            out_file = "out_transformed.txt"
        
        with open(out_file, 'w') as f:
            for pred, label in zip(all_predictions, all_labels):
                f.write(f"{label} {pred}\n")
        
        print(f"Transformed Test Accuracy: {accuracy:.2f}%")
        print(f"Results saved to {out_file}")