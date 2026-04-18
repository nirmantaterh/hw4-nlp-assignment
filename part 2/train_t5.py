import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb

from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    parser = argparse.ArgumentParser(description='T5 training loop')
    
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"])
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=30)
    parser.add_argument('--patience_epochs', type=int, default=7)
    
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    
    args = parser.parse_args()
    return args


def ensure_dirs_exist():
    dirs = ['checkpoints', 'results', 'records', 'data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def train(args, model, tokenizer, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0
    
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    gt_sql_path = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{model_type}_experiment_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_experiment_{args.experiment_name}_dev.pkl')
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, tokenizer, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        
        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {tr_loss:.4f} | Eval Loss: {eval_loss:.4f}")
        print(f"F1: {record_f1:.4f} | SQL EM: {sql_em:.4f} | Err: {error_rate*100:.2f}%")
        
        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            print(f"🔥 BEST F1: {best_f1:.4f}")
            save_model(checkpoint_dir, model, best=True)
        else:
            epochs_since_improvement += 1
        
        save_model(checkpoint_dir, model, best=False)
        
        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping at epoch {epoch}")
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_batches = 0
    
    optimizer.zero_grad()
    
    for step, (encoder_input, encoder_mask, labels) in enumerate(tqdm(train_loader, desc="Training")):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            total_loss += loss.item() * args.gradient_accumulation_steps
            total_batches += 1
    
    if (step + 1) % args.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / total_batches


def eval_epoch(args, model, tokenizer, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    total_loss = 0
    total_batches = 0
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, labels in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1
            
            # Simple greedy generation - FAST and WORKS
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=768,
                num_beams=1,  # Greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            for gen_ids in generated:
                sql_string = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_queries.append(sql_string)
    
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    error_rate = len([e for e in error_msgs if e]) / len(error_msgs) if error_msgs else 0
    eval_loss = total_loss / total_batches if total_batches > 0 else 0
    
    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, tokenizer, test_loader, model_sql_path, model_record_path):
    model.eval()
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask in tqdm(test_loader, desc="Test inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            #using beam search for final test only
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=768,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            for gen_ids in generated:
                sql_string = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_queries.append(sql_string)
    
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    print(f"Test predictions saved to {model_sql_path}")


def main():
    args = get_args()
    ensure_dirs_exist()
    
    if args.use_wandb:
        setup_wandb(args)
    
    print("Loading tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("Loading data...")
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    print(f"Train: {len(train_loader)} batches")
    print(f"Dev: {len(dev_loader)} batches")
    print(f"Test: {len(test_loader)} batches")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    print("\nInitializing model...")
    model = initialize_model(args)
    print(f"Model on {DEVICE}")
    
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader) // args.gradient_accumulation_steps)
    print(f"Optimizer: {args.optimizer_type}, LR: {args.learning_rate}")
    print(f"Scheduler: {args.scheduler_type}")
    
    print("\nTraining...")
    train(args, model, tokenizer, train_loader, dev_loader, optimizer, scheduler)
    
    print("\nLoading best model...")
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = os.path.join('results', f't5_{model_type}_experiment_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_experiment_{args.experiment_name}_dev.pkl')
    
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, tokenizer, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    print(f"\n Final Dev Results:")
    print(f"   Loss: {dev_loss:.4f}")
    print(f"   Record F1: {dev_record_f1:.4f}")
    print(f"   Record EM: {dev_record_em:.4f}")
    print(f"   SQL EM: {dev_sql_em:.4f}")
    print(f"   Error Rate: {dev_error_rate*100:.2f}%")
    
    print("\nInferring on test set...")
    model_sql_path = os.path.join('results', f't5_{model_type}_experiment_{args.experiment_name}_test.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_experiment_{args.experiment_name}_test.pkl')
    test_inference(args, model, tokenizer, test_loader, model_sql_path, model_record_path)
    
    print("\ntraining completed")


if __name__ == "__main__":
    main()