import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast
from tqdm import tqdm

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split, use_schema=True, curriculum_stage=None):
        """
        Initialize T5Dataset
        
        Args:
            data_folder: path to data folder (e.g., 'data')
            split: 'train', 'dev', or 'test'
            use_schema: whether to add schema info to input
            curriculum_stage: None, 'easy', 'medium', or 'hard' for curriculum learning
        """
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.use_schema = use_schema
        self.curriculum_stage = curriculum_stage
        
        #add schema information
        self.schema_info = " | SCHEMA: flights(flight_id,from_airport,to_airport,airline_code,flight_days), airports(airport_code,city_code), cities(city_name,city_code), airlines(airline_code,airline_name)" 
        self.data = self.process_data(data_folder, split)
        
    def process_data(self, data_folder, split):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        #load NL queries
        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines() if line.strip()]
        
        #add schema info to queries
        if self.use_schema:
            nl_queries = [f"{q}{self.schema_info}" for q in nl_queries]
        
        if split != 'test':
            with open(sql_path, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines() if line.strip()]
            
            #handle dev.nl/dev.sql mismatch
            if split == 'dev' and len(nl_queries) > len(sql_queries):
                nl_queries = nl_queries[:len(sql_queries)]
        else:
            sql_queries = None
        
        #tokenize NL queries
        nl_encodings = self.tokenizer(
            nl_queries,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None
        )
        
        data = []
        
        if split != 'test':
            #tokenize SQL queries
            sql_encodings = self.tokenizer(
                sql_queries,truncation=True,max_length=768, padding=False,return_tensors=None)
                        
            #dataset with filtering
            for i in range(len(nl_queries)):
                encoder_ids = torch.tensor(nl_encodings['input_ids'][i], dtype=torch.long)
                labels = torch.tensor(sql_encodings['input_ids'][i], dtype=torch.long)
                sql_length = len(labels)
                
                #learning filter
                if self.curriculum_stage == 'easy' and sql_length > 150:
                    continue
                elif self.curriculum_stage == 'medium' and (sql_length <= 150 or sql_length > 300):
                    continue
                elif self.curriculum_stage == 'hard' and sql_length <= 300:
                    continue
                
                data.append((encoder_ids, labels, sql_length))
        else:
            for i in range(len(nl_queries)):
                encoder_ids = torch.tensor(nl_encodings['input_ids'][i], dtype=torch.long)
                data.append((encoder_ids,))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    if len(batch[0]) == 3:  # train/dev with sql_length
        encoder_ids_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
    else:
        encoder_ids_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    return encoder_ids, encoder_mask, labels


def test_collate_fn(batch):
    encoder_ids_list = [item[0] for item in batch]
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    return encoder_ids, encoder_mask


def get_dataloader(batch_size, split, use_schema=True, curriculum_stage=None):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, use_schema=use_schema, curriculum_stage=curriculum_stage)
    shuffle = split == 'train'
    collate_fn = normal_collate_fn if split != 'test' else test_collate_fn
    
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size, use_schema=True, curriculum_stage=None):
    train_loader = get_dataloader(batch_size, 'train', use_schema=use_schema, curriculum_stage=curriculum_stage)
    dev_loader = get_dataloader(test_batch_size, 'dev', use_schema=use_schema)
    test_loader = get_dataloader(test_batch_size, 'test', use_schema=use_schema)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines