import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import torch.nn.functional as F
import statistics
from nltk.translate.bleu_score import sentence_bleu

# Prepare data
df = pd.read_csv('mises_brasil_dataset.csv', encoding='utf-8')
text = df['text']

# create column with text lenght less than 1024
df['text_45'] = df['text'].astype(str).apply(lambda x: ' '.join(x.split(' ')[:45]))

# Create a very small test set to compare generated text with the reality
test_set = df.sample(200)
df = df.loc[~df.index.isin(test_set.index)]

# reset the indexes
test_set = test_set.reset_index()
df = df.reset_index()

class MisesDataset(Dataset):
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.text = []

        for row in df['text_45'].astype(str):
            self.text.append(torch.tensor(self.tokenizer.encode(f'<|{control_code}|> {row[:max_length]} <|endoftext|>')))
        
        if truncate:
            self.text = self.text[:20000]
        self.text_count = len(self.text)
    
    def __len__(self):
        return self.text_count
    
    def __getitem__(self, item):
        return self.text[item]

dataset = MisesDataset(df['text_45'], truncate=True, gpt2_type='gpt2')

# get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
        dataset, model, tokenizer,
        batch_size=16, epochs=5, lr=2e-5,
        max_seq_len=400, warmup_steps=200,
        gpt_type='gpt2', output_dir='.', output_prefix='wreckgar',
        test_mode=False, save_model_on_epoch=False,
):
    acc_steps = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        print(f'Training epoch {epoch}')
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue
        
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if(accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(model.state_dict(), f'{output_prefix}-{epoch}.pt')
    return model

model = train(dataset, model, tokenizer)

def generate(model,
             tokenizer,
             prompt,
             entry_count=10,
             entry_length=30, #maximum length of the generated text
             top_p= 0.8,
             temperature=1.0,):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float('Inf')

    with torch.no_grad():
        for entry_idx in trange(entry_count):
            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :]   / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode('<|endoftext|>'):
                    entry_finished = True
                    
                if entry_finished:
                    generated_num += 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f'{tokenizer.decode(output_list)}<|endoftext|>'
                generated_list.append(output_text)
    
    return generated_list

# Function to generate multiple sentences. Test data should be a dataframe with a column named 'text_45'
def text_generation(test_data):
    generated_text = []
    for i in range(len(test_data)):
        x = generate(model.to('cpu'), tokenizer, test_data['text_45'][i], entry_count=1)
        generated_text.append(x)
    return generated_text

# Run the function to generate the text
generated_text = text_generation(test_set)