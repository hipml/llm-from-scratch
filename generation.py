import torch
import torch.nn as nn
import tiktoken

from notebook import generate_text_simple
from src.gpt_model import GPTModel
from notebooks.dataloader import create_dataloader_v1

GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 256,
    "emb_dim": 768, 
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False 
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = 'Every effort moves you'
tokenizer = tiktoken.get_encoding('gpt2')

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M['context_length']
)

print(f'Output text:\n {token_ids_to_text(token_ids, tokenizer)}')

inputs = torch.tensor([[16833, 3626, 6100],
                       [40, 1107, 588]])

targets = torch.tensor([[3626, 6100, 345],
                        [588, 428, 11311]])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape) # [batch_size, num_tokens, emb_dim or vocab_size] => [2, 3, 50257]

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(f'Token IDs:\n{token_ids}')

text_idx = [0, 1]
for idx in text_idx:
    print(f'Targets @ {idx}: {token_ids_to_text(targets[idx], tokenizer)}')
    print(f'Outputs @ {idx}: {token_ids_to_text(token_ids[idx].flatten(), tokenizer)}')

target_probas_1 = probas[0, [0, 1, 2], targets[0]]
target_probas_2 = probas[1, [0, 1, 2], targets[1]]

# for idx in text_idx:
#     print(f'Text {idx}: {target_probas}')

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

print(f'Logits shape: {logits.shape}') # [2, 3, 50257]
print(f'Targets shape: {targets.shape}') # [2, 3]

# for cross entropy loss in PyTorch, flatten these tensors by combining them over the batch dimension
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

print(f'Flattened logits: {logits_flat.shape}') # [6, 50257]
print(f'Flattened targets: {targets_flat.shape}') # [6]

# previously, we applied the softmax function,
# selected the probability scores corresponding to to the target IDs,
# computed the negative avg log probabilities
# pytorch's cross_entropy function will take care of all these steps for us

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

file_path = 'the-verdict.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(f'Characters: {total_characters}')
print(f'Tokens: {total_tokens}')

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    drop_last=True,
    shuffle=True
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    drop_last=False,
    shuffle=False
)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print(f'{train_tokens}, {val_tokens}')

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_loss= calc_loss_loader(train_loader, model, device)
val_loss =  calc_loss_loader(val_loader, model, device)
print(f'Training loss {train_loss}')
print(f'Val loss: {val_loss}')
