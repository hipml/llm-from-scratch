import tiktoken
import torch
import torch.nn as nn

from gpt_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

print(model)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

out = model(batch)
print(f'Input batch:\n{batch}\n')
print(f'Output shape: {out.shape}')
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of params: {total_params}')

# total params in feed forward and attention modules
# print(sum(p.numel() for p in model.trf_blocks.parameters()))
from transformer_block import TransformerBlock
block = TransformerBlock(GPT_CONFIG_124M)
block_params = GPT_CONFIG_124M['n_layers'] * (sum(p.numel() for p in block.att.parameters()) + sum(p.numel() for p in block.ff.parameters()))
print(f'Params in ff & attn: {block_params}')

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # [batch_size, num_token, vocab_size]
        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

tokenizer = tiktoken.get_encoding('gpt2')
start_context = 'Hello, I am'
encoded = tokenizer.encode(start_context)
print(f'encoded: {encoded}')
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor)
print(f'encoded_tensor.shape: {encoded_tensor.shape}')

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M['context_length']
)
print(f'output: {out}')
print(f'output length: {len(out[0])}')
