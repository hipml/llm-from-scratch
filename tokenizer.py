import re
import tiktoken

# with open('the-verdict.txt', 'r', encoding='utf-8') as f:
#     raw_text = f.read()
# preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
# preprocessed = [item for item in preprocessed if item.strip()]
#  
# all_words = sorted(list(set(preprocessed)))
# all_words.extend(["<|endoftext|>", "<|unk|>"])
# 
# vocab = {token:integer for integer,token in enumerate(all_words)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
       
# tokenizer = SimpleTokenizerV2(vocab)
# tokenizer = tiktoken.get_encoding('gpt2')
# text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))
# 
# text = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text, text2))
# 
# integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
# 
# print(tokenizer.decode(integers))

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f'x: {x}')
print(f'y:      {y}')

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), '---->', tokenizer.decode([desired]))
