# BART-with-lexical-syntactic-constraints

Generate a sentence or a slogan with given keywords and syntactic structure. 


```python
from transformers import BartTokenizer
from SynSemBartForConditionalGeneration import SynSemBartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# special tokens for slogans 
tokenizer.add_tokens('<name>') # brand name 
tokenizer.add_tokens('<loc>') # location name 
tokenizer.add_tokens('<year>') # year established 

pos_tokenizer = BartTokenizer(vocab_file='tokenizer/POSvocab.json', merges_file='tokenizer/merges.txt')

model =  SynSemBartForConditionalGeneration.from_pretrained("model/SynSemBart")
model.resize_token_embeddings(len(tokenizer))

# lexical constraints 
mask_seq = '<mask> bake <mask> cake <mask>'
# syntactic constraints
pos = ['VERB', 'DET', 'NOUN', 'PUNCT', 'VERB', 'DET', 'NOUN']

inputs = tokenizer(mask_seq, return_tensors="pt")
pos_inputs = pos_tokenizer.encode_plus(pos, is_pretokenized=True, return_tensors='pt')
outputs = model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask,
                         pos_input_ids = pos_inputs.input_ids, pos_attention_mask = pos_inputs.attention_mask)[0]

tokenizer.decode(outputs, skip_special_tokens=True)
# >> 'Bake the cake, Get the break.'
```
