import sys
import torch
import transformers


MODEL_NAME = "gpt2"


def gpt_custom_gen():

    input_str = "Once upon a time in Hollywood"

    # Create a model
    model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer([input_str], return_tensors='pt')


    # Take IDs of tokens and predict next token one at a time
    input_ids = enc["input_ids"]
    for i in range(20):
        attention_mask = torch.ones(input_ids.shape, dtype = torch.int64)
        # Do not use the 'generate' function here to not get the output of the
        # last layer of the model. 
        logits = model(input_ids = input_ids, attention_mask = attention_mask)['logits']
        # ID of the next predicted token
        next_id = logits[:, -1, :].argmax(dim = 1)
        # Add next ID to existing ones
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim = 1)
        print("\n\n" + (tokenizer.batch_decode(input_ids))[0])
        

if __name__ == "__main__":
    gpt_custom_gen()