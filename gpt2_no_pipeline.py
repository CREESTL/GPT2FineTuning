import sys
import torch
import transformers


MODEL_NAME = "gpt2"



def gpt2_no_pipe():

    input_str = "Robert DeNiro once said"

    # Load model from the HUB
    config = transformers.GPT2Config.from_pretrained(MODEL_NAME)
    # Tweak temperature for output to be random
    config.do_sample = config.task_specific_params['text-generation']['do_sample']
    config.max_length = config.task_specific_params['text-generation']['max_length']
    # Create a model
    model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=config)

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer([input_str], return_tensors='pt')

    # Generate
    out = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], max_length=100)

    # Print output
    print("\n\n" + (tokenizer.batch_decode(out))[0])
    
if __name__ == "__main__":
    gpt2_no_pipe()