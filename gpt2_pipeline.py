import sys
import torch
import transformers


MODEL_NAME = "gpt2"


def gpt2_pipe():
    pipe = transformers.pipeline(
        task = "text-generation",
        model = MODEL_NAME,
        device = "cpu"
    )
    
    print(pipe("I want to finish")[0]['generated_text'])
    
if __name__ == "__main__":
    gpt2_pipe()