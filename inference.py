import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch
# Check if a GPU is available
if torch.cuda.is_available():
    # Get the current device index (default is 0 if no other device is specified)
    current_device = torch.cuda.current_device()
    
    # Get the name of the GPU at this device index
    gpu_name = torch.cuda.get_device_name(current_device)
    print(f"Current GPU: {gpu_name}")
else:
    print("No GPU available.")


from transformers import pipeline

model_id = sys.argv[1] #"fine-tuned-model"
pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
)
messages = [
    {"role": "user", "content": "create the boolean syntax to find the accounting intern profiles on Linkedin through Google"},
]
outputs = pipe(
    messages,
    max_new_tokens=128
)
print(outputs[0]["generated_text"][-1]["content"])