import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
from helper import calculate_metrics
import argparse

def main(data_file_path, model_id="fine-tuned-model", cuda_device="0", max_length=128):
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # Check if a GPU is available
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"Current GPU: {gpu_name}")
    else:
        print("No GPU available.")

    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
    )

    # Load dataset
    dataset = load_dataset("csv", data_files=data_file_path, split="train")

    reference_sentences = []
    candidate_sentences = []

    # Process each example in the dataset
    for example in tqdm(dataset):
        messages = [
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": example['answer']}
        ]

        outputs = pipe(messages, max_length=max_length)
        assistant_answer = outputs[0]["generated_text"][-1]
        answer = assistant_answer["content"]

        reference_sentences.append(example['answer'])
        candidate_sentences.append(answer)

    # Calculate metrics
    metrics = calculate_metrics(reference_sentences, candidate_sentences)
    print(metrics)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run text generation pipeline with specified dataset and model")
    parser.add_argument("--data_file_path", type=str, required=True, help="Path to the dataset file (CSV)")
    parser.add_argument("--model_id", type=str, default="fine-tuned-model", help="Model ID for the text generation pipeline")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device index to use")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the generated text")

    args = parser.parse_args()

    main(
        data_file_path=args.data_file_path,
        model_id=args.model_id,
        cuda_device=args.cuda_device,
        max_length=args.max_length
    )
