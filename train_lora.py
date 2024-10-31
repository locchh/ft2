import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.2 with LoRA and TensorBoard logging")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the training data file (CSV format).")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID from Hugging Face hub.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank of the LoRA adaptation matrices.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up GPU if available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Current GPU: {gpu_name}")
    else:
        logger.warning("No GPU available.")

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA adaptation applied to model.")

    # Load the dataset
    dataset = load_dataset("csv", data_files=args.data_file, split="train")

    # Apply the chat template function
    def apply_chat_template(example):
        messages = [
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": example['answer']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}
    
    dataset = dataset.map(apply_chat_template)
    dataset = dataset.train_test_split(test_size=0.05)
    
    # Tokenize the data
    def tokenize_function(example):
        tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=args.max_length)
        tokens['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']]
        return tokens

    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'prompt'])

    # Set up training arguments with TensorBoard
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=40,
        logging_steps=40,
        save_steps=150,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        fp16=False,
        report_to=["tensorboard"],  # Enable TensorBoard logging
        logging_dir=f"{args.output_dir}/logs",  # TensorBoard logs directory
        log_level="info",
        learning_rate=args.learning_rate,
        max_grad_norm=2
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer
    )

    # Train the model with LoRA
    trainer.train()
    
    # Save the model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
