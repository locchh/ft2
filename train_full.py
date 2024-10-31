import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset

# Set up logging
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.2")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the training data file (CSV format).")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID from Hugging Face hub.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    return parser.parse_args()

# Callback for logging evaluation loss
class LogEvalLossCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        logger.info(f"Evaluation loss: {metrics['eval_loss']}")

def main():
    args = parse_args()
    global logger
    logger = setup_logging(args.output_dir)
    
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

    # Set up training arguments
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
        report_to="none",
        log_level="info",
        learning_rate=args.learning_rate,
        max_grad_norm=2,
        logging_dir=args.output_dir  # TensorBoard logs are also saved here if desired
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        callbacks=[LogEvalLossCallback()]  # Register the callback
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
