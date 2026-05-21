import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig

def run_test():
    print("--- STEP 1: Verifying Local GPU Detection ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found! The pipeline is falling back to CPU.")

    # 1. Base Model Path
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = "/workspace/storage/test_output"

    print(f"--- STEP 2: Loading Tokenizer and Model ({model_id}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if cuda_available else torch.float32,
        device_map="auto" if cuda_available else None
    )

    print("--- STEP 3: Creating Mock Datasets for Testing ---")
    # Tiny, hardcoded conversational format data
    mock_data = {
        "messages": [
            [
                {"role": "user", "content": "Hello, can you help me test my local setup?"},
                {"role": "assistant", "content": "Yes, your local GPU and pipeline configurations are working correctly."}
            ],
            [
                {"role": "user", "content": "What language model are we testing right now?"},
                {"role": "assistant", "content": "We are currently running a test using Qwen 2.5 0.5B Instruct."}
            ],
            [
                {"role": "user", "content": "Confirm pipeline storage status."},
                {"role": "assistant", "content": "The storage volume path is correctly mounted to workspace."}
            ]
        ]
    }
    dataset = Dataset.from_dict(mock_data)

    print("--- STEP 4: Configuring SFTTrainer Parameters ---")
    # Low profiling for fast smoke testing
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=5,                       # Force exit after 5 steps
        per_device_train_batch_size=1,      # Lower memory overhead
        logging_steps=1,                   # Print logs immediately 
        save_strategy="no",                # Skip intermediate saves for speed
        learning_rate=2e-5,
        report_to="none",                  # Turn off external telemetry tracking
        max_seq_length=512
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("--- STEP 5: Executing Fine-Tuning Job Loop ---")
    trainer.train()

    print(f"--- STEP 6: Saving Weights to PVC Volume -> {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    print("SUCCESS: Test pipeline completed cleanly.")

if __name__ == "__main__":
    run_test()