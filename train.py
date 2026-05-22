import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

def run_production_training():
    print("--- STEP 1: Verifying Local GPU Detection ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
    # --- PRODUCTION PATH MANAGEMENT ---
    # Instead of hardcoding paths, we read from our mounted PVC directories
    MODEL_BASE_DIR = "/models"
    DATASET_BASE_DIR = "/datasets"
    CHECKPOINT_BASE_DIR = "/checkpoints"

    # If your worker node is completely offline, change this model ID string 
    # to the exact folder path inside your PVC: f"{MODEL_BASE_DIR}/Qwen2.5-0.5B-Instruct"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    
    output_dir = os.path.join(CHECKPOINT_BASE_DIR, "outputs")
    tensorboard_log_dir = os.path.join(CHECKPOINT_BASE_DIR, "tensorboard_logs")

    print(f"--- STEP 2: Loading Tokenizer and Model ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if cuda_available else torch.float32,
        device_map="auto" if cuda_available else None
    )

    print("--- STEP 3: Preparing Dataset Context ---")
    # In production, change this dummy data loader to read your real files out of the dataset volume:
    # dataset = load_from_disk(os.path.join(DATASET_BASE_DIR, "tokenized_train_data"))
    mock_data = {"messages": [[
        {"role": "user", "content": "Hello, can you help me test my local setup?"},
        {"role": "assistant", "content": "Yes, your local GPU and pipeline configurations are working correctly."}
    ]]}
    dataset = Dataset.from_dict(mock_data)

    print("--- STEP 4: Setting Up Industry-Standard SFTConfig ---")
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=10,                      # Low limit for testing execution
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,     # Standard stability approach
        logging_steps=1,                   # High frequency feedback for smoke test
        save_strategy="steps",
        save_steps=5,                      # Verifies that JuiceFS can save weight blocks mid-run
        save_total_limit=2,                # Keeps your disk clean by purging old checkpoints
        
        # --- TensorBoard Configuration ---
        logging_dir=tensorboard_log_dir,
        report_to=["tensorboard"],
        
        # --- Hardware Optimization ---
        bf16=cuda_available,               # Use bfloat16 to optimize memory execution on modern GPUs
        max_seq_length=512,
        dataset_kwargs={"append_concat_token": False} # Prevents formatting warning alerts in Qwen architectures
    )

    print("--- STEP 5: Initializing Pipeline Trainer ---")
    trainer = SFTTrainer(
        model=model, 
        train_dataset=dataset, 
        args=training_args, 
        processing_class=tokenizer  # Modern parameter name replacing deprecated 'tokenizer' keyword
    )
    
    print("--- STEP 6: Commencing Training Pass ---")
    trainer.train()
    
    print(f"SUCCESS: Test complete. Weight matrix snapshots stored to: {output_dir}")

if __name__ == "__main__":
    run_production_training()