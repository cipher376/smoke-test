#!/usr/bin/env python
"""
Optimized SFT Training Script for GTX 1660 Ti (6GB VRAM)
With improved error handling, logging, and memory optimization
"""

import argparse
import torch
import os
import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments with validation"""
    parser = argparse.ArgumentParser(
        description="1660Ti Optimized SFT Training with 4-bit Quantization and LoRA"
    )
    
    # Model arguments
    parser.add_argument("--model_dir", type=str, default="microsoft/phi-2",
                        help="Pretrained model ID or path")
    parser.add_argument("--dataset_dir", type=str, default="/datasets",
                        help="Directory containing train.json")
    parser.add_argument("--output_dir", type=str, default="/checkpoints",
                        help="Directory to save model checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per device training batch size (1-2 for 6GB)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Maximum sequence length (lower = less VRAM)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=5,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    
    # Debug and testing
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with sample data")
    parser.add_argument("--test", action="store_true",
                        help="Run quick test instead of full training")
    
    return parser.parse_args()

def setup_quantization_config():
    """Setup 4-bit quantization for GTX 1660 Ti (6GB VRAM)"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Extra compression
    )

def setup_lora_config(args):
    """Setup LoRA configuration"""
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # More comprehensive
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_dataset_with_fallback(dataset_path: str, debug: bool = False) -> Optional[Dataset]:
    """Load dataset with fallback to dummy data for testing"""
    try:
        train_file = Path(dataset_path) / "train.json"
        
        if debug or not train_file.exists():
            logger.warning(f"Dataset not found at {train_file}, creating dummy dataset")
            return create_dummy_dataset()
        
        logger.info(f"Loading dataset from {train_file}")
        dataset = load_dataset("json", data_files=str(train_file), split="train")
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        if debug:
            logger.info("Debug mode: Creating dummy dataset")
            return create_dummy_dataset()
        raise

def create_dummy_dataset(num_samples: int = 100) -> Dataset:
    """Create dummy dataset for testing"""
    data = {
        "text": [
            f"Sample training text for fine-tuning. This is example {i} of {num_samples}. "
            "The model should learn to generate similar text patterns." 
            for i in range(num_samples)
        ]
    }
    return Dataset.from_dict(data)

def setup_training_args(args) -> SFTConfig:
    """Setup training arguments optimized for 6GB VRAM"""
    return SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1 if args.test else args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=True,  # GTX 1660 Ti supports fp16
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",  # Disable external logging
        remove_unused_columns=False,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # Save memory
        optim="paged_adamw_8bit",  # 8-bit optimizer for memory savings
        max_grad_norm=0.3,
    )

def log_system_info():
    """Log system information for debugging"""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Log memory usage
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

def save_training_config(args, output_dir: str):
    """Save training configuration for reproducibility"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training config to {config_path}")

def run_training():
    """Main training function with comprehensive error handling"""
    args = parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Log system info
        log_system_info()
        logger.info(f"=== Starting Training ===")
        logger.info(f"Model: {args.model_dir}")
        logger.info(f"Output: {args.output_dir}")
        
        # 1. Setup quantization
        logger.info("Step 1: Setting up 4-bit quantization...")
        bnb_config = setup_quantization_config()
        
        # 2. Load tokenizer
        logger.info("Step 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, 
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # 3. Load model with quantization
        logger.info("Step 3: Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Enable gradient checkpointing for memory savings
        model.gradient_checkpointing_enable()
        
        # 4. Setup LoRA
        logger.info("Step 4: Configuring LoRA...")
        peft_config = setup_lora_config(args)
        
        # 5. Load dataset
        logger.info("Step 5: Loading dataset...")
        dataset = load_dataset_with_fallback(args.dataset_dir, args.debug)
        
        # Log dataset info
        if dataset:
            logger.info(f"Dataset size: {len(dataset)} samples")
            logger.info(f"Dataset features: {dataset.column_names}")
            
            # Show sample if debug
            if args.debug:
                sample = dataset[0]
                logger.info(f"Sample text: {sample.get('text', 'N/A')[:200]}...")
        
        # 6. Setup training arguments
        logger.info("Step 6: Configuring training arguments...")
        training_args = setup_training_args(args)
        
        # 7. Initialize trainer
        logger.info("Step 7: Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            packing=False,  # Disable packing for simpler memory management
        )
        
        # Save configuration for reproducibility
        save_training_config(args, args.output_dir)
        
        # 8. Start training
        logger.info("Step 8: Starting training loop...")
        trainer.train()
        
        # 9. Save final model
        logger.info("Step 9: Saving final model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # 10. Log final statistics
        logger.info("=== Training Complete ===")
        logger.info(f"Model saved to: {args.output_dir}")
        
        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_training()