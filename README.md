# LoRA Fine-tuning of TinyLlama-1.1B on Databricks Dolly-15k

## Overview
This project focuses on fine-tuning the **TinyLlama/TinyLlama-1.1B-Chat-v1.0** model using **Low-Rank Adaptation (LoRA)** on a subset of the **Databricks Dolly-15k dataset**. The goal is to efficiently adapt a smaller language model to follow instructions, leveraging parameter-efficient fine-tuning techniques to achieve specialized performance without extensive computational resources.

## Key Findings

🎯 **Efficient Fine-tuning with LoRA**
LoRA significantly reduces the number of trainable parameters, making fine-tuning large language models feasible on consumer-grade GPUs or with limited resources.

📊 **Instruction Following Capability**
By fine-tuning on the Dolly-15k dataset, the TinyLlama model learns to generate contextually relevant and instruction-aligned responses.

⚙️ **Model Merging for Deployment**
LoRA adapters can be merged back into the base model, creating a standalone, fine-tuned model ready for deployment without additional dependencies.

## Dataset Details

**Source**: [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) from the Hugging Face Datasets library.

**Description**: A dataset of 15,000 high-quality, human-generated instruction-following records covering a wide range of tasks such as brainstorming, classification, closed QA, open QA, and summarization.

**Variables Used (from original Dolly-15k)**:
- `instruction`: The prompt or instruction given to the model.
- `context`: Additional context for the instruction (optional).
- `response`: The desired output or response for the given instruction and context.
- *Transformed into a single `text` column for training.*

## Methodology

### Data Preprocessing
- Loaded a subset of 1000 examples from the `databricks/databricks-dolly-15k` dataset.
- Converted each example into a structured text format combining `instruction`, `context`, and `response` into a new `text` column (e.g., `### Instruction: ... ### Context: ... ### Response: ...`).
- Shuffled the dataset for training robustness.

### LoRA Configuration
- **Rank (r)**: `8` (controls the dimensionality of the update matrices).
- **LoRA Alpha (`lora_alpha`)**: `16` (scaling factor for LoRA updates).
- **Target Modules (`target_modules`)**: `["q_proj", "v_proj"]` (specifying which attention layers to inject LoRA into).
- **LoRA Dropout (`lora_dropout`)**: `0.05` (regularization to prevent overfitting).
- **Bias**: `"none"` (keeps bias terms frozen).
- **Task Type**: `"CAUSAL_LM"` (for next-word prediction tasks).

### Training (Supervised Fine-Tuning - SFT)
- Utilized the `trl` library's `SFTTrainer` for efficient training.
- **Batch Size**: `2` (`per_device_train_batch_size`).
- **Gradient Accumulation**: `4` (`gradient_accumulation_steps`).
- **Learning Rate**: `2e-4`.
- **Epochs**: `1` (`num_train_epochs`).
- **Precision**: `fp16=True` for faster training on supported hardware.
- Gradient checkpointing and input gradient requirement enabled for memory efficiency.

### Model Merging and Inference
- The trained LoRA adapter weights were merged with the original `TinyLlama/TinyLlama-1.1B-Chat-v1.0` base model.
- The merged model was saved locally as `merged_model_new`.
- Performed inference with a sample prompt to demonstrate the model's instruction-following capabilities.

## Technical Implementation

**Dependencies**:
- Python 3.8+
- `torch`
- `transformers`
- `datasets`
- `peft`
- `trl`

**Key Libraries Used**:
- **`torch`**: Core deep learning framework.
- **`transformers`**: For loading pre-trained models and tokenizers.
- **`datasets`**: For efficient dataset loading and processing.
- **`peft`**: For Low-Rank Adaptation (LoRA) implementation.
- **`trl`**: For Supervised Fine-Tuning (SFTTrainer).

## Future Work

### Model Enhancements
- Experiment with different LoRA ranks and target modules.
- Explore full fine-tuning versus LoRA for comparison.
- Evaluate on more diverse instruction-following benchmarks.

### Data Improvements
- Increase the size of the fine-tuning dataset.
- Incorporate domain-specific instruction data for specialized applications.

### Deployment Considerations
- Quantize the merged model for further optimization and reduced memory footprint.
- Develop a simple web interface for real-time inference.

## Files Structure

```
LoRA-SFT/
├── README.md                           # This project documentation
├── LoRA_Notebook.ipynb                      # Complete Jupyter notebook with analysis
├── data/
│   └── dolly_train.json                # Subset of Dolly-15k dataset
├── lora-adapter/                       # Saved LoRA adapter weights and config
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│   └── tokenizer_config.json
│   └── ...
└── merged_model_new/                   # Fully merged fine-tuned model
    ├── config.json
    └── model.safetensors
    └── ...
```

## Quick Start

1.  **Clone the repository** (if hosted on GitHub).
2.  **Install dependencies**:
    ```bash
    pip install torch transformers datasets peft trl
    ```
3.  **Open the Jupyter notebook** (e.g., `(https://colab.research.google.com/drive/1DE_UDqMqIgGkAIIFHhhOGOaoewInUKa7#scrollTo=6482d33b)`).
4.  **Run the cells sequentially** to execute the data preparation, LoRA configuration, training, model merging, and inference steps.

## Results Summary

**Fine-tuning Approach**: LoRA for parameter-efficient fine-tuning.
**Base Model**: TinyLlama-1.1B-Chat-v1.0.
**Dataset**: Databricks Dolly-15k (subset).
**Outcome**: A fine-tuned language model capable of generating responses to instructions, significantly improved through efficient LoRA adaptation.

## Contributor

**[Priyanka Jagadala]**
[https://www.linkedin.com/in/priyankajagadala/] 
