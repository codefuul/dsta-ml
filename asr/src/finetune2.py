import os
import re
import json
import random
import torch
import torchaudio # Often used by datasets for audio loading

from datasets import Dataset, DatasetDict, Audio
from evaluate import load as load_metric # Standard library for metrics like WER
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, # Using Seq2Seq specific arguments
    Seq2SeqTrainer,          # Using Seq2Seq specific trainer
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# --- Configuration ---
AUDIO_DIR = "/home/jupyter/novice/asr"  # CHANGE THIS TO YOUR AUDIO DIRECTORY
MODEL_CHECKPOINT = "openai/whisper-base"  # You can choose other Whisper models like "openai/whisper-small", "openai/whisper-tiny"
REPO_NAME = f"{MODEL_CHECKPOINT.split('/')[-1]}-ft-novice-asr-from-jsonl"
LANGUAGE = "english"  # Set your target language (e.g., "english", "spanish", etc.)
TASK = "transcribe"    # Task can be "transcribe" or "translate"

if not os.path.isdir(AUDIO_DIR):
    print(f"ERROR: Audio directory not found: {AUDIO_DIR}")
    exit()
print(f"Using audio directory: {AUDIO_DIR}")

# --- 1. Load and Prepare Custom Dataset (from JSONL) ---
print("\n--- Loading and Preparing Custom Dataset from JSONL ---")
JSONL_FILE_PATH = os.path.join(AUDIO_DIR, "asr.jsonl") # Assumes your jsonl file is named asr.jsonl
audio_file_paths = []
transcriptions = []

if not os.path.exists(JSONL_FILE_PATH):
    print(f"ERROR: JSONL file not found: {JSONL_FILE_PATH}")
    exit()

print(f"Loading data from {JSONL_FILE_PATH}...")
with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
    for line_number, line in enumerate(f, 1):
        try:
            data_entry = json.loads(line.strip())
            audio_filename = data_entry.get("audio")
            transcript_text = data_entry.get("transcript")
            if audio_filename and transcript_text is not None:
                full_audio_path = os.path.join(AUDIO_DIR, audio_filename)
                if os.path.exists(full_audio_path):
                    audio_file_paths.append(full_audio_path)
                    transcriptions.append(transcript_text)
                else:
                    print(f"Warning (Line {line_number}): Audio file '{full_audio_path}' not found. Skipping.")
            else:
                print(f"Warning (Line {line_number}): Skipping line due to missing 'audio' or 'transcript'.")
        except json.JSONDecodeError:
            print(f"Warning (Line {line_number}): Skipping malformed JSON line.")

if not audio_file_paths:
    print(f"ERROR: No valid audio file paths loaded from '{JSONL_FILE_PATH}'.")
    exit()
print(f"Loaded {len(audio_file_paths)} audio files and {len(transcriptions)} transcriptions.")

data_dict = {"audio": audio_file_paths, "text": transcriptions}
custom_dataset = Dataset.from_dict(data_dict)

test_split_size = 0.1
if len(custom_dataset) == 0:
    print("ERROR: Dataset is empty after loading.")
    exit()
elif len(custom_dataset) == 1:
    print("Warning: Only one data sample. Using for train/test.")
    split_datasets = DatasetDict({"train": custom_dataset, "test": custom_dataset})
else:
    if len(custom_dataset) * test_split_size < 1:
        print(f"Warning: Dataset size ({len(custom_dataset)}) very small. Using entire dataset for train/test.")
        split_datasets = DatasetDict({"train": custom_dataset, "test": custom_dataset})
    else:
        split_datasets = custom_dataset.train_test_split(test_size=test_split_size, seed=42, shuffle=True)


print(f"Dataset loaded and split: {split_datasets}")
# Whisper models expect audio sampled at 16kHz
split_datasets = split_datasets.cast_column("audio", Audio(sampling_rate=16000))

# --- 2. Initialize Feature Extractor, Tokenizer, and Processor ---
print("\n--- Initializing Feature Extractor, Tokenizer, and Processor ---")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_CHECKPOINT, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_CHECKPOINT, language=LANGUAGE, task=TASK)

# Optional: Text normalization (Whisper's tokenizer also performs normalization)
def remove_special_characters(batch):
    # This regex is from your original script.
    # Consider if Whisper's built-in normalization is sufficient.
    chars_to_ignore_regex = r"[,\?\.\!;\:\"\“\%\‘\”\]-]"
    text_input = batch["text"]
    if isinstance(text_input, str):
        batch["text"] = re.sub(chars_to_ignore_regex, '', text_input).lower()
        batch["text"] = re.sub(r'\s+', ' ', batch["text"]).strip()
    else:
        # print(f"Warning: Non-string text data: '{text_input}'. Replacing with empty string.")
        batch["text"] = "" # Or handle as an error
    return batch

print("Applying text normalization (if any defined beyond tokenizer's default)...")
split_datasets = split_datasets.map(remove_special_characters, num_proc=1) # num_proc can be increased
print("Text normalization applied.")


# --- 3. Preprocess Data ---
print("\n--- Preprocessing Data for Model ---")
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # Store the length of the input_features, which is used by group_by_length
    batch["input_length"] = len(batch["input_features"])

    # encode target text to label ids
    text_for_tokenization = batch["text"] if batch["text"] is not None else ""
    batch["labels"] = tokenizer(text_for_tokenization).input_ids
    return batch

train_column_names = split_datasets["train"].column_names if "train" in split_datasets else []
processed_datasets = split_datasets.map(
    prepare_dataset, remove_columns=train_column_names, num_proc=1 # num_proc can be increased
)
print("Dataset preprocessed.")

if "train" not in processed_datasets or not processed_datasets["train"] or len(processed_datasets["train"]) == 0:
    print("ERROR: Training dataset empty after preprocessing.")
    exit()
print(f"Processed training dataset features: {processed_datasets['train'].features}")
if "test" in processed_datasets and processed_datasets["test"] and len(processed_datasets["test"]) > 0:
    print(f"Processed test dataset features: {processed_datasets['test'].features}")


# --- 4. Define Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. Define Evaluation Metrics ---
print("\n--- Setting up Evaluation Metrics ---")
wer_metric = load_metric("wer")
# cer_metric = load_metric("cer") # Optional

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # cer = cer_metric.compute(predictions=pred_str, references=label_str) # Optional
    # return {"wer": wer, "cer": cer}
    return {"wer": wer}

# --- 6. Load Pretrained Model ---
print("\n--- Loading Pretrained Model ---")
MODEL_LOCAL_PATH = os.path.join("src", "models", MODEL_CHECKPOINT.split('/')[-1]) # Local cache path
os.makedirs(MODEL_LOCAL_PATH, exist_ok=True)
config_path = os.path.join(MODEL_LOCAL_PATH, "config.json")

if not os.path.exists(config_path):
    print(f"Downloading pretrained model '{MODEL_CHECKPOINT}' to {MODEL_LOCAL_PATH}...")
    temp_model = WhisperForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
    temp_model.save_pretrained(MODEL_LOCAL_PATH)
    processor.save_pretrained(MODEL_LOCAL_PATH) # Save processor alongside for completeness
    print("Pretrained model and processor downloaded and saved locally.")
    model = temp_model
else:
    print(f"Loading pretrained model from local path: {MODEL_LOCAL_PATH}.")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_LOCAL_PATH)
    # Ensure processor is also loaded if needed, though we initialized it earlier
    # processor = WhisperProcessor.from_pretrained(MODEL_LOCAL_PATH) # Usually not needed again here
    print("Pretrained model loaded.")

# Optional: Freeze parts of the model
# For Whisper, you might freeze the encoder to fine-tune only the decoder and cross-attention.
# model.freeze_encoder()

# Configure model for training
if hasattr(model, 'config') and hasattr(model.config, 'forced_decoder_ids'):
    model.config.forced_decoder_ids = None # Recommended for fine-tuning single language/task
if hasattr(model, 'config') and hasattr(model.config, 'suppress_tokens'):
    model.config.suppress_tokens = []


# --- 7. Define Training Arguments ---
print("\n--- Defining Training Arguments ---")
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{REPO_NAME}",
    per_device_train_batch_size=8,  # Adjust based on your GPU memory (e.g., 4, 8, 16)
    gradient_accumulation_steps=1,  # Increase for larger effective batch size if memory is an issue (e.g., 2, 4)
    learning_rate=1e-5,             # Common learning rate for Whisper fine-tuning
    warmup_steps=500,               # Number of steps for learning rate warmup
    # max_steps=4000,               # Total number of training steps. Overrides num_train_epochs.
    num_train_epochs=5,             # Number of training epochs (adjust for your dataset size and desired training time)
    # eval_strategy="epoch",        # Evaluate at the end of each epoch
    eval_strategy="steps",          # Evaluate at certain step intervals
    eval_steps=500,                # Evaluation frequency (if eval_strategy="steps")
    save_steps=500,                 # Model checkpoint saving frequency
    logging_steps=50,               # Logging frequency
    group_by_length=True,           # Groups samples of similar audio length for efficiency
    length_column_name="input_length",# Name of the column containing input lengths for group_by_length
    fp16=torch.cuda.is_available(), # Enable mixed precision training if a GPU is available
    gradient_checkpointing=True,    # Use gradient checkpointing to save memory (at cost of some speed)
    predict_with_generate=True,     # Crucial for Seq2Seq models to use .generate() in evaluation
    generation_max_length=225,      # Max length for generated sequences during evaluation
    report_to=["tensorboard"],      # Logging destination
    load_best_model_at_end=True,    # Load the best model found during training at the end
    metric_for_best_model="wer",    # Metric to determine the "best" model
    greater_is_better=False,        # For WER, lower is better
    push_to_hub=False,              # Set to True if you want to push to Hugging Face Hub
    save_total_limit=2,             # Limit the total number of checkpoints saved
)

# --- 8. Initialize Trainer ---
print("\n--- Initializing Trainer ---")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets.get("test") if "test" in processed_datasets and processed_datasets.get("test") and len(processed_datasets["test"]) > 0 else None,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor, # For padding input features, tokenizer part handled by processor in collator/metrics
)

# --- 9. Training ---
print("\n--- Starting Training ---")
if "train" not in processed_datasets or not processed_datasets["train"] or len(processed_datasets["train"]) == 0:
    print("CRITICAL ERROR: No training data. Exiting.")
    exit()
if trainer.eval_dataset is None:
    print("Warning: No evaluation dataset. Metrics on a test/validation set will not be computed by the trainer.")

try:
    print(f"Training with {len(processed_datasets['train'])} samples.")
    if trainer.eval_dataset: print(f"Evaluating with {len(trainer.eval_dataset)} samples.")
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\nCUDA OUT OF MEMORY! Consider reducing batch_size, enabling gradient_accumulation_steps, or using a smaller model.")
    else: raise e
    exit()
print("\n--- Training Finished ---")

# --- 10. Save Final Model and Processor ---
final_model_save_path = os.path.join(f"./{REPO_NAME}", "final_model")
os.makedirs(final_model_save_path, exist_ok=True)
trainer.save_model(final_model_save_path)
processor.save_pretrained(final_model_save_path)
print(f"Final model and processor saved to {final_model_save_path}/")

# --- 11. Evaluation (Manual Display of Examples) ---
print("\n--- Evaluating Model on Original Test Set (for display) ---")
original_test_set = split_datasets.get("test")
if original_test_set and len(original_test_set) > 0:
    print("Evaluating on original test set...")
    eval_model = WhisperForConditionalGeneration.from_pretrained(final_model_save_path)
    eval_processor = WhisperProcessor.from_pretrained(final_model_save_path) # This loads language/task settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model.to(device)
    eval_model.eval()

    num_examples_to_show = min(5, len(original_test_set))
    print(f"\nShowing predictions for {num_examples_to_show} examples:")
    example_dataset_original_test = original_test_set.shuffle(seed=42).select(range(num_examples_to_show))
    all_preds, all_targets = [], []

    for example in example_dataset_original_test:
        audio_input = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        
        input_features = eval_processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
        
        with torch.no_grad():
            # Use generate method. It will use the processor's language/task settings by default
            # or you can pass forced_decoder_ids if needed for specific control.
            # forced_decoder_ids = eval_processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK, no_timestamps=True)
            # predicted_ids = eval_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            predicted_ids = eval_model.generate(input_features) # Simpler, relies on config

        pred_str = eval_processor.batch_decode(predicted_ids, skip_special_tokens=True, group_tokens=False)[0]
        target_text = example["text"] # This is the text after your custom normalization
        
        print(f"Target    : {target_text}\nPrediction: {pred_str}\n")
        all_preds.append(pred_str)
        all_targets.append(target_text)

    if all_preds: # Ensure list is not empty
        manual_eval_wer = wer_metric.compute(predictions=all_preds, references=all_targets)
        print(f"WER on these {num_examples_to_show} examples: {manual_eval_wer:.4f}")
elif not (original_test_set and len(original_test_set) > 0):
     print("No test dataset from initial split for manual evaluation.")
else:
    print("Original test dataset for display evaluation empty/not created.")

print("\n--- Evaluating on processed test set (used during training by Trainer): ---")
if trainer.eval_dataset and len(trainer.eval_dataset) > 0 :
    print("Running trainer.evaluate()...")
    results = trainer.evaluate() # This uses the processed test set and compute_metrics
    print(f"Test WER (from trainer.evaluate on processed data): {results.get('eval_wer', 'N/A'):.4f}")
else:
    print("No processed test set for trainer.evaluate().")

print("\nScript finished.")

