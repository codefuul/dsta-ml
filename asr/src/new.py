import os
import torch
from pathlib import Path # For relative path handling
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# --- Configuration with Relative Paths ---

# Get the absolute path of the directory where this script (new.py) is located
# e.g., /home/jupyter/Nusurvivors/asr/src/
SCRIPT_DIR = Path(__file__).resolve().parent
print(f"Script directory (SCRIPT_DIR): {SCRIPT_DIR}")

# Get the parent directory of 'src', which should be 'asr'
# e.g., /home/jupyter/Nusurvivors/asr/
ASR_DIR = SCRIPT_DIR.parent
print(f"ASR project directory (ASR_DIR): {ASR_DIR}")

# Path to your dataset directory containing asr.jsonl and audio files
# IMPORTANT: This path is still absolute. If /novice/asr is not accessible for reading
# by the 'jupyter' user, you will get an error later when loading data.
# If 'novice/asr' is relative to your project (e.g., at the same level as 'Nusurvivors'
# or inside it), this path needs to be made relative too.
# Example: If 'novice' is at the same level as the 'Nusurvivors' directory:
#   NUSURVIVORS_DIR = ASR_DIR.parent # e.g., /home/jupyter/Nusurvivors/
#   PROJECT_ROOT = NUSURVIVORS_DIR.parent # e.g., /home/jupyter/
#   DATA_DIR = str(PROJECT_ROOT / "novice" / "asr")
DATA_DIR = "/home/jupyter/novice/asr"
print(f"Data directory (DATA_DIR): {DATA_DIR}")


# Path to save the fine-tuned model and training outputs
# This will be <ASR_DIR>/models/, e.g., /home/jupyter/Nusurvivors/asr/models/
MODEL_OUTPUT_DIR = str(ASR_DIR / "models")
print(f"Model output directory (MODEL_OUTPUT_DIR): {MODEL_OUTPUT_DIR}")

# Name of the pretrained Whisper model to fine-tune
PRETRAINED_MODEL_NAME = "openai/whisper-small"
# Language of your audio data
# CRITICAL: Update this to your dataset's language if not English.
LANGUAGE = "english"
TASK = "transcribe"

# Training/testing split ratio
TEST_SPLIT_SIZE = 0.10 # 10% for testing (450 samples out of 4500)

# Create model output directory if it doesn't exist
print(f"Attempting to create model output directory at: {MODEL_OUTPUT_DIR}")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) # This uses the relative path
print(f"Successfully ensured model output directory exists or was created.")

# --- 1. Load Your Custom Dataset from /novice/asr using asr.jsonl ---
print(f"\nSTEP 1: Attempting to load dataset from {DATA_DIR} using asr.jsonl...")

manifest_file_path = os.path.join(DATA_DIR, "asr.jsonl")

if not os.path.exists(manifest_file_path):
    print(f"  ERROR: Manifest file not found at {manifest_file_path}")
    print(f"  Please ensure 'asr.jsonl' exists in '{DATA_DIR}'.")
    exit()

try:
    raw_dataset_dict = load_dataset("json", data_files={"all_data": manifest_file_path})
    raw_dataset = raw_dataset_dict["all_data"]

    print(f"  Successfully loaded {manifest_file_path}. Number of entries: {len(raw_dataset)}")
    print(f"  Column names found in your asr.jsonl: {raw_dataset.column_names}")
    print("  First entry example from asr.jsonl (before any script processing):")
    print(f"    {raw_dataset[0]}")

    # Set the known column names based on your asr.jsonl file content
    actual_audio_col = "audio"      # Key for audio filenames in your asr.jsonl
    actual_text_col = "transcript"  # Key for transcription text in your asr.jsonl
        
    print(f"  Using '{actual_audio_col}' as the key for audio filenames.")
    print(f"  Using '{actual_text_col}' as the key for transcriptions.")

    if actual_audio_col not in raw_dataset.column_names:
        print(f"  ERROR: The specified audio column '{actual_audio_col}' was not found in asr.jsonl.")
        print(f"  Available columns: {raw_dataset.column_names}. Please check your asr.jsonl structure.")
        exit()
    if actual_text_col not in raw_dataset.column_names:
        print(f"  ERROR: The specified text column '{actual_text_col}' was not found in asr.jsonl.")
        print(f"  Available columns: {raw_dataset.column_names}. Please check your asr.jsonl structure.")
        exit()

    def process_dataset_entry(batch):
        audio_paths = [os.path.join(DATA_DIR, str(fname)) for fname in batch[actual_audio_col]]
        batch["audio_full_path"] = audio_paths
        batch["sentence_text"] = batch[actual_text_col]
        return batch
    
    raw_dataset = raw_dataset.map(
        process_dataset_entry,
        batched=True
    )
    
    raw_dataset = raw_dataset.select_columns(["audio_full_path", "sentence_text"])
    raw_dataset = raw_dataset.rename_column("audio_full_path", "audio")
    raw_dataset = raw_dataset.rename_column("sentence_text", "sentence")

    if "audio" not in raw_dataset.column_names or "sentence" not in raw_dataset.column_names:
        print("  ERROR: Dataset processing failed. Standardized 'audio' or 'sentence' column is missing after map and rename.")
        print(f"  Current columns after processing: {raw_dataset.column_names}")
        exit()
    
    print("\n  Dataset processed. 'audio' column now contains full paths, 'sentence' contains transcriptions.")
    print("  First entry after this initial processing:")
    print(f"    {raw_dataset[0]}")
    print("STEP 1 COMPLETE.\n")

except Exception as e:
    print(f"  Could not load or process dataset from {manifest_file_path}. Error: {e}")
    print("  Please check the structure of your 'asr.jsonl' file and the column name assignments in STEP 1 of the script.")
    exit()

# --- 2. Split Dataset into Training and Testing (90/10) ---
print("STEP 2: Splitting dataset into training and testing sets...")
split_dataset = raw_dataset.train_test_split(test_size=TEST_SPLIT_SIZE, shuffle=True, seed=42)
common_voice = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})
print(f"  Dataset split: {len(common_voice['train'])} train samples, {len(common_voice['test'])} test samples.")
print("STEP 2 COMPLETE.\n")

# --- 3. Prepare Feature Extractor, Tokenizer, and Processor ---
print("STEP 3: Preparing feature extractor, tokenizer, and processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL_NAME, language=LANGUAGE, task=TASK)
print("STEP 3 COMPLETE.\n")

# --- 4. Pre-Process Data (Load audio, resample, extract features, tokenize text) ---
print("STEP 4: Pre-processing data...")
try:
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
    print("  Audio column cast and resampling to 16kHz initiated.")
    print("  First train sample after casting 'audio' column:")
    print(f"    {common_voice['train'][0]}")

except Exception as e:
    print(f"  Error casting 'audio' column during cast_column: {e}")
    exit()

def prepare_dataset(batch):
    audio_arrays = [item["array"] for item in batch["audio"]]
    sampling_rates = [item["sampling_rate"] for item in batch["audio"]]
    batch["input_features"] = [processor.feature_extractor(arr, sampling_rate=sr).input_features[0] for arr, sr in zip(audio_arrays, sampling_rates)]
    batch["labels"] = [processor.tokenizer(text).input_ids for text in batch["sentence"]]
    return batch

common_voice = common_voice.map(
    prepare_dataset, 
    batched=True,
    batch_size=100, 
    remove_columns=common_voice.column_names["train"]
)

print("\n  Data pre-processing complete.")
print("  First train sample after full pre-processing:")
print(f"    {common_voice['train'][0]}")
print("STEP 4 COMPLETE.\n")

# --- 5. Define Data Collator ---
print("STEP 5: Defining data collator...")
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features_list = [{"input_features": feature["input_features"]} for feature in features]
        batch_input_features = self.processor.feature_extractor.pad(input_features_list, return_tensors="pt")
        label_features_list = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features_list, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch_input_features["labels"] = labels
        return batch_input_features
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("STEP 5 COMPLETE.\n")

# --- 6. Evaluation Metrics ---
print("STEP 6: Defining evaluation metrics (WER)...")
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
print("STEP 6 COMPLETE.\n")

# --- 7. Load Pre-trained Model ---
print(f"STEP 7: Loading pre-trained model '{PRETRAINED_MODEL_NAME}'...")
model = WhisperForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
print("STEP 7 COMPLETE.\n")

# --- 8. Define Training Arguments ---
print("STEP 8: Defining training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUTPUT_DIR, # Uses the relative path defined earlier
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_strategy="steps",
    save_steps=250,
    eval_steps=250,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=2,
)
print("STEP 8 COMPLETE.\n")

# --- 9. Initialize Trainer ---
print("STEP 9: Initializing Trainer...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print("STEP 9 COMPLETE.\n")

# --- 10. Train the Model ---
print("STEP 10: Starting training...")
try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n  CRITICAL ERROR: CUDA out of memory.")
        # ... (error message as before) ...
    else:
        raise e
    exit()
print("STEP 10 COMPLETE.\n")

# # --- 11. Save the final best model and processor ---
# print("STEP 11: Saving the final best model and processor...")
# final_best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_fine_tuned_model")
# trainer.save_model(final_best_model_path)
# processor.save_pretrained(final_best_model_path)
# print(f"  The best fine-tuned model and processor have been saved to: {final_best_model_path}")
# print("STEP 11 COMPLETE.\n")
# print("Script finished successfully!")
# CHOOSE A CONSISTENT MODEL DIRECTORY NAME

FINE_TUNED_MODEL_NAME = "my_whisper_ft_novice_asr" 
final_model_save_path = os.path.join(MODEL_OUTPUT_DIR, FINE_TUNED_MODEL_NAME) # Changed from "best_fine_tuned_model"

# Ensure you are saving to MODEL_OUTPUT_DIR, which should be .../Nusurvivors/asr/models/
# So the full path will be .../Nusurvivors/asr/models/my_whisper_ft_novice_asr

trainer.save_model(final_model_save_path)
processor.save_pretrained(final_model_save_path)
print(f"  The fine-tuned model and processor have been saved to: {final_model_save_path}")
print("STEP 11 COMPLETE.\n")