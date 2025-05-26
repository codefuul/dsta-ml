import os
import re
import json
import random
import torch
import torchaudio # Often used by datasets for audio loading
# import librosa # Not explicitly used, torchaudio/datasets handle resampling

from datasets import Dataset, DatasetDict, Audio
from evaluate import load # Standard library for metrics like WER
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# --- Configuration ---
AUDIO_DIR = "/home/jupyter/novice/asr"
MODEL_CHECKPOINT = "facebook/wav2vec2-base"
REPO_NAME = "wav2vec2-base-ft-novice-asr-from-jsonl"

if not os.path.isdir(AUDIO_DIR):
    print(f"ERROR: Audio directory not found: {AUDIO_DIR}")
    exit()
print(f"Using audio directory: {AUDIO_DIR}")

# --- 1. Load and Prepare Custom Dataset (from JSONL) ---
print("\n--- Loading and Preparing Custom Dataset from JSONL ---")
JSONL_FILE_PATH = os.path.join(AUDIO_DIR, "asr.jsonl")
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
if len(custom_dataset) > 1:
    if len(custom_dataset) * test_split_size < 1 and len(custom_dataset) > 1:
        print(f"Warning: Dataset size ({len(custom_dataset)}) very small. Using entire dataset for train/test.")
        split_datasets = DatasetDict({"train": custom_dataset, "test": custom_dataset})
    else:
        split_datasets = custom_dataset.train_test_split(test_size=test_split_size, seed=42, shuffle=True)
else:
    print("Warning: Only one data sample. Using for train/test.")
    split_datasets = DatasetDict({"train": custom_dataset, "test": custom_dataset})

print(f"Dataset loaded and split: {split_datasets}")
split_datasets = split_datasets.cast_column("audio", Audio(sampling_rate=16000))

# --- 2. Create Tokenizer ---
print("\n--- Creating Tokenizer ---")
def remove_special_characters(batch):
    chars_to_ignore_regex = r"[,\?\.\!;\:\"\“\%\‘\”\]-]"
    text_input = batch["text"]
    if isinstance(text_input, str):
        batch["text"] = re.sub(chars_to_ignore_regex, '', text_input).lower()
        batch["text"] = re.sub(r'\s+', ' ', batch["text"]).strip()
    else:
        print(f"Warning: Non-string text data: '{text_input}'. Replacing with empty string.")
        batch["text"] = ""
    return batch

split_datasets = split_datasets.map(remove_special_characters, num_proc=1)
print("Removed special characters and lowercased text.")

if "train" in split_datasets and len(split_datasets["train"]) > 0 and any(t for t in split_datasets["train"]["text"] if t):
    print("Example processed text:", random.choice([t for t in split_datasets["train"]["text"] if t]))
else:
    print("Warning: No training data or all text is empty after 'remove_special_characters'.")
    if "train" not in split_datasets or len(split_datasets["train"]) == 0:
        print("ERROR: Training dataset empty. Check JSONL and 'remove_special_characters'.")
        exit()
def extract_all_chars(batch):
    all_text = " ".join(text for text in batch["text"] if text)
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

train_column_names = split_datasets["train"].column_names if "train" in split_datasets else []
vocabs = split_datasets.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_column_names)

train_vocab_list = []
if "train" in vocabs:
    train_split_data = vocabs["train"]
    if "vocab" in train_split_data.column_names and len(train_split_data["vocab"]) > 0:
        train_vocab_list = train_split_data["vocab"][0]
test_vocab_list = []
if "test" in vocabs:
    test_split_data = vocabs["test"]
    if "vocab" in test_split_data.column_names and len(test_split_data["vocab"]) > 0:
        test_vocab_list = test_split_data["vocab"][0]

vocab_list = list(set(train_vocab_list) | set(test_vocab_list))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["|"] = vocab_dict.get(" ", len(vocab_dict))
if " " in vocab_dict and vocab_dict[" "] != vocab_dict["|"]:
     del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

if not vocab_list and ' ' not in vocab_dict and "|" not in vocab_dict :
    print("ERROR: Vocabulary effectively empty. Check JSONL and 'remove_special_characters'.")
    exit()
print(f"Vocabulary size: {len(vocab_dict)}")

os.makedirs("./", exist_ok=True)
with open('vocab.json', 'w', encoding='utf-8') as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
print("Tokenizer created.")

# --- 3. Create Feature Extractor ---
print("\n--- Creating Feature Extractor ---")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)

# --- 4. Create Processor ---
print("\n--- Creating Processor ---")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor_save_path = f"./{REPO_NAME}"
os.makedirs(processor_save_path, exist_ok=True)
processor.save_pretrained(processor_save_path)
print(f"Processor created and saved to ./{processor_save_path}/")

# --- 5. Preprocess Data ---
print("\n--- Preprocessing Data for Model ---")
def prepare_dataset(batch):
    audio_details = batch["audio"]
    text_for_tokenization = batch["text"] if batch["text"] is not None else ""

    # Process audio
    audio_processed = processor(audio=audio_details["array"], sampling_rate=audio_details["sampling_rate"])
    batch["input_values"] = audio_processed.input_values[0]
    # Store the length of the input_values, which is used by LengthGroupedSampler
    batch["input_length"] = len(batch["input_values"])

    # Process text to get labels
    labels_processed = processor(text=text_for_tokenization)
    batch["labels"] = labels_processed.input_ids
    
    # Add input_ids to batch
    batch["input_ids"] = batch["input_values"]  # Add this line
    return batch

processed_datasets = split_datasets.map(
    prepare_dataset, remove_columns=train_column_names, num_proc=1
)
print("Dataset preprocessed.")

if "train" not in processed_datasets or not processed_datasets["train"] or len(processed_datasets["train"]) == 0:
    print("ERROR: Training dataset empty after preprocessing.")
    exit()
print(f"Processed training dataset features: {processed_datasets['train'].features}")
if "test" in processed_datasets and processed_datasets["test"] and len(processed_datasets["test"]) > 0:
    print(f"Processed test dataset features: {processed_datasets['test'].features}")

# --- 6. Set-up Trainer ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch_data = self.processor.pad(
            input_features, padding=self.padding, max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels, return_tensors="pt",
            )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch_data["labels"] = labels
        return batch_data

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.from_numpy(pred_logits), axis=-1)
    label_ids_copy = pred.label_ids.copy()
    label_ids_copy[label_ids_copy == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids_copy, group_tokens=False, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

MODEL_LOCAL_PATH = os.path.join("src", "models", MODEL_CHECKPOINT.split('/')[-1])
os.makedirs(MODEL_LOCAL_PATH, exist_ok=True)
config_path = os.path.join(MODEL_LOCAL_PATH, "config.json")

if not os.path.exists(config_path):
    print(f"Downloading pretrained model '{MODEL_CHECKPOINT}' to {MODEL_LOCAL_PATH}...")
    temp_model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_CHECKPOINT, ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    temp_model.save_pretrained(MODEL_LOCAL_PATH)
    print("Pretrained model downloaded.")
    model = temp_model
else:
    print(f"Loading pretrained model from local path: {MODEL_LOCAL_PATH}.")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_LOCAL_PATH, ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    print("Pretrained model loaded.")

model.freeze_feature_extractor()

# MODIFIED: Added length_column_name
training_args = TrainingArguments(
    output_dir=f"./{REPO_NAME}",
    group_by_length=True,
    length_column_name="input_length", # Tells Trainer to use this column for LengthGroupedSampler
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    num_train_epochs=5, # INCREASE FOR REAL TRAINING
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=500,
    save_total_limit=2,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets.get("test") if "test" in processed_datasets and processed_datasets.get("test") and len(processed_datasets["test"]) > 0 else None,
    tokenizer=processor.tokenizer,
)

# --- 7. Training ---
print("\n--- Starting Training ---")
if "train" not in processed_datasets or not processed_datasets["train"] or len(processed_datasets["train"]) == 0:
    print("CRITICAL ERROR: No training data. Exiting.")
    exit()
if trainer.eval_dataset is None:
    print("Warning: No evaluation dataset. Metrics on a test/validation set will not be computed.")

try:
    print(f"Training with {len(processed_datasets['train'])} samples.")
    if trainer.eval_dataset: print(f"Evaluating with {len(trainer.eval_dataset)} samples.")
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\nCUDA OUT OF MEMORY! Reduce batch sizes.")
    else: raise e
    exit()
print("\n--- Training Finished ---")

# --- 8. Save Final Model and Processor ---
final_model_save_path = os.path.join(f"./{REPO_NAME}", "final_model")
os.makedirs(final_model_save_path, exist_ok=True)
trainer.save_model(final_model_save_path)
processor.save_pretrained(final_model_save_path)
print(f"Final model and processor saved to ./{final_model_save_path}/")

# --- 9. Evaluation ---
print("\n--- Evaluating Model on Original Test Set (for display) ---")
original_test_set = split_datasets.get("test")
if original_test_set and len(original_test_set) > 0:
    print("Evaluating on original test set...")
    eval_model = Wav2Vec2ForCTC.from_pretrained(final_model_save_path)
    eval_processor = Wav2Vec2Processor.from_pretrained(final_model_save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model.to(device); eval_model.eval()
    num_examples_to_show = min(5, len(original_test_set))
    print(f"\nShowing predictions for {num_examples_to_show} examples:")
    example_dataset_original_test = original_test_set.shuffle(seed=42).select(range(num_examples_to_show))
    all_preds, all_targets = [], []
    for example in example_dataset_original_test:
        audio_input_values = eval_processor(
            example["audio"]["array"], sampling_rate=example["audio"]["sampling_rate"], return_tensors="pt"
        ).input_values.to(device)
        with torch.no_grad(): logits = eval_model(audio_input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = eval_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        target_text = example["text"]
        print(f"Target    : {target_text}\nPrediction: {pred_str}\n")
        all_preds.append(pred_str); all_targets.append(target_text)
    manual_eval_wer = wer_metric.compute(predictions=all_preds, references=all_targets)
    print(f"WER on these {num_examples_to_show} examples: {manual_eval_wer:.4f}")
elif not (original_test_set and len(original_test_set) > 0):
     print("No test dataset from initial split for manual evaluation.")
else:
    print("Original test dataset for display evaluation empty/not created.")

print("\n--- Evaluating on processed test set (used during training): ---")
if trainer.eval_dataset and len(trainer.eval_dataset) > 0 :
    print("Running trainer.evaluate()...")
    results = trainer.evaluate()
    print(f"Test WER (from trainer.evaluate): {results.get('eval_wer', 'N/A'):.4f}")
else:
    print("No processed test set for trainer.evaluate().")
print("\nScript finished.")
