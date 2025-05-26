# root/Nusurvivors/asr/src/makemodel.py

import os
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset, Audio, DatasetDict, load_dataset
import evaluate # For WER/CER metrics
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    DataCollatorSpeechSeq2SeqWithPadding, # Ideal collator for Whisper
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
# import torchaudio # Not directly used in this script's main logic but a core dependency for transformers.speech
from tqdm import tqdm
import random
import logging

# --- Configuration ---
# USER ACTION REQUIRED: Verify and set these paths and parameters
PRETRAINED_MODEL_ID = "openai/whisper-base" # You can choose "whisper-tiny", "whisper-small", etc.
LANGUAGE = "english" # Target language for fine-tuning
TASK = "transcribe"  # Task for the Whisper model
TARGET_SAMPLE_RATE = 16000 # Whisper models expect 16kHz audio
RANDOM_SEED = 42

# --- Path Definitions ---
# These paths assume the script is in root/Nusurvivors/asr/src/
SCRIPT_DIR = Path(__file__).resolve().parent
NUSURVIVORS_PROJECT_DIR = SCRIPT_DIR.parent.parent # Goes up to root/Nusurvivors/
COMMON_PARENT_DIR = NUSURVIVORS_PROJECT_DIR.parent   # Goes up to root/

# USER ACTION REQUIRED: Verify these paths point to your data
# Path to your JSON Lines manifest file (e.g., asr.jsonl)
FINETUNE_DATA_MANIFEST_FILE = COMMON_PARENT_DIR / "novice" / "asr" / "asr.jsonl"

# USER ACTION REQUIRED: Base directory where audio files are stored.
# Paths in your asr.jsonl should be relative to this directory, or absolute.
AUDIO_FILES_BASE_DIR = COMMON_PARENT_DIR / "novice" / "asr"

# Output directory for the fine-tuned model and checkpoints
MODEL_NAME_SUFFIX = PRETRAINED_MODEL_ID.split('/')[-1]
MODEL_OUTPUT_DIR = SCRIPT_DIR / f"whisper-{MODEL_NAME_SUFFIX}-ft-{LANGUAGE}-custom-final"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def load_and_prepare_data(manifest_path: Path, audio_base_dir: Path) -> Dataset:
    """
    Loads data from a JSON Lines manifest, verifies audio paths, and creates a Hugging Face Dataset.
    """
    logger.info(f"Attempting to load data from manifest: {manifest_path}")
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    data = []
    problematic_entries = 0
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                audio_path_str = record.get("audio") or record.get("audio_filepath")
                transcript_str = record.get("transcript") or record.get("text")

                if not audio_path_str or not transcript_str:
                    logger.warning(f"Skipping line {line_num} in manifest: missing audio or transcript. Record: {record}")
                    problematic_entries += 1
                    continue

                audio_file_manifest_path = Path(audio_path_str)
                if audio_file_manifest_path.is_absolute():
                    actual_audio_file_path = audio_file_manifest_path
                else:
                    actual_audio_file_path = audio_base_dir / audio_file_manifest_path
                
                # Additional check if the base dir itself was part of a relative path
                if not actual_audio_file_path.exists() and \
                   not audio_file_manifest_path.is_absolute() and \
                   len(audio_file_manifest_path.parts) > 0 and \
                   audio_base_dir.name == audio_file_manifest_path.parts[0]:
                     alt_audio_path = audio_base_dir.parent / audio_file_manifest_path
                     if alt_audio_path.exists():
                         actual_audio_file_path = alt_audio_path

                if not actual_audio_file_path.exists():
                    logger.warning(f"Audio file not found for record on line {line_num}: {actual_audio_file_path} (from manifest entry: {audio_path_str})")
                    problematic_entries +=1
                    continue
                
                data.append({"audio": str(actual_audio_file_path), "transcript": str(transcript_str).strip()})

            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON on line {line_num} in manifest: {line.strip()}")
                problematic_entries += 1
            except Exception as e:
                logger.warning(f"Skipping line {line_num} due to unexpected error: {e}. Record: {line.strip()}")
                problematic_entries += 1
    
    if problematic_entries > 0:
        logger.warning(f"Encountered {problematic_entries} problematic entries during manifest loading.")

    if not data:
        logger.error("No valid data loaded from the manifest.")
        raise ValueError("No valid data loaded from the manifest.")

    logger.info(f"Successfully loaded {len(data)} valid records from manifest.")
    
    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pd.DataFrame(data))
    # Cast "audio" column to Audio feature, resampling to TARGET_SAMPLE_RATE
    try:
        hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    except Exception as e:
        logger.error(f"Error casting 'audio' column. Ensure audio files are valid and accessible. Error: {e}")
        logger.error("Common issues: non-audio files listed, corrupted files, or permission problems.")
        logger.error(f"Example audio path being processed: {data[0]['audio'] if data else 'N/A'}")
        raise
        
    return hf_dataset

def preprocess_function_for_whisper(batch, processor: WhisperProcessor):
    """
    Preprocesses a batch of data for Whisper model input.
    Assumes `DataCollatorSpeechSeq2SeqWithPadding` is used, so no dummy `input_ids` are needed here.
    """
    # The 'audio' column now contains dicts like {'path': ..., 'array': ..., 'sampling_rate': ...}
    # We need to extract the audio arrays.
    audio_arrays = [item["array"] for item in batch["audio"]]
    transcripts = batch["transcript"]

    # 1. Extract audio features
    inputs = processor.feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SAMPLE_RATE, # Ensure consistency
        return_attention_mask=False       # Whisper feature_extractor doesn't typically need separate attention_mask
    )
    batch["input_features"] = inputs.input_features

    # 2. Tokenize transcripts for labels
    batch["labels"] = processor.tokenizer(text_target=transcripts).input_ids
    
    return batch

# --- Metrics Computation ---
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred, tokenizer: WhisperTokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in the labels as we can't decode them
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


def main():
    logger.info("Starting Whisper ASR model fine-tuning script.")
    set_seed(RANDOM_SEED)

    # --- 0. Verify User-Defined Paths ---
    logger.info(f"Using manifest file: {FINETUNE_DATA_MANIFEST_FILE}")
    logger.info(f"Using audio base directory: {AUDIO_FILES_BASE_DIR}")
    logger.info(f"Output model directory: {MODEL_OUTPUT_DIR}")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Whisper Processor, Feature Extractor, and Tokenizer ---
    logger.info(f"Loading WhisperProcessor for {PRETRAINED_MODEL_ID} (Language: {LANGUAGE}, Task: {TASK})")
    try:
        processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL_ID, language=LANGUAGE, task=TASK)
    except Exception as e:
        logger.error(f"Failed to load WhisperProcessor: {e}")
        logger.error("Ensure model ID is correct and you have internet access if model is not cached.")
        return

    # --- 2. Load and Prepare Dataset ---
    try:
        raw_dataset = load_and_prepare_data(FINETUNE_DATA_MANIFEST_FILE, AUDIO_FILES_BASE_DIR)
    except Exception as e:
        logger.error(f"Failed to load or prepare data: {e}")
        return
    
    logger.info(f"Raw dataset loaded with {len(raw_dataset)} samples.")
    logger.info(f"Dataset features: {raw_dataset.features}")
    logger.info(f"Sample record: {raw_dataset[0] if len(raw_dataset) > 0 else 'Dataset is empty'}")


    # --- 3. Preprocess Dataset ---
    logger.info("Preprocessing dataset for Whisper model inputs...")
    try:
        # Remove original columns after mapping as they are no longer needed
        # and to keep the dataset clean for the data collator.
        processed_dataset = raw_dataset.map(
            lambda batch: preprocess_function_for_whisper(batch, processor),
            batched=True,
            batch_size=8, # Adjust batch_size based on your system's memory
            remove_columns=raw_dataset.column_names 
        )
    except Exception as e:
        logger.error(f"Error during dataset mapping (preprocessing): {e}")
        logger.error("Check the preprocess_function_for_whisper and ensure data format is as expected.")
        import traceback
        traceback.print_exc()
        return

    logger.info(f"Dataset preprocessed. New columns: {processed_dataset.column_names}")
    # Expected columns: ['input_features', 'labels']

    # --- 4. Split Dataset into Training and Evaluation Sets ---
    if len(processed_dataset) < 2:
        logger.error("Not enough data to split into training and evaluation sets. Need at least 2 samples.")
        return
    
    # Ensure a minimum of 1 sample for evaluation, or 10% if dataset is large enough
    test_split_size = max(1, int(0.1 * len(processed_dataset))) if len(processed_dataset) >= 10 else 1
    
    if len(processed_dataset) - test_split_size < 1: # Not enough for a distinct training set
        logger.warning("Very small dataset. Using all available distinct samples for training and evaluation.")
        # This case needs careful handling if dataset is extremely small (e.g. 1 sample)
        # For simplicity, if only 1 sample total, it's used for both, but Trainer might complain.
        # If 2 samples, one for train, one for eval is typical.
        if len(processed_dataset) == 1:
             train_dataset = processed_dataset
             eval_dataset = processed_dataset # Trainer might not like this for distinct evaluation
             logger.warning("Dataset has only 1 sample. Using it for both train and eval. This is not ideal.")
        else: # len is >= 2
            # Ensure distinct train and eval if possible, even if small
            all_indices = list(range(len(processed_dataset)))
            random.shuffle(all_indices) # Shuffle to pick random eval sample
            eval_indices = all_indices[:test_split_size]
            train_indices = [i for i in all_indices if i not in eval_indices]
            if not train_indices: # Should not happen if len(processed_dataset) >=2 and test_split_size is reasonable
                train_indices = eval_indices # Fallback, not ideal
                logger.warning("Fallback: using eval set as train set due to split logic.")


            train_dataset = processed_dataset.select(train_indices)
            eval_dataset = processed_dataset.select(eval_indices)

    else:
        split_datasets = processed_dataset.train_test_split(test_size=test_split_size, seed=RANDOM_SEED, shuffle=True)
        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["test"]

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty after split. Check data and split logic.")
        return
    if len(eval_dataset) == 0:
        logger.warning("Evaluation dataset is empty after split. Evaluation will be skipped or might error.")
        # Potentially assign train_dataset to eval_dataset if no eval data, but this is for testing only
        # eval_dataset = train_dataset 

    # --- 5. Load Pre-trained Whisper Model ---
    logger.info(f"Loading WhisperForConditionalGeneration model: {PRETRAINED_MODEL_ID}")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_ID)
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return

    # Configure model for fine-tuning
    # Set decoder prompt IDs for the target language and task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.suppress_tokens = [] # Ensure no tokens are suppressed unless intended

    # --- 6. Initialize Data Collator ---
    # This is the recommended collator for Whisper.
    # Ensure your environment can import this (depends on torch, torchaudio, librosa, datasets).
    logger.info("Initializing DataCollatorSpeechSeq2SeqWithPadding.")
    try:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, # Pass the entire processor
            padding="longest"    # Pad to the longest sequence in the batch
        )
    except NameError:
        logger.error("CRITICAL: DataCollatorSpeechSeq2SeqWithPadding is not available!")
        logger.error("This usually means an issue with your 'transformers' library installation or its dependencies (torch, torchaudio, librosa, datasets).")
        logger.error("Please resolve your environment issues to use this correct collator.")
        logger.error("Attempting to fall back to DataCollatorForSeq2Seq with dummy input_ids (less ideal)...")
        logger.error("Ensure 'preprocess_function_for_whisper' creates dummy 'input_ids' and 'attention_mask' if using this fallback.")
        # If you must fallback, uncomment the dummy input_ids lines in preprocess_function_for_whisper
        # and use the CustomWhisperTrainer as in previous discussions.
        # For this "completely works" script, we assume the environment is correct.
        return 
    except Exception as e:
        logger.error(f"Error initializing DataCollatorSpeechSeq2SeqWithPadding: {e}")
        return

    # --- 7. Define Training Arguments ---
    logger.info("Defining Seq2SeqTrainingArguments.")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR / "training_checkpoints"),
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        per_device_eval_batch_size=4,   # Adjust based on GPU memory
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        eval_accumulation_steps=4,      # Accumulate predictions for evaluation to save memory
        learning_rate=1e-5,
        warmup_steps=100,               # Number of steps for learning rate warmup
        # max_steps=1000,               # Optionally set max_steps for shorter runs
        num_train_epochs=3,             # Number of training epochs
        eval_strategy="epoch",        # Evaluate at the end of each epoch ("steps" is also an option)
        save_strategy="epoch",        # Save model at the end of each epoch
        # eval_steps=200,               # If eval_strategy="steps"
        # save_steps=200,               # If save_strategy="steps"
        predict_with_generate=True,     # Use generate for evaluation (required for WER/CER with Whisper)
        generation_max_length=225,      # Max length for generated sequences
        logging_steps=25,               # Log training progress every 25 steps
        load_best_model_at_end=True,    # Load the best model found during training at the end
        metric_for_best_model="wer",    # Use WER to determine the best model
        greater_is_better=False,        # Lower WER is better
        fp16=torch.cuda.is_available(), # Enable mixed-precision training if a GPU is available
        remove_unused_columns=False,    # Important: DataCollatorSpeechSeq2SeqWithPadding handles columns.
                                        # If set to True, ensure your map function removes all original cols.
        report_to=["tensorboard"],      # Log to TensorBoard
        seed=RANDOM_SEED,
    )

    # --- 8. Initialize Trainer ---
    logger.info("Initializing Seq2SeqTrainer.")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer, # Pass tokenizer for decoding, not feature_extractor.
                                       # The collator uses processor.feature_extractor internally.
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
    )

    # --- 9. Start Fine-Tuning ---
    logger.info("Starting Whisper model fine-tuning...")
    try:
        train_result = trainer.train()
        logger.info("Fine-tuning finished successfully.")
        logger.info(f"Training metrics: {train_result.metrics}")
    except Exception as e:
        logger.error(f"ERROR during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 10. Save the Best Model and Processor ---
    logger.info(f"Saving best fine-tuned Whisper model and processor to: {MODEL_OUTPUT_DIR}")
    try:
        trainer.save_model(str(MODEL_OUTPUT_DIR)) # Saves the best model if load_best_model_at_end=True
        processor.save_pretrained(str(MODEL_OUTPUT_DIR))
        logger.info(f"All fine-tuned components saved to {MODEL_OUTPUT_DIR}.")
        logger.info(f"Files in output directory: {os.listdir(MODEL_OUTPUT_DIR)}")
    except Exception as e:
        logger.error(f"Error saving model/processor: {e}")
        return

    # --- 11. Final Evaluation (Optional - on the best model loaded) ---
    if eval_dataset and len(eval_dataset) > 0:
        logger.info("\nEvaluating the final (best) saved model on the evaluation set:")
        try:
            final_eval_results = trainer.evaluate(eval_dataset=eval_dataset)
            logger.info(f"Final evaluation results: {final_eval_results}")
            logger.info(f"Final evaluation - WER: {final_eval_results.get('eval_wer', 'N/A')}, CER: {final_eval_results.get('eval_cer', 'N/A')}")
        except Exception as e:
            logger.error(f"Error during final evaluation: {e}")
    else:
        logger.info("Skipping final evaluation as evaluation dataset is empty or not provided.")

    logger.info("Script execution completed.")

if __name__ == "__main__":
    main()
