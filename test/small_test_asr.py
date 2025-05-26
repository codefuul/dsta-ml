import base64
import json
from pathlib import Path
import itertools
# requests is not used in this simplified test if we simulate the response directly
# from tqdm import tqdm # tqdm is also optional for a very small test

BATCH_SIZE = 2

# Create a small sample of instances for testing
sample_instances = [
    {
        "key": "sample1",
        "audio": "sample1.wav", # audio field is present but not used by this sample_generator
        "transcript": "hello world"
    },
    {
        "key": "sample2",
        "audio": "sample2.wav", # audio field is present but not used by this sample_generator
        "transcript": "test case"
    }
]

def sample_generator(instances_list, data_dir_path): # Renamed for clarity
    for instance_item in instances_list:
        # For testing, simulate audio bytes as base64 of transcript string
        audio_bytes = instance_item["transcript"].encode("utf-8")
        yield {
            "key": instance_item["key"],
            "b64": base64.b64encode(audio_bytes).decode("ascii"),
            "transcript": instance_item["transcript"]  # <<< ADD THIS LINE
        }

def main():
    data_dir = Path("./test_audio")  # Dummy path for test
    # Using the predefined sample_instances directly
    # instances = sample_instances # This line is redundant if sample_instances is global

    # Use sample_instances directly
    batch_generator = itertools.batched(sample_generator(sample_instances, data_dir), n=BATCH_SIZE)

    results = []
    # total=1 assumes BATCH_SIZE >= len(sample_instances) or only one batch
    # A more robust total would be math.ceil(len(sample_instances) / BATCH_SIZE)
    # from tqdm import tqdm # if you want the progress bar
    # import math
    # for batch in tqdm(batch_generator, total=math.ceil(len(sample_instances) / BATCH_SIZE)):
    for batch in batch_generator: # Simpler loop for small test
        # Now 'inst' will have 'transcript' key
        simulated_response = {"predictions": [inst["transcript"] for inst in batch]}
        results.extend(simulated_response["predictions"])

    print("Test results:", results)
    # Expected output: Test results: ['hello world', 'test case']

if __name__ == "__main__":
    main()
