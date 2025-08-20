import os
import json
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
import glob

def tokenize_and_save_dataset_chunked(jsonl_file_path, text_column, model_name, save_path, chunk_size=100000):
    """
    Reads a massive JSONL file chunk by chunk, tokenizes each chunk, and saves it
    to disk as a separate shard. This is a robust method for processing datasets
    that are too large to handle in a single operation.

    Args:
        jsonl_file_path (str): Path to the large JSONL file.
        text_column (str): The name of the column in the JSON that contains the text.
        model_name (str): The name of the pretrained model for tokenization.
        save_path (str): The directory where the sharded dataset will be saved.
        chunk_size (int): The number of lines/examples to process in each chunk.
    """
    # --- 1. Create save directory and load tokenizer ---
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving sharded dataset to '{save_path}'")
    
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- 2. Define the tokenization function (no changes needed here) ---
    def tokenize_function(examples):
        # This function will be applied to each chunk
        return tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=512)

    # --- 3. Read the file, process, and save in chunks ---
    buffer = []
    shard_index = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Add the parsed JSON object to the buffer
            buffer.append(json.loads(line))
            
            # When the buffer is full, process and save the chunk
            if len(buffer) == chunk_size:
                # Convert the list of dicts into a Hugging Face Dataset
                chunk_dataset = Dataset.from_list(buffer)
                
                print(f"\nProcessing chunk {shard_index} with {len(chunk_dataset)} examples...")
                
                # Tokenize the chunk
                tokenized_chunk = chunk_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column]
                )
                
                # Save the tokenized chunk as a shard
                shard_path = os.path.join(save_path, f'shard-{shard_index:05d}.arrow')
                tokenized_chunk.save_to_disk(shard_path)
                print(f"Saved shard {shard_index} to {shard_path}")

                # Reset buffer and increment shard index
                buffer = []
                shard_index += 1

    # --- 4. Save any remaining examples in the buffer as the final shard ---
    if buffer:
        chunk_dataset = Dataset.from_list(buffer)
        print(f"\nProcessing final chunk {shard_index} with {len(chunk_dataset)} examples...")
        tokenized_chunk = chunk_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_column]
        )
        shard_path = os.path.join(save_path, f'shard-{shard_index:05d}.arrow')
        tokenized_chunk.save_to_disk(shard_path)
        print(f"Saved final shard {shard_index} to {shard_path}")

    print("\nTokenization and sharded saving complete!")


if __name__ == '__main__':
    # --- Configuration ---
    # Create a dummy JSONL file for demonstration purposes
    dummy_file = 'large_dataset.jsonl'
    with open(dummy_file, 'w') as f:
        for i in range(25): # Create 25 lines to show multiple chunks
            f.write(f'{{"text": "This is sentence number {i} in our large dataset."}}\n')

    JSON_FILE_PATH = dummy_file
    TEXT_COLUMN = 'text'
    MODEL_NAME = 'bert-base-uncased'
    SAVE_PATH = './tokenized_dataset_chunked'
    CHUNK_SIZE = 10 # Use a small chunk size for this demo

    tokenize_and_save_dataset_chunked(JSON_FILE_PATH, TEXT_COLUMN, MODEL_NAME, SAVE_PATH, chunk_size=CHUNK_SIZE)

    # --- How to load the sharded dataset for training (this part remains the same) ---
    print("\n--- Loading the saved sharded dataset for training ---")
    
    # Use a glob pattern to find all the shard directories
    shard_paths = glob.glob(os.path.join(SAVE_PATH, 'shard-*'))
    
    # Load all shards from disk
    # The list comprehension is lazy, it doesn't load all data into RAM at once
    reloaded_shards = [load_from_disk(shard_path) for shard_path in sorted(shard_paths)]
    
    # You can now pass this list of datasets directly to many training frameworks,
    # or concatenate them if you have enough RAM.
    from datasets import concatenate_datasets
    full_dataset = concatenate_datasets(reloaded_shards)

    print("\nFull reloaded dataset info:")
    print(full_dataset)
    print("Total examples:", len(full_dataset))
    print("First example:", full_dataset[0])
    print("Twenty-first example (from the third shard):", full_dataset[20])
