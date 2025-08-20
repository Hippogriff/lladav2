import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import glob

def tokenize_and_save_dataset_sharded(json_file_path, text_column, model_name, save_path, chunk_size=100000):
    """
    Tokenizes a large JSON dataset and saves it to disk in sharded Arrow files
    to handle datasets that don't fit in RAM.

    Args:
        json_file_path (str): Path to the large JSONL file.
        text_column (str): The name of the column in the JSON file that contains the text.
        model_name (str): The name of the pretrained model for tokenization.
        save_path (str): The directory where the sharded dataset will be saved.
        chunk_size (int): The number of examples to save in each shard.
    """
    # --- 1. Create save directory ---
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving sharded dataset to '{save_path}'")

    # --- 2. Load the dataset in streaming mode ---
    print("Loading dataset in streaming mode...")
    # Using 'json' which works for json lines (.jsonl) as well.
    raw_dataset = load_dataset('json', data_files=json_file_path, split='train', streaming=True)

    # --- 3. Load the tokenizer ---
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- 4. Define the tokenization function ---
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=512)

    # --- 5. Tokenize the dataset using .map() ---
    print("Tokenizing the dataset (this may take a while for large datasets)...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column]
    )

    # --- 6. Iterate and save the processed dataset in chunks ---
    buffer = []
    shard_index = 0
    for i, example in enumerate(tokenized_dataset):
        buffer.append(example)
        if len(buffer) == chunk_size:
            # Convert buffer to a Dataset object
            shard = Dataset.from_list(buffer)
            
            # Save the shard to a file
            shard_path = os.path.join(save_path, f'shard-{shard_index:05d}.arrow')
            shard.save_to_disk(shard_path)
            
            print(f"Saved shard {shard_index} with {len(buffer)} examples to {shard_path}")
            
            # Reset buffer and increment shard index
            buffer = []
            shard_index += 1
    
    # Save any remaining examples in the buffer as the last shard
    if buffer:
        shard = Dataset.from_list(buffer)
        shard_path = os.path.join(save_path, f'shard-{shard_index:05d}.arrow')
        shard.save_to_disk(shard_path)
        print(f"Saved final shard {shard_index} with {len(buffer)} examples to {shard_path}")

    print("\nTokenization and sharded saving complete!")


if __name__ == '__main__':
    # --- Configuration ---
    # Create a dummy JSONL file for demonstration purposes
    dummy_file = 'large_dataset.jsonl'
    # Make a slightly larger dummy file to demonstrate chunking
    with open(dummy_file, 'w') as f:
        for i in range(15): # Create 15 lines
            f.write(f'{{"text": "This is sentence number {i} in our large dataset."}}\n')

    JSON_FILE_PATH = dummy_file
    TEXT_COLUMN = 'text'
    MODEL_NAME = 'bert-base-uncased'
    SAVE_PATH = './tokenized_dataset_sharded'
    CHUNK_SIZE = 10 # Use a small chunk size for this demo

    tokenize_and_save_dataset_sharded(JSON_FILE_PATH, TEXT_COLUMN, MODEL_NAME, SAVE_PATH, chunk_size=CHUNK_SIZE)

    # --- How to load the sharded dataset for training ---
    print("\n--- Loading the saved sharded dataset for training ---")
    
    # Use a glob pattern to find all the shard directories
    shard_paths = glob.glob(os.path.join(SAVE_PATH, 'shard-*'))
    
    # Load all shards from disk
    reloaded_dataset = [load_from_disk(shard_path) for shard_path in sorted(shard_paths)]
    
    # If you need to combine them into a single Dataset object (requires memory)
    from datasets import concatenate_datasets
    full_dataset = concatenate_datasets(reloaded_dataset)

    print("\nFull reloaded dataset info:")
    print(full_dataset)
    print("Total examples:", len(full_dataset))
    print("First example:", full_dataset[0])
    print("Eleventh example (from the second shard):", full_dataset[10])
