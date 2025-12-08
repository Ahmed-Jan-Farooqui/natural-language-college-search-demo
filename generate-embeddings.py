from openai import OpenAI
import json
import time # Import time for rate limiting (good practice)

# 1. Configuration
client = OpenAI()
BATCH_SIZE = 1000 # Number of names per API request
MODEL_NAME = "text-embedding-3-small"

# 2. Helper function to break a list into chunks
def chunk_list(data, chunk_size):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# --- Main Script ---

# 3. Load Data
with open('data/current-colleges.json', 'r') as file:
    college_list_of_dicts = json.load(file)

# Extract only the names into a flat list
college_names = [college['name'] for college in college_list_of_dicts]

final_dict = {}

# 4. Process in Batches
print(f"Total items to embed: {len(college_names)}")

# Use the helper function to iterate over chunks
for i, name_chunk in enumerate(chunk_list(college_names, BATCH_SIZE)):
    print(f"Processing batch {i + 1} with {len(name_chunk)} items...")
    
    try:
        # Send the entire list of names (the chunk) in a single API call
        response = client.embeddings.create(
            input=name_chunk,  # This is the list of 1000 names
            model=MODEL_NAME,
        )

        # 5. Store Results
        # The response.data contains one embedding object for each input name,
        # and they are returned in the same order they were sent.
        for j, embedding_object in enumerate(response.data):
            # Map the embedding back to the original college name
            name = name_chunk[j]
            final_dict[name] = embedding_object.embedding
            
    except Exception as e:
        print(f"An error occurred in batch {i + 1}: {e}")
        # Consider a sleep here if it's a rate limit error
        time.sleep(5) 
        continue # Skip to the next batch

# 6. Write to the file
print(f"Successfully embedded {len(final_dict)} names. Writing to file...")
with open('data/embeddings.json', 'w') as file:
    file.write(json.dumps(final_dict))

print("Embedding process complete.")