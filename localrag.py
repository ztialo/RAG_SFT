import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def chunk_text_500_chars(text, chunk_size=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks
# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
def file_to_array(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.readlines()
# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=2):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='embeddinggemma:300m', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context


def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model):
    # Rewriting query if needed
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content)

    context_str = "\n".join(relevant_context)

    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context += "\n\nRelevant Context:\n" + context_str

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input_with_context}
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=20,
    )

    answer = response.choices[0].message.content.strip()
    return answer, relevant_context

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3.2:1b", help="Ollama model to use (default: llama3.2:1b)")
parser.add_argument("--question_file", default=None, help="Path to file containing list of questions")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")


# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
if not vault_content:
    raise ValueError("vault.txt is empty or missing.")

print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='embeddinggemma:300m', prompt=content)
    vault_embeddings.append(response["embedding"])

# Convert to tensor and print embeddings
print("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

system_message = ("You are a straightforward assistant that picks out information from context without being verbose."
" Answer using the *fewest possible characters*.  Do not use full sentences. Use fragments only. Maximum length: 30 characters."
"If possible, answer in one or two words."
"Based on the context answer this question:"
)
#question_list
questions = []
if args.question_file:
    print(NEON_GREEN + f"Loading questions from {args.question_file}..." + RESET_COLOR)
    questions = file_to_array(args.question_file)


results = []

if questions:
    print(NEON_GREEN + "Beginning automated question answering..." + RESET_COLOR)
    for q in questions:
        q = q.strip()
        if not q:
            continue

        print(YELLOW + f"\n▶ QUESTION: {q}" + RESET_COLOR)

        answer, context = ollama_chat(
            q,
            system_message,
            vault_embeddings_tensor,
            vault_content,
            args.model
        )

        print(NEON_GREEN + f"ANSWER: {answer}" + RESET_COLOR)

        results.append({
            "input": q,
            "context": "\n".join(context),
            "response": answer
        })

    # Write JSON file
    output_path = "rolling_rag_mymodel_q555.json"
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=4)

    print(NEON_GREEN + f"\nSaved results to {output_path}" + RESET_COLOR)

else:
    print("No question file provided. Use interactive mode or provide --question_file.")


