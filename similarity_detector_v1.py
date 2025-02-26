# Install necessary packages if not already installed:
# pip install sentence-transformers transformers torch

from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

# Load models
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Function to calculate sentence similarity
def calculate_similarity(sentence1, sentence2, model):
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

# Function to calculate perplexity
def calculate_perplexity(sentence, model, tokenizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity

# Example usage
original_sentence = "The quick brown fox jumps over the lazy dog."
predicted_sentence = "The fast brown fox leaps over a tired dog."

def calculate_similarity_2(original_sentence, predicted_sentence):
    # Calculate similarity
    similarity_score = calculate_similarity(original_sentence, predicted_sentence, similarity_model)
    print(f"Similarity Score: {similarity_score:.4f}")

    # Calculate perplexity
    perplexity_original = calculate_perplexity(original_sentence, gpt2_model, tokenizer)
    perplexity_predicted = calculate_perplexity(predicted_sentence, gpt2_model, tokenizer)

    print(f"Original Sentence Perplexity: {perplexity_original:.2f}")
    print(f"Predicted Sentence Perplexity: {perplexity_predicted:.2f}")

    return similarity_score, perplexity_original, perplexity_predicted

calculate_similarity_2(original_sentence, predicted_sentence)

# Analysis criteria:
# - High similarity score indicates good linguistic alignment.
# - Lower perplexity on the predicted sentence implies better natural language structure.
