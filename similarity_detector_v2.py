# Install necessary packages if not already installed:
# pip install sentence-transformers transformers torch bert-score spacy

from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score import score
import torch
import math
import spacy

# Load models
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to calculate sentence similarity using Cosine Similarity
def calculate_cosine_similarity(sentence1, sentence2, model):
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

# Function to calculate BERTScore similarity
def calculate_bert_score(sentence1, sentence2):
    P, R, F1 = score([sentence1], [sentence2], lang="en", model_type="bert-base-uncased")
    return F1.item()

# Function to calculate syntactic similarity using spaCy
def calculate_syntactic_similarity(sentence1, sentence2):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    return doc1.similarity(doc2)

# Example sentences
original_sentence = "The quick brown fox jumps over the lazy dog."
predicted_sentence = "The fast brown fox leaps over a tired dog."

# Compute similarity metrics
cosine_similarity = calculate_cosine_similarity(original_sentence, predicted_sentence, similarity_model)
bert_similarity = calculate_bert_score(original_sentence, predicted_sentence)
syntactic_similarity = calculate_syntactic_similarity(original_sentence, predicted_sentence)

# Compute perplexity
perplexity_original = calculate_perplexity(original_sentence, gpt2_model, tokenizer)
perplexity_predicted = calculate_perplexity(predicted_sentence, gpt2_model, tokenizer)

# Weighted similarity score (0.4 Cosine, 0.4 BERT, 0.2 Syntax)
final_similarity_score = (0.4 * cosine_similarity) + (0.4 * bert_similarity) + (0.2 * syntactic_similarity)

# Display results
print(f"Cosine Similarity: {cosine_similarity:.4f}")
print(f"BERTScore (F1): {bert_similarity:.4f}")
print(f"Syntactic Similarity: {syntactic_similarity:.4f}")
print(f"Original Sentence Perplexity: {perplexity_original:.2f}")
print(f"Predicted Sentence Perplexity: {perplexity_predicted:.2f}")
print(f"Final Weighted Similarity Score: {final_similarity_score:.4f}")

# Interpretation:
# - High Final Similarity Score (close to 1.0) means high semantic similarity.
# - Low Perplexity means the sentence is natural and fluent.
