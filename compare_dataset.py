import pandas as pd
import sys

# Add the directory containing the external Python file to the system path
sys.path.append('/Users/fiona/VSCode/SHS291_Speech_To_Text')

# Import the similarity calculation function from the provided Python file
from similarity_detector_v1 import calculate_similarity_2

# Load the uploaded Excel file
file_path = '/Users/fiona/VSCode/SHS291_Speech_To_Text/Source Data/TextTranscriptions.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Concatenate columns D to M (3 to 12) and R to AA (17 to 32) to form sentences
df['TTS Sentence'] = df.iloc[:, 3:13].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df['Answer Sentence'] = df.iloc[:, 17:33].apply(lambda x: ' '.join([word for word in x.dropna().astype(str) if word != '0']), axis=1)

# Filter rows with non-empty TTS and Answer sentences
sentence_pairs = df[['TTS Sentence', 'Answer Sentence']].dropna()

# Prepare a list to store the actual similarity and perplexity results
actual_results = []

# Iterate over the valid sentence pairs and compute the similarity and perplexity
for index, row in sentence_pairs.iterrows():
    tts_sentence = row['TTS Sentence']
    answer_sentence = row['Answer Sentence']
    
    # Calculate similarity and perplexity using the provided function
    try:
        similarity_score, perplexity_original, perplexity_predicted = calculate_similarity_2(tts_sentence, answer_sentence)
        
        actual_results.append({
            'TTS Sentence': tts_sentence,
            'Answer Sentence': answer_sentence,
            'Similarity Score': round(similarity_score, 4),
            'Original Perplexity': round(perplexity_original, 2),
            'Predicted Perplexity': round(perplexity_predicted, 2)
        })
    except Exception as e:
        print(f"Error processing sentences: {e}")

# Convert the results to a DataFrame and save as an Excel file
actual_results_df = pd.DataFrame(actual_results)
actual_results_df.to_excel('/Users/fiona/VSCode/SHS291_Speech_To_Text/Source Data/Similarity_Perplexity_Results.xlsx', index=False)

print("Results saved to 'Similarity_Perplexity_Results.xlsx'")
