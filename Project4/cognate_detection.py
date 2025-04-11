import pandas as pd
import numpy as np
from collections import defaultdict
import re
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance


def load_data(ukhrul_file, target_file, gold_file=None):
    """
    Load datasets for Ukhrul and the target language from TSV files.

    Parameters:
    - ukhrul_file: Path to the Ukhrul language .tsv file
    - target_file: Path to the target language .tsv file
    - gold_file: Optional path to a .tsv file containing gold labels

    Returns:
    - ukhrul_df: DataFrame with Ukhrul language data
    - target_df: DataFrame with target language data
    - gold_labels: DataFrame with gold labels (or None if not provided)
    """

    if not ukhrul_file.endswith(".tsv"):
        raise ValueError(f"File {ukhrul_file} is not a TSV file.")
    if not target_file.endswith(".tsv"):
        raise ValueError(f"File {target_file} is not a TSV file.")
    if gold_file and not gold_file.endswith(".tsv"):
        raise ValueError(f"File {gold_file} is not a TSV file.")

    ukhrul_df = pd.read_csv(ukhrul_file, sep='\t', header=None, names=['form', 'gloss'])
    target_df = pd.read_csv(target_file, sep='\t', header=None, names=['form', 'gloss'])

    gold_labels = None
    if gold_file:
        gold_labels = pd.read_csv(gold_file, sep='\t', header=None, names=['cognate'])

    return ukhrul_df, target_df, gold_labels


def normalized_levenshtein(str1, str2):
    """
    Calculate normalized Levenshtein distance (0-1, where 1 is identical)
    """
    if not str1 or not str2:
        return 0
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1
    return 1 - levenshtein_distance(str1, str2) / max_len

def sequence_similarity(str1, str2):
    """
    Calculate sequence similarity using SequenceMatcher
    """
    return SequenceMatcher(None, str1, str2).ratio()

def semantic_similarity(gloss1, gloss2):
    """
    Calculate semantic similarity based on overlap of tokens in glosses
    """
    # Normalize and tokenize glosses
    tokens1 = set(re.findall(r'\b\w+\b', gloss1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', gloss2.lower()))
    
    # Calculate Jaccard similarity
    if not tokens1 or not tokens2:
        return 0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union)


def extract_sound_correspondences(ukhrul_forms, cognate_forms):
    """
    Extract recurring sound correspondence patterns from known cognate pairs
    """
    correspondences = defaultdict(lambda: defaultdict(int))
    
    for ukhrul_word, cognate_word in zip(ukhrul_forms, cognate_forms):
        # Simple character-by-character alignment (could be improved)
        min_len = min(len(ukhrul_word), len(cognate_word))
        for i in range(min_len):
            correspondences[ukhrul_word[i]][cognate_word[i]] += 1
    
    # Normalize counts to probabilities
    probability_map = {}
    for uk_char, target_chars in correspondences.items():
        total = sum(target_chars.values())
        probability_map[uk_char] = {t_char: count/total for t_char, count in target_chars.items()}
    
    return probability_map


def calculate_correspondence_score(ukhrul_word, candidate_word, correspondence_map):
    """
    Calculate how well a candidate fits known sound correspondence patterns
    """
    score = 0
    count = 0
    
    # Simple character-by-character scoring
    min_len = min(len(ukhrul_word), len(candidate_word))
    for i in range(min_len):
        uk_char = ukhrul_word[i]
        cand_char = candidate_word[i]
        
        if uk_char in correspondence_map and cand_char in correspondence_map[uk_char]:
            score += correspondence_map[uk_char][cand_char]
            count += 1
    
    return score / max(count, 1)  

def combined_similarity(ukhrul_word, ukhrul_gloss, candidate_word, candidate_gloss, correspondence_map, weights):
    """
    Calculate combined similarity score with weighted components
    """
    # Calculate individual similarity scores
    lev_sim = normalized_levenshtein(ukhrul_word, candidate_word)
    seq_sim = sequence_similarity(ukhrul_word, candidate_word)
    sem_sim = semantic_similarity(ukhrul_gloss, candidate_gloss)
    corr_sim = calculate_correspondence_score(ukhrul_word, candidate_word, correspondence_map)
    
    # Length ratio penalty (penalize big differences in word length)
    len_ratio = min(len(ukhrul_word), len(candidate_word)) / max(len(ukhrul_word), len(candidate_word))
    
    # Combine scores with weights
    combined_score = (
        weights['levenshtein'] * lev_sim +
        weights['sequence'] * seq_sim +
        weights['semantic'] * sem_sim +
        weights['correspondence'] * corr_sim +
        weights['length_ratio'] * len_ratio
    )
    
    return combined_score

def find_top_candidates(ukhrul_df, target_df, correspondence_map, weights, top_n=5):
    """
    Find top cognate candidates for each Ukhrul word
    """
    results = []
    
    for _, uk_row in ukhrul_df.iterrows():
        uk_word = uk_row['form']
        uk_gloss = uk_row['gloss']
        
        candidates = []
        for _, target_row in target_df.iterrows():
            target_word = target_row['form']
            target_gloss = target_row['gloss']
            
            score = combined_similarity(
                uk_word, uk_gloss, target_word, target_gloss, 
                correspondence_map, weights
            )
            
            candidates.append((target_word, target_gloss, score))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates[:top_n]
        
        row = []
        for word, gloss, _ in top_candidates:
            row.extend([word, gloss])
        
        while len(row) < 10:
            row.extend(['', ''])  
        
        results.append('\t'.join(row))
    
    return results



def train_weights(ukhrul_df, target_df, gold_df):
    cognate_forms = gold_df['cognate'].tolist()
    ukhrul_forms = ukhrul_df['form'].tolist()
    
    correspondence_map = extract_sound_correspondences(ukhrul_forms, cognate_forms)
    
    weights = {
        'levenshtein': 0.3,
        'sequence': 0.2,
        'semantic': 0.3,
        'correspondence': 0.15,
        'length_ratio': 0.05
    }
    
    return weights, correspondence_map


def main():
    # Load development data (Ukhrul-Huishu)
    ukhrul_huishu_df, huishu_df, gold_df = load_data(
        'dataset/ukhrul-huishu_inputs.tsv', 
        'dataset/huishu_candidates.tsv', 
        'dataset/ukhrul-huishu_gold.tsv'
    )
    
    # Train weights and extract correspondence patterns
    weights, correspondence_map = train_weights(ukhrul_huishu_df, huishu_df, gold_df)
    
    # Process Ukhrul-Tusom
    ukhrul_tusom_df, tusom_df, _ = load_data('dataset/ukhrul-tusom_inputs.tsv', 'dataset/tusom_candidates.tsv')
    tusom_results = find_top_candidates(ukhrul_tusom_df, tusom_df, correspondence_map, weights)
    
    # Process Ukhrul-Kachai
    ukhrul_kachai_df, kachai_df, _ = load_data('dataset/ukhrul-kachai_inputs.tsv', 'dataset/kachai_candidates.tsv')
    kachai_results = find_top_candidates(ukhrul_kachai_df, kachai_df, correspondence_map, weights)
    
    # Save results
    with open('ukhrul-tusom_out.tsv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(tusom_results))
    
    with open('ukhrul-kachai_out.tsv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(kachai_results))
    
    print("Processing complete. Output files saved.")

if __name__ == "__main__":
    main()

