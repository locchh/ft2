import numpy as np
from rouge import Rouge
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu

def calculate_metrics(reference_texts, candidate_texts):
    """
    Calculate BERTScore, ROUGE-L, BLEU-4, F1-Score
    :param reference_texts: List of reference sentences (ground truth).
    :param candidate_texts: List of candidate sentences (generated by the model).
    :return: A dictionary with calculated metrics.
    """
     # Ensure the inputs are valid
    if len(reference_texts) != len(candidate_texts):
        raise ValueError("Reference and candidate lists must be of the same length.")

    # Calculate BERTScore
    P, R, F1 = bert_score(candidate_texts, reference_texts, lang='en', return_hash=False)

    # Calculate ROUGE-L
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate_texts, reference_texts, avg=True)

    # Calculate BLEU-4
    bleu_scores = [
        sentence_bleu([ref.split() for ref in reference_texts], candidate.split(), weights=(0.25, 0.25, 0.25, 0.25))
        for candidate in candidate_texts
    ]
    bleu_mean = np.mean(bleu_scores)
    
    # Calculate F1-Score
    f1_score = 2 * (P.mean() * R.mean()) / (P.mean() + R.mean() + 1e-10)  # Add a small value to avoid division by zero

    # Prepare results
    results = {
        'BERTScore': {
            'Precision': P.mean().item(),
            'Recall': R.mean().item(),
            'F1': F1.mean().item()
        },
        'ROUGE-L': {
            'F1': rouge_scores['rouge-l']['f'],
            'Precision': rouge_scores['rouge-l']['p'],
            'Recall': rouge_scores['rouge-l']['r']
        },
        'BLEU-4': bleu_mean,
        'F1-Score': f1_score.item()  # Converting to a scalar
    }

    return results

