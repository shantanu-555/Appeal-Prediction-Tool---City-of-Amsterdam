import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data():
    """Load the training data and embeddings"""
    training_embeddings = joblib.load('training_embeddings.joblib')
    training_texts = joblib.load('training_texts.joblib')
    return training_embeddings, training_texts

def find_similar_cases(
    text_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    all_texts: List[Dict],
    n_cases: int = 2
) -> List[Dict]:
    """Find similar cases for a given text embedding"""
    similarities = cosine_similarity(text_embedding.reshape(1, -1), all_embeddings)[0]
    top_indices = similarities.argsort()[-n_cases:][::-1]
    
    similar_cases = []
    for idx in top_indices:
        similar_cases.append({
            'text': all_texts[idx]['text'],
            'label': all_texts[idx]['label'],
            'similarity': similarities[idx]
        })
    return similar_cases

def calculate_composite_prediction(
    model_pred: int,
    model_prob: float,
    similar_cases: List[Dict],
    model_weight: float
) -> Tuple[int, float]:
    """Calculate composite prediction and confidence using given weights"""
    if not similar_cases:
        return model_pred, model_prob
    
    # Calculate similarity-weighted vote from similar cases
    total_similarity = sum(case['similarity'] for case in similar_cases)
    if total_similarity == 0:
        return model_pred, model_prob
        
    similar_vote = sum(
        case['similarity'] * case['label']
        for case in similar_cases
    ) / total_similarity
    
    # Calculate composite confidence
    similar_weight = 1 - model_weight
    composite_confidence = (model_weight * model_prob) + (similar_weight * similar_vote)
    
    # Determine final prediction
    composite_pred = 1 if composite_confidence >= 0.5 else 0
    
    return composite_pred, composite_confidence

def evaluate_weights(
    embeddings: np.ndarray,
    texts: List[Dict],
    model_predictions: np.ndarray,
    model_probabilities: np.ndarray,
    true_labels: np.ndarray,
    weight_range: np.ndarray
) -> Dict:
    """Evaluate different weight combinations"""
    results = []
    
    for weight in tqdm(weight_range):
        composite_preds = []
        composite_confs = []
        
        # For each sample, calculate composite prediction
        for i in range(len(texts)):
            # Get similar cases (excluding the current case)
            mask = np.ones(len(texts), dtype=bool)
            mask[i] = False
            similar_cases = find_similar_cases(
                embeddings[i],
                embeddings[mask],
                [texts[j] for j, m in enumerate(mask) if m],
                n_cases=2
            )
            
            # Calculate composite prediction
            pred, conf = calculate_composite_prediction(
                model_predictions[i],
                model_probabilities[i],
                similar_cases,
                weight
            )
            composite_preds.append(pred)
            composite_confs.append(conf)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, composite_preds)
        f1 = f1_score(true_labels, composite_preds)
        
        results.append({
            'model_weight': weight,
            'accuracy': accuracy,
            'f1': f1
        })
    
    return pd.DataFrame(results)

def plot_results(results: pd.DataFrame):
    """Plot the evaluation results"""
    plt.figure(figsize=(10, 6))
    plt.plot(results['model_weight'], results['accuracy'], label='Accuracy')
    plt.plot(results['model_weight'], results['f1'], label='F1 Score')
    plt.xlabel('Model Weight')
    plt.ylabel('Score')
    plt.title('Performance vs Model Weight')
    plt.legend()
    plt.grid(True)
    plt.savefig('weight_evaluation.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    embeddings, texts = load_data()
    
    # Load model predictions (you'll need to generate these)
    # For this example, we'll use the stored labels as predictions
    model_predictions = np.array([text['label'] for text in texts])
    model_probabilities = np.array([1.0 if pred == 1 else 0.0 for pred in model_predictions])
    true_labels = np.array([text['label'] for text in texts])
    
    # Evaluate different weights
    print("Evaluating weights...")
    weight_range = np.arange(0.1, 1.0, 0.1)
    results = evaluate_weights(
        embeddings,
        texts,
        model_predictions,
        model_probabilities,
        true_labels,
        weight_range
    )
    
    # Find optimal weight
    best_f1_idx = results['f1'].argmax()
    optimal_weight = results.iloc[best_f1_idx]['model_weight']
    
    print("\nResults:")
    print(f"Optimal model weight: {optimal_weight:.2f}")
    print(f"Optimal similar cases weight: {1-optimal_weight:.2f}")
    print(f"Best F1 score: {results.iloc[best_f1_idx]['f1']:.3f}")
    print(f"Corresponding accuracy: {results.iloc[best_f1_idx]['accuracy']:.3f}")
    
    # Plot results
    plot_results(results)
    print("\nResults plot saved as 'weight_evaluation.png'")

if __name__ == "__main__":
    main() 