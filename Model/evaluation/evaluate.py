from rouge import Rouge
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def compute_metrics(predictions_file, references_file):
    with open(predictions_file, 'r') as pred, open(references_file, 'r') as ref:
        predictions = pred.readlines()
        references = ref.readlines()

    assert len(predictions) == len(references), "Mismatch in predictions and references."

    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        p, r, f, _ = precision_recall_fscore_support([ref_tokens], [pred_tokens], average='macro', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    print("ROUGE:", scores)
    print("Precision:", np.mean(precisions))
    print("Recall:", np.mean(recalls))
    print("F1-Score:", np.mean(f1s))

if __name__ == "__main__":
    compute_metrics('outputs/predictions.txt', 'data/test.target')
