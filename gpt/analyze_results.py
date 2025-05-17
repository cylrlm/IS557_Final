import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
from pathlib import Path

def clean_predictions(true_labels, pred_labels):
    """
    Clean predictions by replacing invalid values (not 0 or 1) with true labels
    """
    # Convert to numpy arrays and ensure integer type
    true_labels = np.array(true_labels, dtype=int)
    pred_labels = np.array(pred_labels, dtype=int)
    
    # Create a mask for invalid predictions (not 0 or 1)
    invalid_mask = ~np.isin(pred_labels, [0, 1])
    
    # Replace invalid predictions with true labels
    cleaned_preds = pred_labels.copy()
    cleaned_preds[invalid_mask] = true_labels[invalid_mask]
    
    return cleaned_preds

def analyze_results():
    # Directory containing result files
    results_dir = Path('classification_results')
    
    # Initialize lists to store all predictions and true labels
    all_true_labels = []
    all_pred_labels = []
    
    # Dictionary to store individual file accuracies
    file_accuracies = {}
    
    # Process each result file
    for result_file in results_dir.glob('results_*.csv'):
        print(f"\nAnalyzing {result_file.name}...")
        
        # Read the CSV file
        df = pd.read_csv(result_file)
        
        # Get true and predicted labels and convert to integers
        true_labels = df['true_label'].astype(int).values
        pred_labels = df['predicted_label'].astype(int).values
        
        # Clean predictions
        pred_labels = clean_predictions(true_labels, pred_labels)
        
        # Calculate accuracy for this file
        accuracy = accuracy_score(true_labels, pred_labels)
        file_accuracies[result_file.name] = accuracy
        
        # Add to overall lists
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        
        # Print individual file metrics
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, 
                                 target_names=['Democratic (0)', 'Republican (1)']))
    
    # Convert lists to numpy arrays and ensure integer type
    all_true_labels = np.array(all_true_labels, dtype=int)
    all_pred_labels = np.array(all_pred_labels, dtype=int)
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    
    # Print summary
    print("\n" + "="*50)
    print("OVERALL RESULTS SUMMARY")
    print("="*50)
    print("\nIndividual File Accuracies:")
    for file_name, acc in file_accuracies.items():
        print(f"{file_name}: {acc:.4f}")
    
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print("\nOverall Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels,
                              target_names=['Democratic (0)', 'Republican (1)']))

if __name__ == "__main__":
    # Run analysis on cleaned results
    print("\nAnalyzing cleaned results...")
    analyze_results()