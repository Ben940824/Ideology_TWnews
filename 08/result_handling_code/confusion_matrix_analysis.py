#!/usr/bin/env python3
"""
Confusion Matrix Analysis for Classification Predictions
Combines all fold results and generates comprehensive confusion matrix analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_combine_predictions():
    """Load all fold predictions and combine them"""
    all_predictions = []
    
    # Load predictions from all 5 folds
    for fold in range(5):
        file_path = f"predictions_cls_fold{fold}.csv"
        print(f"Loading {file_path}...")
        
        try:
            df = pd.read_csv(file_path, dtype={'label_encoded': 'int', 'pred': 'int'})
            print(f"  - Loaded {len(df)} samples from fold {fold}")
            all_predictions.append(df)
        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")
            continue
    
    if not all_predictions:
        raise ValueError("No prediction files could be loaded")
    
    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True)
    print(f"\nTotal combined samples: {len(combined_df)}")
    
    return combined_df

def analyze_predictions(df):
    """Perform comprehensive confusion matrix analysis"""
    # Get true labels and predictions
    y_true = df['label_encoded'].values
    y_pred = df['pred'].values
    
    # Create mapping from encoded labels to actual label names
    label_mapping = {}
    for label_code in df['label_encoded'].unique():
        label_name = df[df['label_encoded'] == label_code]['label'].iloc[0]
        label_mapping[label_code] = label_name
    
    # Define the desired order: 偏綠 - 無明顯立場 - 偏藍
    desired_order = ['偏綠', '無明顯立場', '偏藍']
    english_translations = ['pro-green', 'neutral', 'pro-blue']
    
    # Create ordered labels and their encoded values
    unique_labels = []
    label_names = []
    label_names_en = []
    
    for chinese_label, english_label in zip(desired_order, english_translations):
        # Find the encoded value for this label
        for code, name in label_mapping.items():
            if name == chinese_label:
                unique_labels.append(code)
                label_names.append(f"{english_label}")
                label_names_en.append(english_label)
                break
    
    print(f"Label mapping: {label_mapping}")
    print(f"Ordered labels: {list(zip(unique_labels, desired_order, english_translations))}")
    print(f"Number of classes: {len(unique_labels)}")
    
    # Calculate confusion matrix with ordered labels
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=unique_labels, 
                                 target_names=desired_order, output_dict=True)
    
    return cm, accuracy, report, label_names, label_mapping, desired_order, english_translations

def plot_confusion_matrix(cm, label_names, accuracy):
    """Create and save confusion matrix visualization"""
    plt.figure(figsize=(14, 12))
    
    # Create heatmap using a neutral color scheme (purple/magenta)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Number of Samples'},
                annot_kws={'size': 16})  # Larger annotation font
    
    plt.title(f'Confusion Matrix - All Folds Combined\nOverall Accuracy: {accuracy:.4f}', 
              fontsize=22, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
    plt.ylabel('True Label', fontsize=18, fontweight='bold')
    
    # Rotate x-axis labels for better readability with larger fonts
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_all_folds.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_normalized_confusion_matrix(cm, label_names, accuracy):
    """Create and save normalized confusion matrix"""
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 12))
    
    # Create heatmap using a neutral color scheme (orange/red)
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Oranges', 
                xticklabels=label_names, yticklabels=label_names,
                annot_kws={'size': 22}, cbar=False)  # Larger annotation font
    
    plt.title(f'Normalized Confusion Matrix - All Folds Combined', 
              fontsize=22, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=22, fontweight='bold')
    plt.ylabel('True Label', fontsize=22, fontweight='bold')
    
    # Rotate x-axis labels for better readability with larger fonts
    plt.xticks(rotation=45, ha='right', fontsize=22)
    plt.yticks(rotation=0, fontsize=22)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized_all_folds.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis(cm, accuracy, report, label_names, label_mapping, desired_order, english_translations):
    """Print detailed analysis results"""
    print("\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS - ALL FOLDS COMBINED")
    print("="*80)
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print("-" * 70)
    
    # Print confusion matrix with labels (Chinese and English)
    for chinese, english in zip(desired_order, english_translations):
        print(f"{chinese:<15}", end="")
    print()
    for english in english_translations:
        print(f"({english:<13})", end="")
    print()
    print("-" * 70)
    
    for i, (chinese, english) in enumerate(zip(desired_order, english_translations)):
        print(f"{chinese:<20}", end="")
        for j in range(len(desired_order)):
            print(f"{cm[i,j]:<15}", end="")
        print()
        print(f"({english:<18})", end="")
        print()
    
    print(f"\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 80)
    
    for chinese, english in zip(desired_order, english_translations):
        metrics = report[chinese]
        class_label = f"{chinese} ({english})"
        print(f"{class_label:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<12}")
    
    # Macro and weighted averages
    print("-" * 80)
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    
    print(f"{'Macro Avg':<25} {macro_avg['precision']:<12.4f} {macro_avg['recall']:<12.4f} "
          f"{macro_avg['f1-score']:<12.4f} {int(macro_avg['support']):<12}")
    print(f"{'Weighted Avg':<25} {weighted_avg['precision']:<12.4f} {weighted_avg['recall']:<12.4f} "
          f"{weighted_avg['f1-score']:<12.4f} {int(weighted_avg['support']):<12}")
    
    # Additional insights
    print(f"\nAdditional Insights:")
    print("-" * 50)
    
    # Class-wise accuracy
    for i, (chinese, english) in enumerate(zip(desired_order, english_translations)):
        class_accuracy = cm[i, i] / cm[i, :].sum()
        print(f"{chinese} ({english}) accuracy: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # Most confused classes
    print(f"\nMost Confused Class Pairs:")
    print("-" * 40)
    
    # Create a copy to avoid modifying the original
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # Remove diagonal for finding max confusion
    max_confusion_idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    max_confusion_count = cm_copy[max_confusion_idx]
    
    true_class = f"{desired_order[max_confusion_idx[0]]} ({english_translations[max_confusion_idx[0]]})"
    pred_class = f"{desired_order[max_confusion_idx[1]]} ({english_translations[max_confusion_idx[1]]})"
    
    print(f"'{true_class}' misclassified as '{pred_class}': {max_confusion_count} times")

def save_results_summary(cm, accuracy, report, label_names, total_samples, desired_order, english_translations):
    """Save analysis results to a text file"""
    with open('confusion_matrix_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("CONFUSION MATRIX ANALYSIS - ALL FOLDS COMBINED\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("Label Order: 偏綠 (pro-green) - 無明顯立場 (neutral) - 偏藍 (pro-blue)\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
        f.write("-" * 80 + "\n")
        
        for chinese, english in zip(desired_order, english_translations):
            metrics = report[chinese]
            class_label = f"{chinese} ({english})"
            f.write(f"{class_label:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<12}\n")
        
        f.write("-" * 80 + "\n")
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        
        f.write(f"{'Macro Avg':<25} {macro_avg['precision']:<12.4f} {macro_avg['recall']:<12.4f} "
               f"{macro_avg['f1-score']:<12.4f} {int(macro_avg['support']):<12}\n")
        f.write(f"{'Weighted Avg':<25} {weighted_avg['precision']:<12.4f} {weighted_avg['recall']:<12.4f} "
               f"{weighted_avg['f1-score']:<12.4f} {int(weighted_avg['support']):<12}\n")

def main():
    """Main analysis function"""
    print("Starting Confusion Matrix Analysis...")
    print("="*50)
    
    try:
        # Load and combine all predictions
        combined_df = load_and_combine_predictions()
        
        # Perform analysis
        cm, accuracy, report, label_names, label_mapping, desired_order, english_translations = analyze_predictions(combined_df)
        
        # Print detailed analysis
        print_detailed_analysis(cm, accuracy, report, label_names, label_mapping, desired_order, english_translations)
        
        # Create visualizations
        print(f"\nGenerating confusion matrix visualizations...")
        plot_confusion_matrix(cm, label_names, accuracy)
        plot_normalized_confusion_matrix(cm, label_names, accuracy)
        
        # Save summary
        save_results_summary(cm, accuracy, report, label_names, len(combined_df), desired_order, english_translations)
        
        print(f"\nAnalysis complete!")
        print(f"Files generated:")
        print(f"  - confusion_matrix_all_folds.png")
        print(f"  - confusion_matrix_normalized_all_folds.png") 
        print(f"  - confusion_matrix_analysis_summary.txt")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
