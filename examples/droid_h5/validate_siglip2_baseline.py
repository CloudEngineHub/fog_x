#!/usr/bin/env python3
"""
Simple validation script specifically for SigLIP-2 baseline results.
Generates confusion matrix and accuracy metrics.
"""

import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import argparse
import os


def create_confusion_matrix_display(cm, labels=None):
    """Create a simple text-based confusion matrix display."""
    if labels is None:
        labels = ['Success', 'Failure']
    
    print("\n📊 Confusion Matrix:")
    print("=" * 35)
    print(f"{'':>10} {'Predicted':>20}")
    print(f"{'Actual':>10} {'Success':>10} {'Failure':>10}")
    print("-" * 35)
    print(f"{'Success':>10} {cm[1][1]:>10} {cm[1][0]:>10}")  # True=1, Predicted=1 vs Predicted=0
    print(f"{'Failure':>10} {cm[0][1]:>10} {cm[0][0]:>10}")  # True=0, Predicted=1 vs Predicted=0
    print("-" * 35)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📈 Detailed Breakdown:")
    print(f"   True Positives (TP):  {tp:>3} - Correctly predicted failures")
    print(f"   True Negatives (TN):  {tn:>3} - Correctly predicted successes") 
    print(f"   False Positives (FP): {fp:>3} - Incorrectly predicted as failures")
    print(f"   False Negatives (FN): {fn:>3} - Incorrectly predicted as successes")
    
    return tn, fp, fn, tp


def validate_siglip2_predictions(predictions_file: str, ground_truth_file: str):
    """
    Validate SigLIP-2 predictions against ground truth and generate confusion matrix.
    """
    
    print(f"🔍 Validating SigLIP-2 Baseline Predictions")
    print("=" * 45)
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth  
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"📂 Loaded {len(predictions)} predictions")
    print(f"📂 Loaded {len(ground_truth)} ground truth labels")
    
    # Align predictions with ground truth
    y_true = []  # Ground truth labels (True=Success, False=Failure)
    y_pred = []  # Predicted labels
    trajectory_names = []
    
    matched_count = 0
    
    for traj_path in predictions.keys():
        if traj_path in ground_truth:
            # Ground truth: True=Success, False=Failure
            true_label = ground_truth[traj_path]
            
            # Prediction: success field indicates the prediction  
            pred_success = predictions[traj_path]['success']
            
            y_true.append(true_label)
            y_pred.append(pred_success)
            trajectory_names.append(os.path.basename(traj_path))
            matched_count += 1
    
    if matched_count == 0:
        print("❌ No matching trajectories found between predictions and ground truth!")
        return
    
    print(f"✅ Matched {matched_count} trajectories for validation")
    
    # Convert to numpy arrays for sklearn
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print(f"\n📊 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Generate confusion matrix
    # Note: sklearn uses [0, 1] where 0=False (Failure), 1=True (Success)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = create_confusion_matrix_display(cm)
    
    # Calculate per-class metrics
    print(f"\n📈 Performance Metrics:")
    print(f"   Overall Accuracy:  {accuracy:.3f}")
    print(f"   Precision:         {precision:.3f} (of predicted failures, how many were correct)")
    print(f"   Recall:           {recall:.3f} (of actual failures, how many were caught)")
    print(f"   F1-Score:         {f1:.3f} (harmonic mean of precision & recall)")
    
    # Success/Failure specific metrics
    success_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    success_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    failure_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    failure_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n🎯 Class-Specific Performance:")
    print(f"   Success Prediction:")
    print(f"     Precision: {success_precision:.3f}")
    print(f"     Recall:    {success_recall:.3f}")
    print(f"   Failure Prediction:")
    print(f"     Precision: {failure_precision:.3f}")
    print(f"     Recall:    {failure_recall:.3f}")
    
    # Analyze some specific examples
    print(f"\n🔍 Example Analysis:")
    
    # Show some true positives (correctly identified failures)
    tp_indices = np.where((y_true == False) & (y_pred == False))[0]
    if len(tp_indices) > 0:
        print(f"   ✅ Correctly identified failures (examples):")
        for i in tp_indices[:3]:
            traj_name = trajectory_names[i]
            similarity_score = predictions[list(predictions.keys())[i]]['similarity_score']
            print(f"      {traj_name}: similarity={similarity_score:.4f}")
    
    # Show some false positives (incorrectly predicted as failures)
    fp_indices = np.where((y_true == True) & (y_pred == False))[0]
    if len(fp_indices) > 0:
        print(f"   ❌ False alarms (predicted failure, actually success):")
        for i in fp_indices[:3]:
            traj_name = trajectory_names[i]
            similarity_score = predictions[list(predictions.keys())[i]]['similarity_score']
            print(f"      {traj_name}: similarity={similarity_score:.4f}")
    
    # Show some false negatives (missed failures)
    fn_indices = np.where((y_true == False) & (y_pred == True))[0]
    if len(fn_indices) > 0:
        print(f"   📉 Missed failures (predicted success, actually failure):")
        for i in fn_indices[:3]:
            traj_name = trajectory_names[i]
            similarity_score = predictions[list(predictions.keys())[i]]['similarity_score']
            print(f"      {traj_name}: similarity={similarity_score:.4f}")
    
    # Summary statistics
    print(f"\n📊 Dataset Summary:")
    print(f"   Total trajectories: {len(y_true)}")
    print(f"   Actual successes: {np.sum(y_true)}")
    print(f"   Actual failures: {np.sum(~y_true)}")
    print(f"   Predicted successes: {np.sum(y_pred)}")
    print(f"   Predicted failures: {np.sum(~y_pred)}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description="Validate SigLIP-2 baseline predictions")
    parser.add_argument(
        "--predictions", 
        default="siglip2_baseline_output/siglip2_baseline_predictions.json",
        help="Path to SigLIP-2 predictions JSON file"
    )
    parser.add_argument(
        "--ground-truth",
        default="siglip2_baseline_output/generated_ground_truth.json", 
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output",
        help="Optional output file for metrics JSON"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions):
        print(f"❌ Predictions file not found: {args.predictions}")
        return 1
        
    if not os.path.exists(args.ground_truth):
        print(f"❌ Ground truth file not found: {args.ground_truth}")
        return 1
    
    try:
        metrics = validate_siglip2_predictions(args.predictions, args.ground_truth)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n💾 Metrics saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())