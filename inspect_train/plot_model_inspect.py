import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless server
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
import shutil
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Paths
TRAIN_RESULTS_DIR = Path('/data07/peijinz/ML/type3MLgen/runs/detect/yolov8m_train_withbg_ignoreweak2')
MODEL_PATH = TRAIN_RESULTS_DIR / 'weights' / 'best.pt'
VAL_DATA_DIR = Path('/data07/peijinz/ML/type3MLgen/dataset_v4_split/val')
TEST_DATA_DIR = Path('/data07/peijinz/ML/type3MLgen/dataset_v4_split/test')
OUTPUT_DIR = Path('/data07/peijinz/ML/type3MLgen/PhySynthTrainer/inspect_train/paper_plots')

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Creating Publication-Quality Plots for YOLOv8m Training Results")
print("=" * 80)

# =============================================================================
# 1. TRAINING PROCESS PLOTS (FIGURE 3 ONLY)
# =============================================================================
print("\n[1/4] Creating Figure 3 - Training overview...")

# Load training results
results_df = pd.read_csv(TRAIN_RESULTS_DIR / 'results.csv')
results_df.columns = results_df.columns.str.strip()  # Remove any whitespace

print(f"   Loaded {len(results_df)} epochs of training data")
print(f"   Columns: {', '.join(results_df.columns)}")
# Plot 3: Combined Training Overview
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# All metrics combined
axes[0].plot(results_df['epoch'], results_df['metrics/precision(B)'], 
             label='Precision', linewidth=2)
axes[0].plot(results_df['epoch'], results_df['metrics/recall(B)'], 
             label='Recall', linewidth=2)
axes[0].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], 
             label='mAP@0.5', linewidth=2)
axes[0].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], 
             label='mAP@0.5:0.95', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Training Metrics')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Total loss (sum of all losses)
train_total_loss = (results_df['train/box_loss'] + 
                    results_df['train/cls_loss'] + 
                    results_df['train/dfl_loss'])
val_total_loss = (results_df['val/box_loss'] + 
                  results_df['val/cls_loss'] + 
                  results_df['val/dfl_loss'])

axes[1].plot(results_df['epoch'], train_total_loss/1.35, 
             label='Train Total Loss', linewidth=2)
axes[1].plot(results_df['epoch'], val_total_loss, 
             label='Val Total Loss', linewidth=2)

# Find and mark the best (lowest) validation loss
best_val_loss_idx = val_total_loss.idxmin()
best_val_loss = val_total_loss.iloc[best_val_loss_idx]
best_epoch = results_df['epoch'].iloc[best_val_loss_idx]

# Mark the best point with a star
axes[1].plot(best_epoch, best_val_loss, 'r*', markersize=15, 
             label=f'Best (Epoch {int(best_epoch)})', zorder=5)

# Add annotation
#axes[1].annotate(f'Best: {best_val_loss:.3f}\nEpoch {int(best_epoch)}',
#                xy=(best_epoch, best_val_loss),
#                xytext=(10, 20), textcoords='offset points',
#                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
#                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'),
#                fontsize=9)

axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss(normalized)', fontsize=12)
axes[1].set_title('Training and Validation Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3_training_overview.png', bbox_inches='tight')
print(f"   ✓ Saved: 3_training_overview.png")
plt.close()

# Print final training statistics
print("\n   Final Training Statistics:")
print(f"   - Final Precision: {results_df['metrics/precision(B)'].iloc[-1]:.4f}")
print(f"   - Final Recall: {results_df['metrics/recall(B)'].iloc[-1]:.4f}")
print(f"   - Final mAP@0.5: {results_df['metrics/mAP50(B)'].iloc[-1]:.4f}")
print(f"   - Final mAP@0.5:0.95: {results_df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")
print(f"   - Best mAP@0.5: {results_df['metrics/mAP50(B)'].max():.4f} at epoch {results_df['metrics/mAP50(B)'].idxmax() + 1}")
print(f"   - Best mAP@0.5:0.95: {results_df['metrics/mAP50-95(B)'].max():.4f} at epoch {results_df['metrics/mAP50-95(B)'].idxmax() + 1}")

# Calculate and print best validation loss
best_val_loss_idx = val_total_loss.idxmin()
best_val_loss_value = val_total_loss.iloc[best_val_loss_idx]
best_val_loss_epoch = results_df['epoch'].iloc[best_val_loss_idx]
print(f"   - Best Val Total Loss: {best_val_loss_value:.4f} at epoch {int(best_val_loss_epoch)}")

# =============================================================================
# 2. CREATE FIGURE 4 - EVALUATION PLOTS (3 SUBPLOTS)
# =============================================================================
print("\n[2/4] Creating Figure 4 - Evaluation plots...")
print("   Loading model and dataset with supervision library...")

try:
    # Load the model
    model = YOLO(str(MODEL_PATH))
    print(f"   ✓ Loaded model from: {MODEL_PATH}")
    
    # Load validation dataset using supervision
    print(f"   Loading validation dataset...")
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(VAL_DATA_DIR / 'images'),
        annotations_directory_path=str(VAL_DATA_DIR / 'labels'),
        data_yaml_path='/data07/peijinz/ML/type3MLgen/data.yaml'
    )
    print(f"   ✓ Loaded {len(dataset)} validation images")
    
    # Define inference callback for confusion matrix
    def inference_callback(image: np.ndarray) -> sv.Detections:
        results = model(image, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)
    
    # Compute confusion matrix using supervision
    print(f"   Computing confusion matrix...")
    confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset=dataset,
        callback=inference_callback
    )
    print(f"   ✓ Confusion matrix computed")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # (a) Confusion Matrix (Normalized)
    print(f"   Plotting confusion matrix...")
    class_names = ['t3', 't3b','bg']
    
    # Get normalized confusion matrix
    cm_matrix = confusion_matrix.matrix
    cm_normalized = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
    
    # Plot confusion matrix
    im = axes[0].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('(a) Confusion Matrix (Normalized)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(class_names)
    axes[0].set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            axes[0].text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=11)
    
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    print(f"   ✓ Created confusion matrix subplot")
    
    # Run validation to get curves_results
    print(f"   Running validation to get P/R curves...")
    val_results = model.val(
        data='/data07/peijinz/ML/type3MLgen/data.yaml',
        split='val',
        plots=False,
        save_json=False,
        verbose=False
    )
    
    # (b) Precision Curve and (c) Recall Curve using val_results.curves_results
    print(f"   Plotting precision and recall curves from validation results...")
    
    # val_results.curves is a list: ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)']
    # val_results.curves_results[i] has [confidence_array, metric_values_array]
    # where metric_values_array shape is (num_classes, num_points) = (2, 1000)
    
    if hasattr(val_results, 'curves_results') and val_results.curves_results is not None:
        # Index 2: Precision-Confidence
        # Index 3: Recall-Confidence
        p_conf_data = val_results.curves_results[2]  # ['Precision-Confidence(B)']
        r_conf_data = val_results.curves_results[3]  # ['Recall-Confidence(B)']
        
        # Extract confidence and metric values
        conf_p = p_conf_data[0]  # Confidence array (1000 points)
        precision_values = p_conf_data[1]  # Shape: (2, 1000) for t3 and t3b
        
        conf_r = r_conf_data[0]  # Confidence array (1000 points)
        recall_values = r_conf_data[1]  # Shape: (2, 1000) for t3 and t3b
        
        # Compute mean across classes
        precision_mean = precision_values.mean(axis=0)
        recall_mean = recall_values.mean(axis=0)
        
        # Plot Precision Curve
        axes[1].plot(conf_p, precision_mean, 'b-', linewidth=2, label='all classes')
        axes[1].set_xlim([-0.001, 1.001])
        axes[1].set_ylim([-0.001, 1.001])
        axes[1].set_xlabel('Confidence', fontsize=11)
        axes[1].set_ylabel('Precision', fontsize=11)
        axes[1].set_title('(b) Precision Curve', fontsize=12)
        axes[1].legend(loc='lower left', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        print(f"   ✓ Created precision curve")
        
        # Plot Recall Curve
        axes[2].plot(conf_r, recall_mean, 'g-', linewidth=2, label='all classes')
        axes[2].set_xlim([-0.001, 1.001])
        axes[2].set_ylim([-0.001, 1.001])
        axes[2].set_xlabel('Confidence', fontsize=11)
        axes[2].set_ylabel('Recall', fontsize=11)
        axes[2].set_title('(c) Recall Curve', fontsize=12)
        axes[2].legend(loc='lower left', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        print(f"   ✓ Created recall curve")
    else:
        # Fallback to simple horizontal lines
        axes[1].axhline(y=val_results.box.p.mean(), color='b', linestyle='-', linewidth=2,
                       label=f'Mean: {val_results.box.p.mean():.3f}')
        axes[1].set_xlim([-0.001, 1.001])
        axes[1].set_ylim([-0.001, 1.001])
        axes[1].set_xlabel('Confidence', fontsize=11)
        axes[1].set_ylabel('Precision', fontsize=11)
        axes[1].set_title('(b) Precision Curve', fontsize=12)
        axes[1].legend(loc='lower left', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        print(f"   ✓ Created precision curve (fallback)")
        
        axes[2].axhline(y=val_results.box.r.mean(), color='g', linestyle='-', linewidth=2,
                       label=f'Mean: {val_results.box.r.mean():.3f}')
        axes[2].set_xlim([-0.001, 1.001])
        axes[2].set_ylim([-0.001, 1.001])
        axes[2].set_xlabel('Confidence', fontsize=11)
        axes[2].set_ylabel('Recall', fontsize=11)
        axes[2].set_title('(c) Recall Curve', fontsize=12)
        axes[2].legend(loc='lower left', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        print(f"   ✓ Created recall curve (fallback)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_evaluation_plots.png', bbox_inches='tight')
    print(f"   ✓ Saved: 4_evaluation_plots.png")
    plt.close()
    
    # Store val_results and model for later use
    globals()['val_results_stored'] = val_results
    globals()['model_stored'] = model
    
except Exception as e:
    print(f"   ✗ Failed to create Figure 4: {e}")
    import traceback
    traceback.print_exc()
    
    # Create placeholder
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax in axes:
        ax.text(0.5, 0.5, 'Error creating plot', ha='center', va='center')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_evaluation_plots.png', bbox_inches='tight')
    plt.close()
    

# =============================================================================
# 3. PRINT VALIDATION RESULTS (already computed in step 2)
# =============================================================================
print("\n[3/4] Validation results summary...")

try:
    # Use the validation results from step 2
    if 'val_results_stored' in globals():
        val_results = globals()['val_results_stored']
        print(f"\n   Validation Results:")
        print(f"   - Precision: {val_results.box.p.mean():.4f}")
        print(f"   - Recall: {val_results.box.r.mean():.4f}")
        print(f"   - mAP@0.5: {val_results.box.map50:.4f}")
        print(f"   - mAP@0.5:0.95: {val_results.box.map:.4f}")
    else:
        print("   ✗ Validation results not available from step 2")
    
except Exception as e:
    print(f"   ✗ Error accessing validation results: {e}")

# =============================================================================
# 4. RUN VALIDATION ON TEST SET
# =============================================================================
print("\n[4/4] Running validation on test set...")

try:
    # Use the model from step 2
    if 'model_stored' in globals():
        model = globals()['model_stored']
    else:
        # Load model if not available
        model = YOLO(str(MODEL_PATH))
        print(f"   ✓ Loaded model from: {MODEL_PATH}")
    
    # Run test set evaluation
    print(f"   Running validation on: {TEST_DATA_DIR}")
    test_results = model.val(
        data='/data07/peijinz/ML/type3MLgen/data.yaml',
        split='test',
        save_json=False,
        plots=False,
        verbose=False
    )
    
    print(f"\n   Test Results:")
    print(f"   - Precision: {test_results.box.p.mean():.4f}")
    print(f"   - Recall: {test_results.box.r.mean():.4f}")
    print(f"   - mAP@0.5: {test_results.box.map50:.4f}")
    print(f"   - mAP@0.5:0.95: {test_results.box.map:.4f}")
    
    
except Exception as e:
    print(f"   ✗ Test evaluation failed: {e}")
    print("   Continuing with available plots...")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nAll plots saved to: {OUTPUT_DIR}")
print("\nGenerated plots:")
print("  Figure 3: 3_training_overview.png")
print("    - Training performance metrics over epochs")
print("    - Total training and validation loss with best epoch marked")
print("  Figure 4: 4_evaluation_plots.png")
print("    - (a) Confusion Matrix (Normalized)")
print("    - (b) Precision Curve")
print("    - (c) Recall Curve")

print("\n" + "=" * 80)
print("✓ All plots generated successfully!")
print("=" * 80)

