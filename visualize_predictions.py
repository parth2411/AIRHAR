import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add current directory to path to import models
sys.path.insert(0, '.')

from models import CoreModel

def load_model_and_data(model_path, data_path, config_path):
    """Load the trained model and test data"""
    # Load config first to get model parameters
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model architecture
    model = CoreModel(
        hidden_size=config['Classification_hidden_size'],
        num_layers=config['Classification_num_layers'],
        backbone_type=config['Classification_backbone'],
        dim=64,  # From training config
        dt_rank=config.get('dt_rank', 0),
        d_state=config.get('d_state', 4),
        image_height=config['image_height'],
        image_width=config['frame_length'],
        num_classes=config['num_classes'],
        channels=config['channels'],
        dropout=config.get('dropout', 0),
        optional_avg_pool=True,
        channel_confusion_layer=config.get('channel_confusion_layer', 1),
        channel_confusion_out_channels=config.get('channel_confusion_out_channels', 1),
        time_downsample_factor=config.get('time_downsample_factor', 4)
    )

    # Load model weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # Load test data
    with h5py.File(data_path, 'r') as f:
        X_test = f['test/spectrograms'][:]
        y_test = f['test/labels'][:]

    return model, X_test, y_test, config

def get_class_names():
    """CI4R activity class names"""
    # Based on common radar activity recognition datasets
    # You may need to adjust these based on actual CI4R classes
    return [
        'Class_0',
        'Class_1',
        'Class_2',
        'Class_3',
        'Class_4',
        'Class_5',
        'Class_6',
        'Class_7',
        'Class_8',
        'Class_9',
        'Class_10'
    ]

def predict_sample(model, spectrogram):
    """Run inference on a single spectrogram"""
    # Add batch dimension only
    # Shape: (224, 224) -> (1, 224, 224)
    # Model will add channel dimension automatically
    x = torch.FloatTensor(spectrogram).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence, probabilities[0].cpu().numpy()

def visualize_predictions(model, X_test, y_test, num_samples=6, save_path='predictions.png'):
    """Visualize model predictions on test samples"""
    class_names = get_class_names()

    # Randomly select samples
    indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(indices):
        if idx >= len(axes):
            break

        # Get data
        spectrogram = X_test[sample_idx]
        true_label = np.argmax(y_test[sample_idx])

        # Predict
        pred_label, confidence, probabilities = predict_sample(model, spectrogram)

        # Plot spectrogram
        ax = axes[idx]
        im = ax.imshow(spectrogram, cmap='viridis', aspect='auto')

        # Title with prediction info
        correct = pred_label == true_label
        color = 'green' if correct else 'red'
        title = f'True: {class_names[true_label]}\n'
        title += f'Pred: {class_names[pred_label]} ({confidence*100:.1f}%)'
        ax.set_title(title, color=color, fontweight='bold')

        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()

def visualize_single_detailed(model, X_test, y_test, sample_idx=0, save_path='detailed_prediction.png'):
    """Detailed visualization of a single prediction with probability distribution"""
    class_names = get_class_names()

    # Get data
    spectrogram = X_test[sample_idx]
    true_label = np.argmax(y_test[sample_idx])

    # Predict
    pred_label, confidence, probabilities = predict_sample(model, spectrogram)

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 6))

    # Left: Spectrogram
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(spectrogram, cmap='viridis', aspect='auto')
    correct = pred_label == true_label
    color = 'green' if correct else 'red'
    title = f'Sample {sample_idx}\n'
    title += f'True: {class_names[true_label]} | Pred: {class_names[pred_label]}'
    ax1.set_title(title, color=color, fontweight='bold', fontsize=12)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax1)

    # Right: Probability distribution
    ax2 = plt.subplot(1, 2, 2)
    colors = ['green' if i == true_label else 'blue' for i in range(len(class_names))]
    colors[pred_label] = 'red' if pred_label != true_label else 'green'

    bars = ax2.barh(class_names, probabilities * 100, color=colors, alpha=0.7)
    ax2.set_xlabel('Confidence (%)', fontsize=11)
    ax2.set_ylabel('Activity Class', fontsize=11)
    ax2.set_title('Prediction Probabilities', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        if prob > 0.01:  # Only show if > 1%
            ax2.text(prob * 100 + 1, i, f'{prob*100:.1f}%',
                    va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='True Label'),
        Patch(facecolor='red', alpha=0.7, label='Predicted (if wrong)'),
        Patch(facecolor='blue', alpha=0.7, label='Other Classes')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed visualization to {save_path}")
    plt.close()

def evaluate_test_set(model, X_test, y_test):
    """Evaluate model on entire test set and show confusion patterns"""
    class_names = get_class_names()
    predictions = []
    true_labels = []

    print("\nEvaluating on test set...")
    for i in range(len(X_test)):
        pred_label, _, _ = predict_sample(model, X_test[i])
        true_label = np.argmax(y_test[i])
        predictions.append(pred_label)
        true_labels.append(true_label)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate accuracy
    accuracy = np.mean(predictions == true_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for class_idx in range(len(class_names)):
        mask = true_labels == class_idx
        if mask.sum() > 0:
            class_acc = np.mean(predictions[mask] == true_labels[mask]) * 100
            print(f"  {class_names[class_idx]}: {class_acc:.1f}% ({mask.sum()} samples)")

    return predictions, true_labels, accuracy

def plot_confusion_matrix(predictions, true_labels, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    class_names = get_class_names()
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - RadMamba on CI4R', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Paths
    model_path = "save/CI4R/classify/CL_S_0_M_RADMAMBA_B_16_LR_0.0050_H_64_P_47865_FL_224_ST_1.pt"
    data_path = "datasets/CI4R/ci4r_data.h5"
    config_path = "datasets/CI4R/spec.json"

    print("Loading model and data...")
    model, X_test, y_test, config = load_model_and_data(model_path, data_path, config_path)

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(get_class_names())}")

    # Create output directory
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)

    # 1. Visualize multiple predictions
    print("\n1. Creating grid visualization...")
    visualize_predictions(model, X_test, y_test, num_samples=6,
                         save_path=output_dir / "predictions_grid.png")

    # 2. Detailed single prediction
    print("\n2. Creating detailed visualization...")
    visualize_single_detailed(model, X_test, y_test, sample_idx=0,
                             save_path=output_dir / "detailed_prediction_0.png")

    # 3. Evaluate entire test set
    print("\n3. Evaluating entire test set...")
    predictions, true_labels, accuracy = evaluate_test_set(model, X_test, y_test)

    # 4. Confusion matrix
    print("\n4. Creating confusion matrix...")
    plot_confusion_matrix(predictions, true_labels,
                         save_path=output_dir / "confusion_matrix.png")

    print(f"\n✓ All visualizations saved to {output_dir}/")
    print(f"  - predictions_grid.png (6 random samples)")
    print(f"  - detailed_prediction_0.png (detailed view with probabilities)")
    print(f"  - confusion_matrix.png (full confusion matrix)")
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
