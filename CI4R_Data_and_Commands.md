# CI4R Dataset - Data Preparation and Training Guide

## Dataset Overview

### CI4R (Cross-frequency Indoor 4 Radar) Dataset
- **Type**: Real 77 GHz FMCW radar data for human activity recognition
- **Classes**: 11 different human activities
- **Radar Type**: Frequency-Modulated Continuous Wave (FMCW)
- **Frequency Band**: 77 GHz (millimeter wave)
- **Application**: Indoor human activity recognition using micro-Doppler signatures

### Activity Classes
1. Walking Towards
2. Walking Away
3. Picking Object
4. Bending
5. Sitting
6. Kneeling
7. Crawling
8. Walking on Toes
9. Limping
10. Short Steps
11. Scissors Gait

### Dataset Statistics
- **Training Samples**: 520
- **Test Samples**: 132
- **Total Samples**: 652
- **Input Size**: 224×224 (after preprocessing)
- **Data Format**: HDF5 (.h5)

## Data Formats Available

### Format 1: MATLAB Files (.mat)
**Location**: `datasets/CI4R/Cross-frequency/77ghz/activity_spect_matrices/`

**Description**: Raw complex STFT (Short-Time Fourier Transform) spectrograms stored in MATLAB format.

**Structure**:
- Each `.mat` file contains a complex spectrogram matrix
- Key: `'sx1'` - Complex-valued array (4096×538)
- Data represents time-frequency representation of radar reflections
- Contains both magnitude and phase information

**Processing Steps**:
1. Load complex STFT matrix from .mat file
2. Convert to magnitude: `magnitude = |complex_signal|`
3. Convert to dB scale: `spectrogram_db = 20 * log10(magnitude + ε)`
4. Clip values: `clip(-45, max_value)`
5. Resize to 224×224 using cubic interpolation
6. Normalize per sample: `(x - μ) / (σ + ε)`

**Advantages**:
- Raw signal data with full control over preprocessing
- Preserves original complex signal information
- Allows custom signal processing pipelines

### Format 2: PNG Spectrograms
**Location**: `datasets/CI4R/Cross-frequency/Spectograms_77_24_Xethrue/activity_spectogram_77GHz/`

**Description**: Pre-rendered spectrogram images in PNG format.

**Structure**:
- Pre-computed time-frequency representations
- Already visualized as grayscale images
- Ready for direct image processing

**Processing Steps**:
1. Load PNG image as grayscale
2. Resize to 224×224 if needed
3. Normalize to [0, 1]: `pixel_value / 255.0`
4. Standardize: `(x - μ) / (σ + ε)`

**Advantages**:
- Faster loading (no complex signal processing needed)
- Pre-visualized data
- Smaller file size


## Command Sequences

### **Approach 1: Using .mat Files (Recommended - Tested)**

This approach processes raw MATLAB files containing complex spectrograms.

#### Step 1: Data Preparation
```bash
cd /Users/parthbhalodiya/Downloads/C781/AIRHAR
python3 datasets/prepare_ci4r_from_mat.py
```

**Output**:
- File: `datasets/CI4R/ci4r_data.h5`
- Train samples: 520
- Test samples: 132
- Shape: (224, 224) per sample

#### Step 2: Train RadMamba Model
```bash
python3 main.py \
  --dataset_name CI4R \
  --Classification_backbone radmamba \
  --Classification_hidden_size 64 \
  --dim 64 \
  --n_epochs 300 \
  --batch_size 16 \
  --lr 0.001 \
  --seed 0
```

**Training Details**:
- Model: RadMamba with bidirectional SSM
- Parameters: 47,865
- Best accuracy achieved: **84.09%** (epoch 24)
- Hardware: CPU (Mac compatible)

**Output**:
- Model checkpoint: `save/CI4R/classify/CL_S_0_M_RADMAMBA_B_16_LR_0.0010_H_64_P_47865_FL_224_ST_1.pt`
- Training logs in console

#### Step 3: Generate Animated Visualization
```bash
python3 animate_pipeline.py
```

**Output**:
- Animated GIF: `visualization_results/pipeline_animation.gif`
- Static frames: `visualization_results/pipeline_frames/frame_*.png`
- 55 frames total (5 samples per class)
- Shows: Raw signal → Spectrogram → RadMamba → Classification


### **Approach 2: Using PNG Spectrograms**

This approach processes pre-rendered spectrogram images.

#### Step 1: Data Preparation
```bash
cd /Users/parthbhalodiya/Downloads/C781/AIRHAR
python3 datasets/prepare_ci4r_from_spectrograms.py
```

**Output**:
- File: `datasets/CI4R/ci4r_data.h5`
- Automatically splits train/test

#### Step 2: Train RadMamba Model
```bash
python3 main.py \
  --dataset_name CI4R \
  --Classification_backbone radmamba \
  --Classification_hidden_size 64 \
  --dim 64 \
  --n_epochs 300 \
  --batch_size 16 \
  --lr 0.001 \
  --seed 0
```

#### Step 3: Generate Animated Visualization
```bash
python3 animate_pipeline.py
```

**Note**: Steps 2 and 3 are identical for both approaches since they operate on the same H5 format.


## Key Differences Between Approaches

| Aspect | .mat Files | PNG Spectrograms |
|--------|-----------|------------------|
| **Input Format** | Complex STFT matrices | Pre-rendered images |
| **File Size** | Larger (contains complex data) | Smaller (grayscale images) |
| **Processing** | Full signal processing pipeline | Simple image loading |
| **Control** | Full control over signal processing | Limited to image operations |
| **Speed** | Slower (complex math operations) | Faster (direct image loading) |
| **Data Fidelity** | Original complex signal preserved | Visual representation only |
| **Status** | ✅ Tested - 84.09% accuracy | ⚠️ Not tested yet |


## Understanding the Pipeline

### 1. Raw Radar Signal
- **Type**: Complex IQ (In-phase/Quadrature) data
- **Source**: 77 GHz FMCW radar reflections
- **Content**: Micro-Doppler signatures from human movements
- **Size**: Variable (4096×538 in .mat files)

### 2. Spectrogram Generation
- **Method**: Short-Time Fourier Transform (STFT)
- **Output**: Time-frequency representation
- **Features**: Shows Doppler frequency shifts over time
- **Micro-Doppler**: Unique patterns for different body movements

### 3. RadMamba Model Architecture
- **Input**: 224×224 normalized spectrogram
- **Channel Confusion Layer**: Cross-channel feature mixing
- **Patch Embedding**: Converts image to sequence
- **RadMamba Block**: Bidirectional State Space Model (SSM)
  - Forward SSM (→): Left-to-right scanning
  - Backward SSM (←): Right-to-left scanning
  - Captures temporal dependencies in both directions
- **Output**: 11-class probability distribution

### 4. Classification Output
- **Prediction**: Highest probability class
- **Confidence**: Softmax probability score
- **Evaluation**: Compared against ground truth label

## Performance Results

### Model Performance (Using .mat Approach)
- **Test Accuracy**: 84.09%
- **Correct Predictions**: 9 out of 11 classes show high accuracy
- **Model Size**: 47,865 parameters
- **Training Time**: ~300 epochs to convergence

### Classes with High Accuracy
- Walking Away
- Kneeling
- Sitting
- Bending
- Walking Towards
- Picking Object
- Short Steps
- Limping
- Crawling

### Classes Needing Improvement
- Walking on Toes
- Scissors Gait


## Hyperparameters Explanation

### Training Hyperparameters
- `--n_epochs 300`: Number of training epochs (increased from 100 to prevent early stopping)
- `--batch_size 16`: Samples per batch (increased from 8 for stability)
- `--lr 0.001`: Learning rate (reduced from 0.005 to prevent overfitting)
- `--seed 0`: Random seed for reproducibility

### Model Hyperparameters
- `--Classification_hidden_size 64`: Hidden dimension size
- `--dim 64`: SSM state dimension
- `--Classification_backbone radmamba`: Model architecture type
- `d_state=4`: State space dimension (in config)
- `dt_rank=0`: Delta rank for time step computation


## File Structure

```
AIRHAR/
├── datasets/
│   ├── CI4R/
│   │   ├── ci4r_data.h5                    # Processed dataset (output)
│   │   ├── spec.json                        # Configuration file
│   │   └── Cross-frequency/
│   │       ├── 77ghz/
│   │       │   └── activity_spect_matrices/ # .mat files (raw data)
│   │       └── Spectograms_77_24_Xethrue/
│   │           └── activity_spectogram_77GHz/ # PNG files (alternative)
│   ├── prepare_ci4r_from_mat.py            # Data prep script (Approach 1)
│   └── prepare_ci4r_from_spectrograms.py   # Data prep script (Approach 2)
├── save/
│   └── CI4R/
│       └── classify/
│           └── CL_S_0_M_RADMAMBA_*.pt      # Trained model checkpoint
├── visualization_results/
│   ├── pipeline_animation.gif               # Animated visualization
│   └── pipeline_frames/                     # Individual frames
├── main.py                                  # Training script
├── animate_pipeline.py                      # Visualization script
└── CI4R_Performance_Summary.md              # Detailed performance report
```


## Quick Start Guide

### Minimal Command Sequence (.mat approach)
```bash
# Navigate to project directory
cd /Users/parthbhalodiya/Downloads/C781/AIRHAR

# 1. Prepare data
python3 datasets/prepare_ci4r_from_mat.py

# 2. Train model (this will take time)
python3 main.py --dataset_name CI4R --Classification_backbone radmamba \
  --Classification_hidden_size 64 --dim 64 --n_epochs 300 \
  --batch_size 16 --lr 0.001 --seed 0

# 3. Generate visualization
python3 animate_pipeline.py
```

### Output Files
- `datasets/CI4R/ci4r_data.h5` - Processed training/test data
- `save/CI4R/classify/CL_S_0_M_RADMAMBA_*.pt` - Trained model
- `visualization_results/pipeline_animation.gif` - Visual demo


## Troubleshooting

### Common Issues

**Issue**: CUDA not available error
- **Solution**: Set `"accelerator": "cpu"` in `datasets/CI4R/spec.json`

**Issue**: Model overfitting (accuracy drops after initial epochs)
- **Solution**: Lower learning rate (0.001) and increase batch size (16)

**Issue**: `.mat` files not loading
- **Solution**: Ensure scipy is installed: `pip install scipy`

**Issue**: Animation generation fails
- **Solution**: Ensure `radmamba_architecture.png` exists in the project root


## References

- **Dataset**: CI4R Cross-frequency Indoor Radar Dataset
- **Model**: RadMamba - Radar-based Mamba with bidirectional SSM
- **Paper**: [RadMamba Paper](https://arxiv.org/pdf/2504.12039)
- **Framework**: PyTorch Lightning

## Notes

1. The .mat file approach is recommended as it has been thoroughly tested
2. Training on CPU takes longer but is stable (Mac compatible)
3. The animation shows balanced class representation (5 samples per class)
4. 84.09% accuracy represents strong performance on real radar data
5. The visualization pipeline demonstrates the complete end-to-end process


Last Updated: 2026-01-08
