# import torch
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
# from matplotlib.gridspec import GridSpec
# import scipy.io
# import cv2
# import json
# import sys
# from pathlib import Path

# sys.path.insert(0, '.')
# from models import CoreModel

# class RadarPipelineAnimator:
#     def __init__(self, model_path, config_path, mat_files_dir):
#         """Initialize the pipeline animator"""
#         self.model, self.config = self.load_model(model_path, config_path)
#         self.mat_files_dir = Path(mat_files_dir)
#         self.mat_files = self.collect_mat_files()
#         self.current_idx = 0

#         # Activity names
#         self.activity_names = [
#             'Walking Towards',
#             'Walking Away',
#             'Picking Object',
#             'Bending',
#             'Sitting',
#             'Kneeling',
#             'Crawling',
#             'Walking on Toes',
#             'Limping',
#             'Short Steps',
#             'Scissors Gait'
#         ]

#     def load_model(self, model_path, config_path):
#         """Load trained RadMamba model"""
#         with open(config_path, 'r') as f:
#             config = json.load(f)

#         model = CoreModel(
#             hidden_size=config['Classification_hidden_size'],
#             num_layers=config['Classification_num_layers'],
#             backbone_type=config['Classification_backbone'],
#             dim=64,
#             dt_rank=config.get('dt_rank', 0),
#             d_state=config.get('d_state', 4),
#             image_height=config['image_height'],
#             image_width=config['frame_length'],
#             num_classes=config['num_classes'],
#             channels=config['channels'],
#             dropout=config.get('dropout', 0),
#             optional_avg_pool=True,
#             channel_confusion_layer=config.get('channel_confusion_layer', 1),
#             channel_confusion_out_channels=config.get('channel_confusion_out_channels', 1),
#             time_downsample_factor=config.get('time_downsample_factor', 4)
#         )

#         state_dict = torch.load(model_path, map_location='cpu')
#         model.load_state_dict(state_dict)
#         model.eval()

#         return model, config

#     def collect_mat_files(self):
#         """Collect 5 samples from each of the 11 classes"""
#         print("Collecting 5 samples from each class...")

#         selected_samples = []

#         # Get samples from each activity directory
#         actual_class_idx = 0
#         for activity_dir in sorted(self.mat_files_dir.iterdir()):
#             if activity_dir.is_dir():
#                 print(f"\n  Analyzing Class {actual_class_idx}: {activity_dir.name}...")
#                 files = list(activity_dir.glob('*.mat'))

#                 class_samples = []  # Store all predictions for this class

#                 # Search through ALL files
#                 for i, mat_file in enumerate(files):
#                     try:
#                         # Load and test prediction
#                         raw_signal = self.load_radar_signal(mat_file)
#                         if raw_signal is None:
#                             continue

#                         spectrogram = self.process_to_spectrogram(raw_signal)
#                         pred_class, confidence, _ = self.predict(spectrogram)

#                         is_correct = (pred_class == actual_class_idx)

#                         # Store all samples with their correctness and confidence
#                         class_samples.append((mat_file, actual_class_idx, confidence, is_correct))

#                     except Exception as e:
#                         continue

#                 if len(class_samples) > 0:
#                     # Sort by correctness first (correct=True comes first), then by confidence
#                     class_samples.sort(key=lambda x: (not x[3], -x[2]))

#                     # Take top 5 samples from this class
#                     top_5 = class_samples[:5]

#                     correct_count = sum(1 for _, _, _, is_correct in top_5 if is_correct)
#                     print(f"    Selected {len(top_5)} samples ({correct_count} correct)")

#                     # Add to selected samples
#                     for mat, cls, conf, is_correct in top_5:
#                         selected_samples.append((mat, cls))

#                 actual_class_idx += 1

#         total_correct = sum(1 for mat, cls in selected_samples
#                           if self.is_prediction_correct(mat, cls))
#         print(f"\n✓ Selected {len(selected_samples)} total samples")
#         print(f"  {total_correct}/{len(selected_samples)} predictions are correct\n")

#         return selected_samples

#     def is_prediction_correct(self, mat_file, true_class):
#         """Helper to check if a prediction is correct"""
#         try:
#             raw_signal = self.load_radar_signal(mat_file)
#             if raw_signal is None:
#                 return False
#             spectrogram = self.process_to_spectrogram(raw_signal)
#             pred_class, _, _ = self.predict(spectrogram)
#             return pred_class == true_class
#         except:
#             return False

#     def load_radar_signal(self, mat_path):
#         """Load raw complex radar signal from .mat file"""
#         mat_data = scipy.io.loadmat(str(mat_path))
#         if 'sx1' in mat_data:
#             return mat_data['sx1']
#         return None

#     def process_to_spectrogram(self, complex_signal):
#         """Convert complex signal to spectrogram"""
#         # Convert to magnitude in dB
#         magnitude = np.abs(complex_signal)
#         spectrogram_db = 20 * np.log10(magnitude + 1e-10)
#         spectrogram_db = np.clip(spectrogram_db, -45, spectrogram_db.max())

#         # Resize to 224x224
#         spectrogram = cv2.resize(spectrogram_db.astype(np.float32),
#                                 (224, 224),
#                                 interpolation=cv2.INTER_CUBIC)

#         # Normalize
#         spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)

#         return spectrogram

#     def predict(self, spectrogram):
#         """Run inference through RadMamba"""
#         x = torch.FloatTensor(spectrogram).unsqueeze(0)

#         with torch.no_grad():
#             logits = self.model(x)
#             probabilities = torch.softmax(logits, dim=1)
#             predicted_class = torch.argmax(probabilities, dim=1).item()
#             confidence = probabilities[0, predicted_class].item()

#         return predicted_class, confidence, probabilities[0].cpu().numpy()
    
#     def draw_radmamba_architecture(self, ax):
#         """Draw clean RadMamba architecture diagram with EXTRA LARGE components"""
#         ax.clear()
#         ax.set_xlim(0, 20)  # Much wider canvas
#         ax.set_ylim(0, 13)  # Much taller canvas
#         ax.axis('off')
#         ax.set_title('RadMamba Architecture', fontsize=26, fontweight='bold', pad=25)

#         # EXTRA SPACIOUS pipeline - horizontal layout with LOTS of spacing
#         y_pos = 8  # Main horizontal line

#         # Layer 1: Input - MUCH BIGGER
#         x1 = 1.8
#         box1 = FancyBboxPatch((x1-1.2, y_pos-1.0), 2.4, 2.0,
#                              boxstyle="round,pad=0.15", facecolor='lightblue',
#                              edgecolor='black', linewidth=3.5)
#         ax.add_patch(box1)
#         ax.text(x1, y_pos, 'Input\n224×224', ha='center', va='center',
#                fontsize=13, fontweight='bold')

#         # Arrow 1->2 - MUCH THICKER and LONGER
#         ax.arrow(x1+1.2, y_pos, 1.0, 0, head_width=0.5, head_length=0.3,
#                 fc='black', ec='black', linewidth=4)

#         # Layer 2: Channel Confusion - MUCH BIGGER
#         x2 = 5.0
#         box2 = FancyBboxPatch((x2-1.3, y_pos-1.0), 2.6, 2.0,
#                              boxstyle="round,pad=0.15", facecolor='lightgreen',
#                              edgecolor='black', linewidth=3.5)
#         ax.add_patch(box2)
#         ax.text(x2, y_pos, 'Channel\nConfusion', ha='center', va='center',
#                fontsize=13, fontweight='bold')

#         # Arrow 2->3 - THICKER
#         ax.arrow(x2+1.3, y_pos, 1.0, 0, head_width=0.5, head_length=0.3,
#                 fc='black', ec='black', linewidth=4)

#         # Layer 3: Patch Embed - MUCH BIGGER
#         x3 = 8.5
#         box3 = FancyBboxPatch((x3-1.3, y_pos-1.0), 2.6, 2.0,
#                              boxstyle="round,pad=0.15", facecolor='lightcoral',
#                              edgecolor='black', linewidth=3.5)
#         ax.add_patch(box3)
#         ax.text(x3, y_pos, 'Patch\nEmbedding', ha='center', va='center',
#                fontsize=13, fontweight='bold')

#         # Arrow 3->4 - THICKER
#         ax.arrow(x3+1.3, y_pos, 1.2, 0, head_width=0.5, head_length=0.3,
#                 fc='black', ec='black', linewidth=4)

#         # Layer 4: RadMamba Block - EXTRA LARGE
#         x4 = 13.0
#         mamba_box = FancyBboxPatch((x4-2.2, y_pos-1.8), 4.4, 3.6,
#                                   boxstyle="round,pad=0.15", facecolor='#F3E5F5',
#                                   edgecolor='purple', linewidth=4)
#         ax.add_patch(mamba_box)
#         ax.text(x4, y_pos+1.4, 'RadMamba Block', ha='center', va='center',
#                fontsize=13, fontweight='bold', color='purple')

#         # Bidirectional SSM inside - MUCH BIGGER
#         ax.text(x4-1.0, y_pos+0.3, '→ SSM', ha='center', va='center',
#                fontsize=13, fontweight='bold', color='green',
#                bbox=dict(boxstyle='round,pad=0.5', facecolor='#B8E6B8',
#                         edgecolor='green', linewidth=2.5))
#         ax.text(x4+1.0, y_pos+0.3, 'SSM ←', ha='center', va='center',
#                fontsize=13, fontweight='bold', color='blue',
#                bbox=dict(boxstyle='round,pad=0.5', facecolor='#B8D8F0',
#                         edgecolor='blue', linewidth=2.5))
#         ax.text(x4, y_pos-0.8, 'Bidirectional', ha='center', va='center',
#                fontsize=13, style='italic', color='purple', fontweight='bold')

#         # Arrow 4->5 - THICKER
#         ax.arrow(x4+2.2, y_pos, 1.0, 0, head_width=0.5, head_length=0.3,
#                 fc='black', ec='black', linewidth=4)

#         # Layer 5: Output - MUCH BIGGER
#         x5 = 17.5
#         box5 = FancyBboxPatch((x5-1.4, y_pos-1.0), 2.8, 2.0,
#                              boxstyle="round,pad=0.15", facecolor='lightyellow',
#                              edgecolor='black', linewidth=3.5)
#         ax.add_patch(box5)
#         ax.text(x5, y_pos+0.4, 'Output', ha='center', va='center',
#                fontsize=13, fontweight='bold')
#         ax.text(x5, y_pos-0.4, '11 Classes', ha='center', va='center',
#                fontsize=13, style='italic')

#         # Info boxes at bottom - BIGGER and MORE SPACED
#         info_y = 4.0
#         info_boxes = [
#             ('47,865\nparams', 3.0, info_y, '#FFECB3'),
#             ('dim=64\nd_state=4', 10.0, info_y, '#E1BEE7'),
#             ('Bi-SSM\nScanning', 16.5, info_y, '#C5CAE9'),
#         ]

#         for text, x, y, color in info_boxes:
#             ax.text(x, y, text, ha='center', va='center',
#                    fontsize=13, bbox=dict(boxstyle='round,pad=0.6',
#                    facecolor=color, edgecolor='gray', linewidth=2.5))

#         # Key innovation - BIGGER with MORE SPACE from diagram
#         ax.text(10.0, 1.2, '⚡ Key: Bidirectional State Space Model + Channel Confusion Layer',
#                ha='center', fontsize=14, style='italic', fontweight='bold',
#                bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat',
#                         alpha=0.9, linewidth=3, edgecolor='orange'))

#     def setup_figure(self):
#         """Setup the main figure with all panels"""
#         fig = plt.figure(figsize=(22, 13))  # Extra large figure for bigger boxes
#         gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

#         # Create subplots - give more space to architecture
#         ax_raw = fig.add_subplot(gs[0, 0])
#         ax_spec = fig.add_subplot(gs[0, 1])
#         ax_arch = fig.add_subplot(gs[0, 2:])  # Wider architecture panel
#         ax_prob = fig.add_subplot(gs[1, :3])
#         ax_result = fig.add_subplot(gs[1, 3])
#         ax_info = fig.add_subplot(gs[2, :])

#         return fig, (ax_raw, ax_spec, ax_arch, ax_prob, ax_result, ax_info)

#     def animate_frame(self, frame_num):
#         """Animate a single frame showing the complete pipeline"""
#         if frame_num >= len(self.mat_files):
#             return

#         # Load current sample (now a tuple of (mat_path, class_idx))
#         mat_path, class_idx = self.mat_files[frame_num]
#         activity_name = mat_path.parent.name

#         # Process pipeline
#         raw_signal = self.load_radar_signal(mat_path)
#         if raw_signal is None:
#             return

#         spectrogram = self.process_to_spectrogram(raw_signal)
#         pred_class, confidence, probabilities = self.predict(spectrogram)

#         # Update visualizations
#         self.update_visualizations(raw_signal, spectrogram,
#                                   pred_class, confidence, probabilities,
#                                   class_idx, activity_name, frame_num)

#     def update_visualizations(self, raw_signal, spectrogram,
#                              pred_class, confidence, probabilities,
#                              true_class, activity_name, frame_num):
#         """Update all visualization panels"""
#         ax_raw, ax_spec, ax_arch, ax_prob, ax_result, ax_info = self.axes

#         # 1. Raw radar signal (magnitude)
#         ax_raw.clear()
#         magnitude = np.abs(raw_signal)
#         im1 = ax_raw.imshow(magnitude, cmap='viridis', aspect='auto')
#         ax_raw.set_title('① Raw Radar Signal\n(Complex IQ Data)',
#                         fontsize=11, fontweight='bold')
#         ax_raw.set_xlabel('Time')
#         ax_raw.set_ylabel('Frequency')

#         # 2. Processed spectrogram
#         ax_spec.clear()
#         im2 = ax_spec.imshow(spectrogram, cmap='viridis', aspect='auto')
#         ax_spec.set_title('② Micro-Doppler Spectrogram\n(224×224 Normalized)',
#                          fontsize=11, fontweight='bold')
#         ax_spec.set_xlabel('Time')
#         ax_spec.set_ylabel('Frequency')

#         # 3. RadMamba architecture (static)
#         self.draw_radmamba_architecture(ax_arch)

#         # 4. Probability distribution
#         ax_prob.clear()
#         colors = ['green' if i == true_class else 'lightcoral' if i == pred_class else 'lightblue'
#                  for i in range(len(self.activity_names))]
#         bars = ax_prob.barh(self.activity_names, probabilities * 100, color=colors)
#         ax_prob.set_xlabel('Confidence (%)', fontsize=11, fontweight='bold')
#         ax_prob.set_title('③ RadMamba Classification Output',
#                          fontsize=12, fontweight='bold')
#         ax_prob.set_xlim(0, 100)
#         ax_prob.grid(axis='x', alpha=0.3)

#         # Add percentage labels
#         for i, (bar, prob) in enumerate(zip(bars, probabilities)):
#             if prob > 0.01:
#                 ax_prob.text(prob * 100 + 1, i, f'{prob*100:.1f}%',
#                            va='center', fontsize=9)

#         # 5. Classification result
#         ax_result.clear()
#         ax_result.axis('off')

#         # Show result
#         correct = pred_class == true_class
#         result_color = 'green' if correct else 'red'
#         result_text = '✓ CORRECT' if correct else '✗ INCORRECT'

#         ax_result.text(0.5, 0.7, '④ RESULT', ha='center', va='center',
#                       fontsize=14, fontweight='bold')
#         ax_result.text(0.5, 0.5, result_text, ha='center', va='center',
#                       fontsize=16, fontweight='bold', color=result_color,
#                       bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.3))
#         ax_result.text(0.5, 0.3, f'Confidence: {confidence*100:.1f}%',
#                       ha='center', va='center', fontsize=12)

#         # 6. Information panel
#         ax_info.clear()
#         ax_info.axis('off')

#         info_text = f"""
#         ═══════════════════════════════════════════════════════════════════════════════════

#         📊 Sample {frame_num + 1}/{len(self.mat_files)}  |  📁 File: {activity_name}

#         ✓ True Activity: {self.activity_names[true_class]}
#         {'✓' if correct else '✗'} Predicted: {self.activity_names[pred_class]} ({confidence*100:.1f}% confidence)

#         🔄 Pipeline: Radar Signal → STFT → Spectrogram → RadMamba → Classification

#         ═══════════════════════════════════════════════════════════════════════════════════
#         """

#         ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
#                     fontsize=10, family='monospace',
#                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

#         plt.suptitle('RadMamba Real-Time Radar Classification Pipeline',
#                     fontsize=16, fontweight='bold', y=0.98)

#     def create_animation(self, output_path='pipeline_animation.gif', fps=1):
#         """Create animated visualization"""
#         fig, self.axes = self.setup_figure()

#         anim = animation.FuncAnimation(
#             fig,
#             self.animate_frame,
#             frames=len(self.mat_files),
#             interval=1000/fps,  # milliseconds per frame
#             repeat=True
#         )

#         # Save animation
#         print(f"Creating animation with {len(self.mat_files)} frames...")
#         anim.save(output_path, writer='pillow', fps=fps)
#         print(f"✓ Animation saved to {output_path}")

#         return anim

#     def create_static_frames(self, output_dir='pipeline_frames'):
#         """Create individual frames as images"""
#         output_dir = Path(output_dir)
#         output_dir.mkdir(exist_ok=True)

#         fig, self.axes = self.setup_figure()

#         print(f"Creating {len(self.mat_files)} frames...")
#         for i in range(len(self.mat_files)):
#             self.animate_frame(i)
#             plt.savefig(output_dir / f'frame_{i:03d}.png',
#                        dpi=150, bbox_inches='tight')
#             print(f"  ✓ Frame {i+1}/{len(self.mat_files)}")

#         plt.close()
#         print(f"✓ All frames saved to {output_dir}/")

# if __name__ == "__main__":
#     # Configuration
#     model_path = "save/CI4R/classify/CL_S_0_M_RADMAMBA_B_16_LR_0.0050_H_64_P_47865_FL_224_ST_1.pt"
#     config_path = "datasets/CI4R/spec.json"
#     mat_files_dir = "/Users/parthbhalodiya/Downloads/C781/AIRHAR/datasets/CI4R/Cross-frequency/77ghz/activity_spect_matrices"

#     # Create animator
#     animator = RadarPipelineAnimator(model_path, config_path, mat_files_dir)

#     # Option 1: Create animated GIF
#     print("\n=== Creating Animated GIF ===")
#     animator.create_animation(output_path='visualization_results/pipeline_animation.gif', fps=0.5)

#     # Option 2: Create individual frames
#     print("\n=== Creating Individual Frames ===")
#     animator.create_static_frames(output_dir='visualization_results/pipeline_frames')

#     print("\n✓ Pipeline visualization complete!")
#     print("  - Animated GIF: visualization_results/pipeline_animation.gif")
#     print("  - Static frames: visualization_results/pipeline_frames/")




# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import matplotlib.image as mpimg
# from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
# from matplotlib.gridspec import GridSpec
# import scipy.io
# import cv2
# import json
# import sys
# from pathlib import Path

# sys.path.insert(0, '.')
# from models import CoreModel


# class RadarPipelineAnimator:
#     def __init__(self, model_path, config_path, mat_files_dir, arch_image_path="radmamba_architecture.png"):
#         """Initialize the pipeline animator"""
#         self.model, self.config = self.load_model(model_path, config_path)
#         self.mat_files_dir = Path(mat_files_dir)
#         self.mat_files = self.collect_mat_files()
#         self.current_idx = 0

#         # Load architecture PNG once
#         self.arch_image = mpimg.imread(arch_image_path)
#         print(f"✓ Loaded architecture image: {arch_image_path}")

#         # Activity names
#         self.activity_names = [
#             'Walking Towards',
#             'Walking Away',
#             'Picking Object',
#             'Bending',
#             'Sitting',
#             'Kneeling',
#             'Crawling',
#             'Walking on Toes',
#             'Limping',
#             'Short Steps',
#             'Scissors Gait'
#         ]

#     def load_model(self, model_path, config_path):
#         """Load trained RadMamba model"""
#         with open(config_path, 'r') as f:
#             config = json.load(f)

#         model = CoreModel(
#             hidden_size=config['Classification_hidden_size'],
#             num_layers=config['Classification_num_layers'],
#             backbone_type=config['Classification_backbone'],
#             dim=64,
#             dt_rank=config.get('dt_rank', 0),
#             d_state=config.get('d_state', 4),
#             image_height=config['image_height'],
#             image_width=config['frame_length'],
#             num_classes=config['num_classes'],
#             channels=config['channels'],
#             dropout=config.get('dropout', 0),
#             optional_avg_pool=True,
#             channel_confusion_layer=config.get('channel_confusion_layer', 1),
#             channel_confusion_out_channels=config.get('channel_confusion_out_channels', 1),
#             time_downsample_factor=config.get('time_downsample_factor', 4)
#         )

#         state_dict = torch.load(model_path, map_location='cpu')
#         model.load_state_dict(state_dict)
#         model.eval()
#         print("✓ Model loaded successfully")

#         return model, config

#     def collect_mat_files(self):
#         """Collect 5 samples from each of the 11 classes"""
#         print("Collecting 5 samples from each class...")

#         selected_samples = []

#         actual_class_idx = 0
#         for activity_dir in sorted(self.mat_files_dir.iterdir()):
#             if activity_dir.is_dir():
#                 print(f"\n  Analyzing Class {actual_class_idx}: {activity_dir.name}...")
#                 files = list(activity_dir.glob('*.mat'))

#                 class_samples = []

#                 for i, mat_file in enumerate(files):
#                     try:
#                         raw_signal = self.load_radar_signal(mat_file)
#                         if raw_signal is None:
#                             continue

#                         spectrogram = self.process_to_spectrogram(raw_signal)
#                         pred_class, confidence, _ = self.predict(spectrogram)

#                         is_correct = (pred_class == actual_class_idx)
#                         class_samples.append((mat_file, actual_class_idx, confidence, is_correct))

#                     except Exception as e:
#                         continue

#                 if len(class_samples) > 0:
#                     class_samples.sort(key=lambda x: (not x[3], -x[2]))
#                     top_5 = class_samples[:5]

#                     correct_count = sum(1 for _, _, _, is_correct in top_5 if is_correct)
#                     print(f"    Selected {len(top_5)} samples ({correct_count} correct)")

#                     for mat, cls, conf, is_correct in top_5:
#                         selected_samples.append((mat, cls))

#                 actual_class_idx += 1

#         total_correct = sum(1 for mat, cls in selected_samples
#                           if self.is_prediction_correct(mat, cls))
#         print(f"\n✓ Selected {len(selected_samples)} total samples")
#         print(f"  {total_correct}/{len(selected_samples)} predictions are correct\n")

#         return selected_samples

#     def is_prediction_correct(self, mat_file, true_class):
#         """Helper to check if a prediction is correct"""
#         try:
#             raw_signal = self.load_radar_signal(mat_file)
#             if raw_signal is None:
#                 return False
#             spectrogram = self.process_to_spectrogram(raw_signal)
#             pred_class, _, _ = self.predict(spectrogram)
#             return pred_class == true_class
#         except:
#             return False

#     def load_radar_signal(self, mat_path):
#         """Load raw complex radar signal from .mat file"""
#         mat_data = scipy.io.loadmat(str(mat_path))
#         if 'sx1' in mat_data:
#             return mat_data['sx1']
#         return None

#     def process_to_spectrogram(self, complex_signal):
#         """Convert complex signal to spectrogram"""
#         magnitude = np.abs(complex_signal)
#         spectrogram_db = 20 * np.log10(magnitude + 1e-10)
#         spectrogram_db = np.clip(spectrogram_db, -45, spectrogram_db.max())

#         spectrogram = cv2.resize(spectrogram_db.astype(np.float32),
#                                 (224, 224),
#                                 interpolation=cv2.INTER_CUBIC)

#         spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)

#         return spectrogram

#     def predict(self, spectrogram):
#         """Run inference through RadMamba"""
#         x = torch.FloatTensor(spectrogram).unsqueeze(0)

#         with torch.no_grad():
#             logits = self.model(x)
#             probabilities = torch.softmax(logits, dim=1)
#             predicted_class = torch.argmax(probabilities, dim=1).item()
#             confidence = probabilities[0, predicted_class].item()

#         return predicted_class, confidence, probabilities[0].cpu().numpy()

#     def draw_radmamba_architecture(self, ax):
#         """Display the pre-rendered architecture PNG (NO TITLE - already in PNG)"""
#         ax.clear()
#         ax.axis('off')
#         ax.imshow(self.arch_image)
#         # ✅ REMOVED TITLE - it's already in the PNG

#     def setup_figure(self):
#         """Setup the main figure - IMPROVED LAYOUT"""
#         # ✅ BIGGER figure, tighter margins
#         fig = plt.figure(figsize=(24, 14))
        
#         # ✅ TIGHTER margins to remove white space
#         fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, hspace=0.3, wspace=0.15)

#         # ✅ IMPROVED GridSpec - bigger images, balanced layout
#         gs = GridSpec(
#             3, 3,
#             figure=fig,
#             height_ratios=[1.3, 1.0, 0.5],
#             width_ratios=[1.0, 1.0, 1.8],
#             hspace=0.25,
#             wspace=0.12
#         )

#         # Row 1: Raw Signal | Spectrogram | Architecture (spans more)
#         ax_raw = fig.add_subplot(gs[0, 0])
#         ax_spec = fig.add_subplot(gs[0, 1])
#         ax_arch = fig.add_subplot(gs[0, 2])

#         # Row 2: Probability Chart (wide) | Result
#         ax_prob = fig.add_subplot(gs[1, 0:2])
#         ax_result = fig.add_subplot(gs[1, 2])

#         # Row 3: Info panel (full width)
#         ax_info = fig.add_subplot(gs[2, :])

#         return fig, (ax_raw, ax_spec, ax_arch, ax_prob, ax_result, ax_info)

#     def animate_frame(self, frame_num):
#         """Animate a single frame showing the complete pipeline"""
#         if frame_num >= len(self.mat_files):
#             return

#         mat_path, class_idx = self.mat_files[frame_num]
#         activity_name = mat_path.parent.name

#         raw_signal = self.load_radar_signal(mat_path)
#         if raw_signal is None:
#             return

#         spectrogram = self.process_to_spectrogram(raw_signal)
#         pred_class, confidence, probabilities = self.predict(spectrogram)

#         self.update_visualizations(raw_signal, spectrogram,
#                                   pred_class, confidence, probabilities,
#                                   class_idx, activity_name, frame_num)

#     def update_visualizations(self, raw_signal, spectrogram,
#                              pred_class, confidence, probabilities,
#                              true_class, activity_name, frame_num):
#         """Update all visualization panels - IMPROVED"""
#         ax_raw, ax_spec, ax_arch, ax_prob, ax_result, ax_info = self.axes

#         # 1. Raw radar signal (magnitude) - BIGGER TITLE
#         ax_raw.clear()
#         magnitude = np.abs(raw_signal)
#         ax_raw.imshow(magnitude, cmap='viridis', aspect='auto')
#         ax_raw.set_title('① Raw Radar Signal\n(Complex IQ Data)',
#                         fontsize=13, fontweight='bold', pad=8)
#         ax_raw.set_xlabel('Time', fontsize=11)
#         ax_raw.set_ylabel('Frequency', fontsize=11)

#         # 2. Processed spectrogram - BIGGER TITLE
#         ax_spec.clear()
#         ax_spec.imshow(spectrogram, cmap='viridis', aspect='auto')
#         ax_spec.set_title('② Micro-Doppler Spectrogram\n(224×224 Normalized)',
#                          fontsize=13, fontweight='bold', pad=8)
#         ax_spec.set_xlabel('Time', fontsize=11)
#         ax_spec.set_ylabel('Frequency', fontsize=11)

#         # 3. RadMamba architecture (PNG - no extra title)
#         self.draw_radmamba_architecture(ax_arch)

#         # 4. Probability distribution - IMPROVED
#         ax_prob.clear()
#         colors = ['#2ecc71' if i == true_class else '#e74c3c' if i == pred_class else '#3498db'
#                  for i in range(len(self.activity_names))]
#         bars = ax_prob.barh(self.activity_names, probabilities * 100, color=colors, height=0.7)
#         ax_prob.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
#         ax_prob.set_title('③ RadMamba Classification Output',
#                          fontsize=14, fontweight='bold', pad=10)
#         ax_prob.set_xlim(0, 100)
#         ax_prob.grid(axis='x', alpha=0.3, linestyle='--')
#         ax_prob.tick_params(axis='y', labelsize=10)

#         # Add percentage labels
#         for i, (bar, prob) in enumerate(zip(bars, probabilities)):
#             if prob > 0.01:
#                 ax_prob.text(prob * 100 + 1.5, i, f'{prob*100:.1f}%',
#                            va='center', fontsize=10, fontweight='bold')

#         # 5. Classification result - IMPROVED STYLING
#         ax_result.clear()
#         ax_result.axis('off')

#         correct = pred_class == true_class
#         result_color = '#2ecc71' if correct else '#e74c3c'
#         result_text = '✓ CORRECT' if correct else '✗ INCORRECT'
#         bg_color = '#d5f5e3' if correct else '#fadbd8'

#         # Background
#         ax_result.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
#                                           boxstyle="round,pad=0.02",
#                                           facecolor=bg_color,
#                                           edgecolor=result_color,
#                                           linewidth=3,
#                                           transform=ax_result.transAxes))

#         ax_result.text(0.5, 0.75, '④ RESULT', ha='center', va='center',
#                       fontsize=16, fontweight='bold', transform=ax_result.transAxes)
#         ax_result.text(0.5, 0.50, result_text, ha='center', va='center',
#                       fontsize=20, fontweight='bold', color=result_color,
#                       transform=ax_result.transAxes)
#         ax_result.text(0.5, 0.28, f'Confidence: {confidence*100:.1f}%',
#                       ha='center', va='center', fontsize=14,
#                       transform=ax_result.transAxes)

#         # 6. Information panel - CLEANER
#         ax_info.clear()
#         ax_info.axis('off')

#         # Cleaner info box
#         info_text = (
#             f"📊 Sample {frame_num + 1}/{len(self.mat_files)}  │  "
#             f"📁 {activity_name}  │  "
#             f"✓ True: {self.activity_names[true_class]}  │  "
#             f"{'✓' if correct else '✗'} Predicted: {self.activity_names[pred_class]} ({confidence*100:.1f}%)"
#         )

#         ax_info.text(0.5, 0.6, info_text, ha='center', va='center',
#                     fontsize=12, fontweight='bold',
#                     transform=ax_info.transAxes,
#                     bbox=dict(boxstyle='round,pad=0.5',
#                              facecolor='#fef9e7',
#                              edgecolor='#f39c12',
#                              linewidth=2))

#         ax_info.text(0.5, 0.2,
#                     "🔄 Pipeline: Radar Signal → STFT → Spectrogram → RadMamba → Classification",
#                     ha='center', va='center', fontsize=11,
#                     transform=ax_info.transAxes,
#                     style='italic', color='#555')

#         # Main title
#         plt.suptitle('RadMamba Real-Time Radar Classification Pipeline',
#                     fontsize=18, fontweight='bold', y=0.97)

#     def create_animation(self, output_path='pipeline_animation.gif', fps=1):
#         """Create animated visualization"""
#         fig, self.axes = self.setup_figure()

#         anim = animation.FuncAnimation(
#             fig,
#             self.animate_frame,
#             frames=len(self.mat_files),
#             interval=1000/fps,
#             repeat=True
#         )

#         print(f"Creating animation with {len(self.mat_files)} frames...")
#         anim.save(output_path, writer='pillow', fps=fps)
#         print(f"✓ Animation saved to {output_path}")

#         return anim

#     def create_static_frames(self, output_dir='pipeline_frames'):
#         """Create individual frames as images"""
#         output_dir = Path(output_dir)
#         output_dir.mkdir(exist_ok=True)

#         fig, self.axes = self.setup_figure()

#         print(f"Creating {len(self.mat_files)} frames...")
#         for i in range(len(self.mat_files)):
#             self.animate_frame(i)
#             plt.savefig(output_dir / f'frame_{i:03d}.png',
#                        dpi=150, bbox_inches='tight',
#                        facecolor='white', edgecolor='none')
#             print(f"  ✓ Frame {i+1}/{len(self.mat_files)}")

#         plt.close()
#         print(f"✓ All frames saved to {output_dir}/")


# if __name__ == "__main__":
#     # Configuration
#     model_path = "save/CI4R/classify/CL_S_0_M_RADMAMBA_B_16_LR_0.0050_H_64_P_47865_FL_224_ST_1.pt"
#     config_path = "datasets/CI4R/spec.json"
#     mat_files_dir = "/Users/parthbhalodiya/Downloads/C781/AIRHAR/datasets/CI4R/Cross-frequency/77ghz/activity_spect_matrices"
#     arch_image_path = "radmamba_architecture.png"

#     # Create animator
#     animator = RadarPipelineAnimator(model_path, config_path, mat_files_dir, arch_image_path)

#     # Option 1: Create animated GIF
#     print("\n=== Creating Animated GIF ===")
#     animator.create_animation(output_path='visualization_results/pipeline_animation.gif', fps=0.5)

#     # Option 2: Create individual frames
#     print("\n=== Creating Individual Frames ===")
#     animator.create_static_frames(output_dir='visualization_results/pipeline_frames')

#     print("\n✓ Pipeline visualization complete!")




import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import scipy.io
import cv2
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from models import CoreModel


class RadarPipelineAnimator:
    def __init__(self, model_path, config_path, mat_files_dir, arch_image_path="radmamba_architecture.png"):
        """Initialize the pipeline animator"""
        self.model, self.config = self.load_model(model_path, config_path)
        self.mat_files_dir = Path(mat_files_dir)
        self.mat_files = self.collect_mat_files()
        self.current_idx = 0

        # Load architecture PNG once
        self.arch_image = mpimg.imread(arch_image_path)
        print(f"✓ Loaded architecture image: {arch_image_path}")

        # Activity names (full - for info panel)
        self.activity_names = [
            'Walking Towards',
            'Walking Away',
            'Picking Object',
            'Bending',
            'Sitting',
            'Kneeling',
            'Crawling',
            'Walking on Toes',
            'Limping',
            'Short Steps',
            'Scissors Gait'
        ]

        # Short names (for chart display - prevents cutting)
        self.short_names = [
            'Walk Towards',
            'Walk Away',
            'Pick Object',
            'Bending',
            'Sitting',
            'Kneeling',
            'Crawling',
            'Walk on Toes',
            'Limping',
            'Short Steps',
            'Scissors Gait'
        ]

    def load_model(self, model_path, config_path):
        """Load trained RadMamba model"""
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = CoreModel(
            hidden_size=config['Classification_hidden_size'],
            num_layers=config['Classification_num_layers'],
            backbone_type=config['Classification_backbone'],
            dim=64,
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

        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("✓ Model loaded successfully")

        return model, config

    def collect_mat_files(self):
        """Collect 5 samples from each of the 11 classes"""
        print("Collecting 5 samples from each class...")

        selected_samples = []

        actual_class_idx = 0
        for activity_dir in sorted(self.mat_files_dir.iterdir()):
            if activity_dir.is_dir():
                print(f"\n  Analyzing Class {actual_class_idx}: {activity_dir.name}...")
                files = list(activity_dir.glob('*.mat'))

                class_samples = []

                for i, mat_file in enumerate(files):
                    try:
                        raw_signal = self.load_radar_signal(mat_file)
                        if raw_signal is None:
                            continue

                        spectrogram = self.process_to_spectrogram(raw_signal)
                        pred_class, confidence, _ = self.predict(spectrogram)

                        is_correct = (pred_class == actual_class_idx)
                        class_samples.append((mat_file, actual_class_idx, confidence, is_correct))

                    except Exception as e:
                        continue

                if len(class_samples) > 0:
                    class_samples.sort(key=lambda x: (not x[3], -x[2]))
                    top_5 = class_samples[:5]

                    correct_count = sum(1 for _, _, _, is_correct in top_5 if is_correct)
                    print(f"    Selected {len(top_5)} samples ({correct_count} correct)")

                    for mat, cls, conf, is_correct in top_5:
                        selected_samples.append((mat, cls))

                actual_class_idx += 1

        total_correct = sum(1 for mat, cls in selected_samples
                          if self.is_prediction_correct(mat, cls))
        print(f"\n✓ Selected {len(selected_samples)} total samples")
        print(f"  {total_correct}/{len(selected_samples)} predictions are correct\n")

        return selected_samples

    def is_prediction_correct(self, mat_file, true_class):
        """Helper to check if a prediction is correct"""
        try:
            raw_signal = self.load_radar_signal(mat_file)
            if raw_signal is None:
                return False
            spectrogram = self.process_to_spectrogram(raw_signal)
            pred_class, _, _ = self.predict(spectrogram)
            return pred_class == true_class
        except:
            return False

    def load_radar_signal(self, mat_path):
        """Load raw complex radar signal from .mat file"""
        mat_data = scipy.io.loadmat(str(mat_path))
        if 'sx1' in mat_data:
            return mat_data['sx1']
        return None

    def process_to_spectrogram(self, complex_signal):
        """Convert complex signal to spectrogram"""
        magnitude = np.abs(complex_signal)
        spectrogram_db = 20 * np.log10(magnitude + 1e-10)
        spectrogram_db = np.clip(spectrogram_db, -45, spectrogram_db.max())

        spectrogram = cv2.resize(spectrogram_db.astype(np.float32),
                                (224, 224),
                                interpolation=cv2.INTER_CUBIC)

        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)

        return spectrogram

    def predict(self, spectrogram):
        """Run inference through RadMamba"""
        x = torch.FloatTensor(spectrogram).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        return predicted_class, confidence, probabilities[0].cpu().numpy()

    def draw_radmamba_architecture(self, ax):
        """Display the pre-rendered architecture PNG (NO extra title)"""
        ax.clear()
        ax.axis('off')
        ax.imshow(self.arch_image)

    def setup_figure(self):
        """Setup the main figure - IMPROVED LAYOUT with proper margins"""
        fig = plt.figure(figsize=(26, 14))
        
        # More left margin for class names
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05, hspace=0.35, wspace=0.20)

        gs = GridSpec(
            3, 3,
            figure=fig,
            height_ratios=[1.2, 1.1, 0.4],
            width_ratios=[1.0, 1.0, 1.5],
            hspace=0.30,
            wspace=0.18
        )

        # Row 1: Raw Signal | Spectrogram | Architecture
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_spec = fig.add_subplot(gs[0, 1])
        ax_arch = fig.add_subplot(gs[0, 2])

        # Row 2: Probability Chart (wider) | Result
        ax_prob = fig.add_subplot(gs[1, 0:2])
        ax_result = fig.add_subplot(gs[1, 2])

        # Row 3: Info panel (full width)
        ax_info = fig.add_subplot(gs[2, :])

        return fig, (ax_raw, ax_spec, ax_arch, ax_prob, ax_result, ax_info)

    def animate_frame(self, frame_num):
        """Animate a single frame showing the complete pipeline"""
        if frame_num >= len(self.mat_files):
            return

        mat_path, class_idx = self.mat_files[frame_num]
        activity_name = mat_path.parent.name

        raw_signal = self.load_radar_signal(mat_path)
        if raw_signal is None:
            return

        spectrogram = self.process_to_spectrogram(raw_signal)
        pred_class, confidence, probabilities = self.predict(spectrogram)

        self.update_visualizations(raw_signal, spectrogram,
                                  pred_class, confidence, probabilities,
                                  class_idx, activity_name, frame_num)

    def update_visualizations(self, raw_signal, spectrogram,
                             pred_class, confidence, probabilities,
                             true_class, activity_name, frame_num):
        """Update all visualization panels - FIXED & IMPROVED"""
        ax_raw, ax_spec, ax_arch, ax_prob, ax_result, ax_info = self.axes

        # 1. Raw radar signal (magnitude)
        ax_raw.clear()
        magnitude = np.abs(raw_signal)
        ax_raw.imshow(magnitude, cmap='viridis', aspect='auto')
        ax_raw.set_title('① Raw Radar Signal\n(Complex IQ Data)',
                        fontsize=13, fontweight='bold', pad=8)
        ax_raw.set_xlabel('Time', fontsize=11)
        ax_raw.set_ylabel('Frequency', fontsize=11)

        # 2. Processed spectrogram
        ax_spec.clear()
        ax_spec.imshow(spectrogram, cmap='viridis', aspect='auto')
        ax_spec.set_title('② Micro-Doppler Spectrogram\n(224×224 Normalized)',
                         fontsize=13, fontweight='bold', pad=8)
        ax_spec.set_xlabel('Time', fontsize=11)
        ax_spec.set_ylabel('Frequency', fontsize=11)

        # 3. RadMamba architecture (PNG - no extra title)
        self.draw_radmamba_architecture(ax_arch)

        # 4. Probability distribution - FIXED (no cutting)
        ax_prob.clear()
        
        colors = ['#2ecc71' if i == true_class else '#e74c3c' if i == pred_class else '#3498db'
                 for i in range(len(self.activity_names))]
        
        bars = ax_prob.barh(self.activity_names, probabilities * 100, color=colors, height=0.6)
        
        ax_prob.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
        ax_prob.set_title('③ RadMamba Classification Output',
                         fontsize=14, fontweight='bold', pad=10)
        ax_prob.set_xlim(0, 105)
        ax_prob.grid(axis='x', alpha=0.3, linestyle='--')
        ax_prob.tick_params(axis='y', labelsize=10)
        ax_prob.margins(y=0.01)

        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > 0.01:
                ax_prob.text(prob * 100 + 1.5, i, f'{prob*100:.1f}%',
                           va='center', fontsize=9, fontweight='bold')

        # 5. Classification result - STYLED
        ax_result.clear()
        ax_result.axis('off')

        correct = pred_class == true_class
        result_color = '#2ecc71' if correct else '#e74c3c'
        result_text = '✓ CORRECT' if correct else '✗ INCORRECT'
        bg_color = '#d5f5e3' if correct else '#fadbd8'

        # Background box
        ax_result.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                          boxstyle="round,pad=0.02",
                                          facecolor=bg_color,
                                          edgecolor=result_color,
                                          linewidth=3,
                                          transform=ax_result.transAxes))

        ax_result.text(0.5, 0.75, '④ RESULT', ha='center', va='center',
                      fontsize=16, fontweight='bold', transform=ax_result.transAxes)
        ax_result.text(0.5, 0.50, result_text, ha='center', va='center',
                      fontsize=20, fontweight='bold', color=result_color,
                      transform=ax_result.transAxes)
        ax_result.text(0.5, 0.28, f'Confidence: {confidence*100:.1f}%',
                      ha='center', va='center', fontsize=14,
                      transform=ax_result.transAxes)

        # 6. Information panel - CLEAN
        ax_info.clear()
        ax_info.axis('off')

        info_text = (
            f"📊 Sample {frame_num + 1}/{len(self.mat_files)}  │  "
            f"📁 {activity_name}  │  "
            f"✓ True: {self.activity_names[true_class]}  │  "
            f"{'✓' if correct else '✗'} Pred: {self.activity_names[pred_class]} ({confidence*100:.1f}%)"
        )

        ax_info.text(0.5, 0.6, info_text, ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='#fef9e7',
                             edgecolor='#f39c12',
                             linewidth=2))

        ax_info.text(0.5, 0.2,
                    "🔄 Pipeline: Radar Signal → STFT → Spectrogram → RadMamba → Classification",
                    ha='center', va='center', fontsize=11,
                    transform=ax_info.transAxes,
                    style='italic', color='#555')

        # Main title
        plt.suptitle('RadMamba Real-Time Radar Classification Pipeline',
                    fontsize=18, fontweight='bold', y=0.97)

    def create_animation(self, output_path='pipeline_animation.gif', fps=1):
        """Create animated visualization"""
        fig, self.axes = self.setup_figure()

        anim = animation.FuncAnimation(
            fig,
            self.animate_frame,
            frames=len(self.mat_files),
            interval=1000/fps,
            repeat=True
        )

        print(f"Creating animation with {len(self.mat_files)} frames...")
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"✓ Animation saved to {output_path}")

        return anim

    def create_static_frames(self, output_dir='pipeline_frames'):
        """Create individual frames as images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        fig, self.axes = self.setup_figure()

        print(f"Creating {len(self.mat_files)} frames...")
        for i in range(len(self.mat_files)):
            self.animate_frame(i)
            plt.savefig(output_dir / f'frame_{i:03d}.png',
                       dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"  ✓ Frame {i+1}/{len(self.mat_files)}")

        plt.close()
        print(f"✓ All frames saved to {output_dir}/")


if __name__ == "__main__":
    # Configuration
    model_path = "save/CI4R/classify/CL_S_0_M_RADMAMBA_B_16_LR_0.0050_H_64_P_47865_FL_224_ST_1.pt"
    config_path = "datasets/CI4R/spec.json"
    mat_files_dir = "/Users/parthbhalodiya/Downloads/C781/AIRHAR/datasets/CI4R/Cross-frequency/77ghz/activity_spect_matrices"
    arch_image_path = "radmamba_architecture.png"

    # Create output directory
    Path("visualization_results").mkdir(exist_ok=True)

    # Create animator
    animator = RadarPipelineAnimator(model_path, config_path, mat_files_dir, arch_image_path)

    # Option 1: Create animated GIF
    print("\n=== Creating Animated GIF ===")
    animator.create_animation(output_path='visualization_results/pipeline_animation.gif', fps=0.5)

    # Option 2: Create individual frames
    print("\n=== Creating Individual Frames ===")
    animator.create_static_frames(output_dir='visualization_results/pipeline_frames')

    print("\n✓ Pipeline visualization complete!")
    print("  - Animated GIF: visualization_results/pipeline_animation.gif")
    print("  - Static frames: visualization_results/pipeline_frames/")