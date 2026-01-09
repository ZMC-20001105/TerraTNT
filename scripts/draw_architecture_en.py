#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Draw System Architecture Diagram (English version to avoid font issues)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.font_manager as fm

# Create figure
fig, ax = plt.subplots(figsize=(16, 11), facecolor='white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# Title
ax.text(8, 10.3, 'TerraTNT System Architecture', 
        fontsize=28, fontweight='bold', ha='center', va='center', family='sans-serif')

# Layer 1: Presentation Layer (UI)
ui_box = FancyBboxPatch((0.5, 8.2), 15, 1.6, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='#1976d2', facecolor='#e3f2fd', linewidth=2.5)
ax.add_patch(ui_box)
ax.text(1, 9.5, 'Presentation Layer (PyQt5 UI)', fontsize=16, fontweight='bold', color='#0d47a1')

# UI modules
ui_modules = [
    'Main Window', 
    'Satellite Config', 
    'Data Loader', 
    'Trajectory Viewer',
    'Training Console'
]
for i, module in enumerate(ui_modules):
    x = 1.2 + i * 2.8
    rect = FancyBboxPatch((x, 8.4), 2.5, 0.7, 
                          boxstyle="round,pad=0.05",
                          edgecolor='#1976d2', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.25, 8.75, module, fontsize=11, ha='center', va='center', fontweight='600')

# Layer 2: Application Layer (Business Logic)
app_box = FancyBboxPatch((0.5, 5.8), 15, 2, 
                        boxstyle="round,pad=0.1",
                        edgecolor='#388e3c', facecolor='#e8f5e9', linewidth=2.5)
ax.add_patch(app_box)
ax.text(1, 7.5, 'Application Layer (Business Logic)', fontsize=16, fontweight='bold', color='#1b5e20')

# Application modules
app_modules = [
    ('Data Synthesis\nEngine', 'Hierarchical A*\nParallel Workers'),
    ('Model\nOrchestrator', 'FAS Phase 1/2/3\nEarly Stopping'),
    ('Evaluation\nFramework', 'ADE / FDE / MR\nBenchmarking')
]
for i, (name, desc) in enumerate(app_modules):
    x = 1.2 + i * 4.7
    rect = FancyBboxPatch((x, 6), 4.2, 1.4,
                          boxstyle="round,pad=0.08",
                          edgecolor='#388e3c', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 2.1, 6.9, name, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(x + 2.1, 6.3, desc, fontsize=9, ha='center', va='center', color='#555', style='italic')

# Layer 3: Model Layer & Data Layer
# Model Layer
model_box = FancyBboxPatch((0.5, 3.5), 7, 2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#f57c00', facecolor='#fff3e0', linewidth=2.5)
ax.add_patch(model_box)
ax.text(1, 5.2, 'Model Layer (Deep Learning)', fontsize=16, fontweight='bold', color='#e65100')

# Model modules
models = [
    ('TerraTNT\nCore', 'CNN+LSTM\nGoal-Driven'),
    ('Baseline\nModels', 'YNet, PECNet\nTrajectron++'),
    ('Speed\nPredictor', 'XGBoost\nOORD-trained')
]
for i, (name, desc) in enumerate(models):
    x = 1.2 + i * 2
    rect = FancyBboxPatch((x, 3.7), 1.8, 1.3,
                          boxstyle="round,pad=0.05",
                          edgecolor='#f57c00', facecolor='#ffe0b2', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.9, 4.7, name, fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(x + 0.9, 4.1, desc, fontsize=8, ha='center', va='center', color='#555')

# Data Layer
data_box = FancyBboxPatch((8.5, 3.5), 7, 2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#c2185b', facecolor='#fce4ec', linewidth=2.5)
ax.add_patch(data_box)
ax.text(9, 5.2, 'Data Persistence Layer', fontsize=16, fontweight='bold', color='#880e4f')

# Data modules
data_modules = [
    ('GIS Assets', 'DEM, LULC\nOSM Roads\nUTM 32630'),
    ('Trajectory\nStore', '100k+ PKLs\nFAS Splits\nCheckpoints')
]
for i, (name, desc) in enumerate(data_modules):
    x = 9.2 + i * 3
    rect = FancyBboxPatch((x, 3.7), 2.8, 1.3,
                          boxstyle="round,pad=0.05",
                          edgecolor='#c2185b', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.4, 4.7, name, fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(x + 1.4, 4.1, desc, fontsize=8, ha='center', va='center', color='#555')

# Layer 4: Infrastructure
infra_box = FancyBboxPatch((0.5, 1.8), 15, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#616161', facecolor='#f5f5f5', linewidth=2.5)
ax.add_patch(infra_box)
ax.text(8, 2.7, 'Infrastructure Layer', fontsize=14, fontweight='bold', ha='center', color='#424242')
ax.text(8, 2.2, 'Ubuntu 24.04 LTS  |  NVIDIA CUDA Core  |  PyTorch 2.0+  |  GDAL & Proj  |  Anaconda Environment', 
        fontsize=11, ha='center', color='#616161')

# Add arrows
arrow_props = dict(arrowstyle='->', lw=2.5, color='#546e7a')

# UI -> Application
arrow1 = FancyArrowPatch((8, 8.2), (8, 7.8), **arrow_props)
ax.add_patch(arrow1)

# Application -> Model
arrow2 = FancyArrowPatch((4, 5.8), (4, 5.5), **arrow_props)
ax.add_patch(arrow2)

# Application -> Data
arrow3 = FancyArrowPatch((12, 5.8), (12, 5.5), **arrow_props)
ax.add_patch(arrow3)

# Model <-> Data (bidirectional)
arrow4 = FancyArrowPatch((7.5, 4.5), (8.5, 4.5), 
                        arrowstyle='<->', lw=2, color='#546e7a')
ax.add_patch(arrow4)

# Add legend
legend_y = 1.2
ax.text(1.5, legend_y, '● Presentation: User Interface & Visualization', 
        fontsize=10, color='#1976d2', fontweight='600')
ax.text(5.5, legend_y, '● Application: Business Logic & Orchestration', 
        fontsize=10, color='#388e3c', fontweight='600')
ax.text(10, legend_y, '● Model: AI/ML Components', 
        fontsize=10, color='#f57c00', fontweight='600')
ax.text(13.5, legend_y, '● Data: Storage', 
        fontsize=10, color='#c2185b', fontweight='600')

# Data flow description
ax.text(8, 0.5, 'Data Flow: User → UI → Application → Model/Data → Results', 
        fontsize=11, style='italic', ha='center', color='#546e7a', fontweight='500')

# Save figure
plt.tight_layout()
plt.savefig('/home/zmc/文档/programwork/docs/architecture_diagram.png', 
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Architecture diagram saved to: /home/zmc/文档/programwork/docs/architecture_diagram.png")
plt.close()
