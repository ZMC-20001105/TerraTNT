"""
System Architecture Diagram - English Version
Clean and professional for academic papers
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Use default fonts (no Chinese needed)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'data': '#E3F2FD',
        'process': '#FFF3E0',
        'model': '#F3E5F5',
        'app': '#E8F5E9',
        'border_data': '#1976D2',
        'border_process': '#F57C00',
        'border_model': '#7B1FA2',
        'border_app': '#388E3C'
    }
    
    # Title
    ax.text(9, 12.3, 'TerraTNT: Environment-Constrained Ground Target Trajectory Prediction System', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # ============ Layer 1: Data Layer ============
    y_data = 10.2
    
    ax.text(0.8, y_data + 0.7, 'Data Layer', 
            fontsize=15, fontweight='bold', color=colors['border_data'])
    
    data_boxes = [
        {'x': 0.8, 'y': y_data, 'w': 2.8, 'h': 0.7, 
         'text': 'DEM Data\nSRTM 30m'},
        {'x': 4.0, 'y': y_data, 'w': 2.8, 'h': 0.7, 
         'text': 'LULC Data\nESA WorldCover'},
        {'x': 7.2, 'y': y_data, 'w': 2.8, 'h': 0.7, 
         'text': 'OSM Roads\n6 Countries'},
        {'x': 10.4, 'y': y_data, 'w': 2.8, 'h': 0.7, 
         'text': 'OORD Tracks\nReal Data'},
        {'x': 13.6, 'y': y_data, 'w': 2.8, 'h': 0.7, 
         'text': 'Synthetic\n14,400 tracks'},
    ]
    
    for box in data_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['data'],
                              edgecolor=colors['border_data'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ============ Layer 2: Processing Layer ============
    y_process = 8.2
    
    ax.text(0.8, y_process + 0.7, 'Processing Layer', 
            fontsize=15, fontweight='bold', color=colors['border_process'])
    
    process_boxes = [
        {'x': 0.8, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': 'Environment Preprocessing\nProjection | Terrain Features'},
        {'x': 5.0, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': 'Cost Map Generation\nPassable Analysis | Multi-intent'},
        {'x': 9.2, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': 'Trajectory Generation\nHierarchical A* | XGBoost Speed'},
        {'x': 13.4, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': 'Data Augmentation\n18-channel Maps | Train/Val/Test'},
    ]
    
    for box in process_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['process'],
                              edgecolor=colors['border_process'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=9.5)
    
    # ============ Layer 3: Model Layer ============
    y_model = 5.8
    
    ax.text(0.8, y_model + 0.7, 'Model Layer (TerraTNT)', 
            fontsize=15, fontweight='bold', color=colors['border_model'])
    
    model_boxes = [
        {'x': 1.5, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': 'CNN Environment\nEncoder (ResNet-18)'},
        {'x': 5.3, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': 'LSTM History\nEncoder (2 Layers)'},
        {'x': 9.1, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': 'Goal Classifier\n(Candidate Scoring)'},
        {'x': 12.9, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': 'LSTM Decoder\n(Hierarchical)'},
    ]
    
    for box in model_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['model'],
                              edgecolor=colors['border_model'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Training framework box
    train_box = FancyBboxPatch((1.5, y_model - 1.3), 14.7, 0.9,
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['model'],
                               edgecolor=colors['border_model'], linewidth=2, linestyle='--')
    ax.add_patch(train_box)
    ax.text(8.85, y_model - 0.85, 
            'Training Framework: PyTorch | Adam Optimizer | Loss: NLL + ADE + FDE | Early Stopping | TensorBoard', 
            ha='center', va='center', fontsize=9.5, style='italic')
    
    # ============ Layer 4: Application Layer ============
    y_app = 2.8
    
    ax.text(0.8, y_app + 0.7, 'Application Layer', 
            fontsize=15, fontweight='bold', color=colors['border_app'])
    
    app_boxes = [
        {'x': 2.2, 'y': y_app, 'w': 3.8, 'h': 0.7, 
         'text': 'Prediction Service\nReal-time | Batch Processing'},
        {'x': 6.6, 'y': y_app, 'w': 3.8, 'h': 0.7, 
         'text': 'Visualization Interface\nMap Display | Comparison'},
        {'x': 11.0, 'y': y_app, 'w': 3.8, 'h': 0.7, 
         'text': 'Evaluation System\nADE/FDE | Ablation Study'},
    ]
    
    for box in app_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['app'],
                              edgecolor=colors['border_app'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ============ Layer 5: User Interface ============
    y_ui = 0.8
    
    ui_box = FancyBboxPatch((3.0, y_ui), 12.0, 0.6,
                            boxstyle="round,pad=0.05", 
                            facecolor='#FFECB3',
                            edgecolor='#FF6F00', linewidth=2.5)
    ax.add_patch(ui_box)
    ax.text(9.0, y_ui + 0.3, 
            'User Interface: Web Dashboard | REST API | Command Line Tools', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # ============ Draw Arrows ============
    arrow_style = "Simple,tail_width=0.6,head_width=10,head_length=10"
    
    # Data -> Processing
    for x in [2.2, 5.4, 8.6, 11.8, 15.0]:
        arrow = FancyArrowPatch((x, y_data - 0.1), (x, y_process + 0.8),
                               arrowstyle=arrow_style, color=colors['border_process'],
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    # Processing -> Model
    for x in [2.7, 6.9, 11.1, 15.3]:
        arrow = FancyArrowPatch((x, y_process - 0.1), (x, y_model + 0.8),
                               arrowstyle=arrow_style, color=colors['border_model'],
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    # Model -> Application
    for x in [4.1, 9.0, 12.9]:
        arrow = FancyArrowPatch((x, y_model - 1.4), (x, y_app + 0.8),
                               arrowstyle=arrow_style, color=colors['border_app'],
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    # Application -> User Interface
    for x in [4.1, 8.5, 12.9]:
        arrow = FancyArrowPatch((x, y_app - 0.1), (x, y_ui + 0.7),
                               arrowstyle=arrow_style, color='#FF6F00',
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    # ============ Legend ============
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor=colors['border_data'], 
                      label='Data Layer - Multi-source Geographic Data'),
        mpatches.Patch(facecolor=colors['process'], edgecolor=colors['border_process'], 
                      label='Processing Layer - Data Preprocessing & Generation'),
        mpatches.Patch(facecolor=colors['model'], edgecolor=colors['border_model'], 
                      label='Model Layer - TerraTNT Deep Learning Model'),
        mpatches.Patch(facecolor=colors['app'], edgecolor=colors['border_app'], 
                      label='Application Layer - Prediction & Evaluation'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.95, edgecolor='black', fancybox=True)
    
    # ============ System Features ============
    feature_box = FancyBboxPatch((16.8, 9.5), 1.0, 2.2,
                                boxstyle="round,pad=0.08", 
                                facecolor='wheat', alpha=0.8,
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(feature_box)
    
    ax.text(17.3, 11.4, 'Key Features', fontsize=10, fontweight='bold', ha='center')
    features = [
        'Multi-region',
        'Parallel',
        'GPU Accel.',
        'Real-time',
        'Scalable'
    ]
    for i, feature in enumerate(features):
        ax.text(17.3, 11.0 - i*0.35, f'✓ {feature}', fontsize=8.5, ha='center')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("Generating system architecture diagram...")
    
    fig = draw_system_architecture()
    
    # Save as PNG
    output_path = '/home/zmc/文档/programwork/docs/system_architecture_en.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PNG saved: {output_path}")
    
    # Save as PDF
    pdf_path = '/home/zmc/文档/programwork/docs/system_architecture_en.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF saved: {pdf_path}")
    
    plt.close()
    print("✅ Done!")
