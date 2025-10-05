import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Create a visual representation of the truss structure
def draw_truss_structure():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Node coordinates (from your code)
    nodes = {
        1: (0.0, 0.0),    # Bottom left
        3: (6.0, 0.0),    # Bottom 
        5: (12.0, 0.0),   # Bottom center
        7: (18.0, 0.0),   # Bottom
        9: (24.0, 0.0),   # Bottom
        11: (30.0, 0.0),  # Bottom right
        2: (3.0, 4.5),    # Top
        4: (9.0, 4.5),    # Top
        6: (15.0, 4.5),   # Top center
        8: (21.0, 4.5),   # Top
        10: (27.0, 4.5)   # Top
    }
    
    # Element connections (from your code)
    # Bottom chord elements (Elements 1-5)
    bottom_connections = [(1,3), (3,5), (5,7), (7,9), (9,11)]
    # Top chord elements (Elements 6-9) 
    top_connections = [(2,4), (4,6), (6,8), (8,10)]
    # Web elements (Elements 10-19)
    web_connections = [(1,2), (2,3), (3,4), (4,5), (5,6), 
                       (6,7), (7,8), (8,9), (9,10), (10,11)]
    
    # Draw bottom chord (thick blue lines)
    for i, (n1, n2) in enumerate(bottom_connections, 1):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=4, label='Bottom Chord' if i == 1 else "")
        # Add element numbers
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y - 0.5, f'E{i}', ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    
    # Draw top chord (thick red lines)
    for i, (n1, n2) in enumerate(top_connections, 6):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=4, label='Top Chord' if i == 6 else "")
        # Add element numbers
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.5, f'E{i}', ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7))
    
    # Draw web members (green lines)
    for i, (n1, n2) in enumerate(web_connections, 10):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2, label='Web Members' if i == 10 else "")
        # Add element numbers for web members
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, f'E{i}', ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='lightgreen', alpha=0.7))
    
    # Draw nodes
    for node_id, (x, y) in nodes.items():
        if node_id in [1, 11]:  # Support nodes
            color = 'black'
            size = 120
        else:
            color = 'darkred'
            size = 80
        
        ax.scatter(x, y, c=color, s=size, zorder=5)
        ax.text(x, y + 0.7, f'N{node_id}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # Add supports
    # Pin support at node 1
    ax.plot(0, -0.8, 's', markersize=15, color='green', label='Pin Support')
    ax.text(0, -1.3, 'Pin Support\n(Fixed X,Y)', ha='center', va='top', fontsize=8)
    
    # Roller support at node 11
    ax.plot(30, -0.8, '^', markersize=15, color='orange', label='Roller Support')
    ax.text(30, -1.3, 'Roller Support\n(Free X, Fixed Y)', ha='center', va='top', fontsize=8)
    
    # Add loads (downward arrows on top nodes)
    load_nodes = [2, 4, 6, 8, 10]
    for node in load_nodes:
        x, y = nodes[node]
        ax.arrow(x, y + 1.5, 0, -1, head_width=0.7, head_length=0.3,
                fc='purple', ec='purple', linewidth=2)
        ax.text(x, y + 2, '10kN', ha='center', va='bottom', color='purple', fontweight='bold')
    
    # Formatting
    ax.set_xlim(-3, 33)
    ax.set_ylim(-3, 8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Truss Structure - Structural Health Monitoring Model\n'
                '19 Elements, 11 Nodes, 50kN Total Load', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    
    # Add dimensions
    ax.annotate('', xy=(30, -2.5), xytext=(0, -2.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(15, -2.8, '30.0 m', ha='center', va='top', fontsize=10)
    
    ax.annotate('', xy=(-1.5, 4.5), xytext=(-1.5, 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(-2.2, 2.25, '4.5 m', ha='center', va='center', rotation=90, fontsize=10)
    
    plt.tight_layout()
    #plt.show()
    
    # Save the figure instead of showing it
    plt.savefig('truss_structure.png', dpi=300, bbox_inches='tight')
    print("Truss diagram saved as 'truss_structure.png'")
    
    # Print element information
    print("TRUSS STRUCTURE DETAILS")
    print("=" * 50)
    print("NODES:")
    print("Bottom Chord: 1, 3, 5, 7, 9, 11 (y = 0.0 m)")
    print("Top Chord:    2, 4, 6, 8, 10    (y = 4.5 m)")
    print("\nELEMENTS:")
    print("Bottom Chord (E1-E5):  Large cross-section (0.01 m²)")
    print("  E1: Node 1→3   E2: Node 3→5   E3: Node 5→7")
    print("  E4: Node 7→9   E5: Node 9→11")
    print("\nTop Chord (E6-E9):     Large cross-section (0.01 m²)")
    print("  E6: Node 2→4   E7: Node 4→6   E8: Node 6→8   E9: Node 8→10")
    print("\nWeb Members (E10-E19): Small cross-section (0.005 m²)")
    print("  E10: Node 1→2   E11: Node 2→3   E12: Node 3→4")
    print("  E13: Node 4→5   E14: Node 5→6   E15: Node 6→7")
    print("  E16: Node 7→8   E17: Node 8→9   E18: Node 9→10  E19: Node 10→11")
    print("\nSUPPORTS:")
    print("  Node 1:  Pin support (fixed in X and Y)")
    print("  Node 11: Roller support (free in X, fixed in Y)")
    print("\nLOADS:")
    print("  10 kN downward at each top node (2, 4, 6, 8, 10)")
    print("  Total applied load: 50 kN")

if __name__ == "__main__":
    # Run the visualization
    draw_truss_structure()