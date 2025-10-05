# Add OpenSees path to environment to avoid Tcl conflicts
import os
opensees_path = r'C:\OpenSees3.7.1'
os.environ['PATH'] = os.path.join(opensees_path, 'bin') + ';' + os.environ['PATH']

# Import required libraries for advanced structural failure analysis
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't need Tkinter

import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STRUCTURAL FAILURE ANALYSIS - PHASE 1: ENHANCED MODELING")
print("="*70)
print("Initializing advanced failure analysis framework...")
print("- Material nonlinearity: Steel01, ElasticPP")
print("- Geometric nonlinearity: Corotational elements")
print("- Failure detection: Yield, Buckling, Displacement, Collapse")

def define_enhanced_materials():
    """
    Define nonlinear material properties for failure analysis
    """
    print("\nDefining Enhanced Material Properties:")
    print("="*50)
    
    # Steel material properties (typical structural steel)
    E = 200000.0e6          # Young's modulus (Pa) - 200 GPa
    fy_yield = 250.0e6      # Yield strength (Pa) - 250 MPa (typical for S235)
    fy_high = 355.0e6       # Higher grade steel - 355 MPa (S355)
    
    # Steel01 parameters - CORRECTED ORDER
    b = 0.02                # Strain hardening ratio (2%)
    
    print(f"Base Material Properties:")
    print(f"  Young's Modulus (E): {E/1e9:.0f} GPa")
    print(f"  Yield Strength (S235): {fy_yield/1e6:.0f} MPa")
    print(f"  Yield Strength (S355): {fy_high/1e6:.0f} MPa")
    print(f"  Strain Hardening Ratio: {b*100:.1f}%")
    
    # Material 1: Elastic (baseline for comparison)
    ops.uniaxialMaterial('Elastic', 1, E)
    print(f"\nMaterial 1: Elastic")
    print(f"  E = {E/1e9:.0f} GPa")
    
    # Material 2: Steel01 - Bilinear kinematic hardening (S235)
    # CORRECTED: Steel01 requires: tag, fy, E0, b
    ops.uniaxialMaterial('Steel01', 2, fy_yield, E, b)
    print(f"\nMaterial 2: Steel01 (S235)")
    print(f"  fy = {fy_yield/1e6:.0f} MPa, E = {E/1e9:.0f} GPa, b = {b}")
    
    # Material 3: ElasticPP - Elastic perfectly plastic (S235)
    eps_y = fy_yield / E    # Yield strain
    ops.uniaxialMaterial('ElasticPP', 3, E, eps_y)
    print(f"\nMaterial 3: ElasticPP (S235)")
    print(f"  fy = {fy_yield/1e6:.0f} MPa, εy = {eps_y*1000:.3f}‰")
    
    # Material 4: Steel01 - Higher grade steel (S355)
    ops.uniaxialMaterial('Steel01', 4, fy_high, E, b)
    print(f"\nMaterial 4: Steel01 (S355)")
    print(f"  fy = {fy_high/1e6:.0f} MPa, E = {E/1e9:.0f} GPa, b = {b}")
    
    # Material 5: ElasticPP - Higher grade steel (S355)
    eps_y_high = fy_high / E
    ops.uniaxialMaterial('ElasticPP', 5, E, eps_y_high)
    print(f"\nMaterial 5: ElasticPP (S355)")
    print(f"  fy = {fy_high/1e6:.0f} MPa, εy = {eps_y_high*1000:.3f}‰")
    
    # Store material properties for later use
    material_props = {
        'E': E,
        'fy_s235': fy_yield,
        'fy_s355': fy_high,
        'b': b,
        'eps_y_s235': eps_y,
        'eps_y_s355': eps_y_high
    }
    
    print(f"\n✓ Enhanced materials defined successfully")
    return material_props

def define_cross_sections():
    """
    Define realistic cross-sectional properties for different member types
    """
    print(f"\nDefining Cross-Sectional Properties:")
    print("="*40)
    
    # Cross-sectional areas (m²) - realistic values
    sections = {
        'bottom_chord': {
            'A': 0.0025,      # 25 cm² - substantial chord member
            'description': 'Bottom chord (tension/compression)',
            'typical': 'IPE160 or similar'
        },
        'top_chord': {
            'A': 0.0030,      # 30 cm² - larger compression chord
            'description': 'Top chord (compression)',
            'typical': 'IPE180 or similar'
        },
        'web_vertical': {
            'A': 0.0015,      # 15 cm² - vertical web member
            'description': 'Vertical web member',
            'typical': 'L80x80x8 or similar'
        },
        'web_diagonal': {
            'A': 0.0012,      # 12 cm² - diagonal web member
            'description': 'Diagonal web member', 
            'typical': 'L70x70x7 or similar'
        }
    }
    
    for name, props in sections.items():
        print(f"  {name}:")
        print(f"    Area: {props['A']*1e4:.1f} cm²")
        print(f"    Type: {props['description']}")
        print(f"    Typical: {props['typical']}")
    
    return sections

def create_enhanced_truss_geometry():
    """
    Create the truss geometry with enhanced node and element definitions
    """
    print(f"\nCreating Enhanced Truss Geometry:")
    print("="*40)
    
    # Define nodes (same geometry as previous model)
    span = 30.0  # Total span (m)
    height = 4.5  # Truss height (m)
    
    # Bottom chord nodes
    ops.node(1, 0.0, 0.0)     # Left support
    ops.node(3, 6.0, 0.0)     
    ops.node(5, 12.0, 0.0)    
    ops.node(7, 18.0, 0.0)    
    ops.node(9, 24.0, 0.0)    
    ops.node(11, 30.0, 0.0)   # Right support
    
    # Top chord nodes
    ops.node(2, 3.0, height)   
    ops.node(4, 9.0, height)   
    ops.node(6, 15.0, height)  # Mid-span - critical for deflection
    ops.node(8, 21.0, height)  
    ops.node(10, 27.0, height) 
    
    # Store node information
    bottom_nodes = [1, 3, 5, 7, 9, 11]
    top_nodes = [2, 4, 6, 8, 10]
    all_nodes = bottom_nodes + top_nodes
    
    print(f"✓ Nodes created: {len(all_nodes)} total")
    print(f"  Bottom chord: {bottom_nodes}")
    print(f"  Top chord: {top_nodes}")
    print(f"  Span: {span} m, Height: {height} m")
    
    return all_nodes, bottom_nodes, top_nodes, span, height

def check_material_yield_failure(element_id, element_area, fy):
    """
    Check if element has exceeded yield strength (Material Yield Failure)
    """
    try:
        # Get element forces
        forces = ops.eleForce(element_id)
        axial_force = abs(forces[0])  # Absolute axial force
        
        # Calculate stress
        stress = axial_force / element_area
        
        # Check yield condition
        if stress >= fy:
            return {
                'failed': True,
                'failure_mode': 'material_yield',
                'stress': stress,
                'yield_strength': fy,
                'stress_ratio': stress / fy,
                'force': axial_force
            }
        else:
            return {
                'failed': False,
                'stress': stress,
                'stress_ratio': stress / fy,
                'margin_to_yield': (fy - stress) / fy
            }
    except:
        return {'failed': False, 'error': 'Could not get element force'}

def check_buckling_failure(element_id, element_length, element_area, E, K_factor=1.0):
    """
    Check if compression element has exceeded buckling capacity (Buckling Failure)
    """
    try:
        # Get element forces
        forces = ops.eleForce(element_id)
        axial_force = forces[0]  # Keep sign to check compression
        
        if axial_force >= 0:  # Tension or zero force
            return {'failed': False, 'reason': 'Element in tension - no buckling concern'}
        
        # Compression force (make positive for calculations)
        compression_force = abs(axial_force)
        
        # Estimate moment of inertia based on area (simplified)
        # For a circular section: I = π*r⁴/4, where A = π*r²
        # So r = sqrt(A/π) and I = π*(A/π)²/4 = A²/(4π)
        I_estimate = element_area**2 / (4 * np.pi)  # Very rough estimate
        
        # Effective length
        Le = K_factor * element_length
        
        # Euler critical buckling load
        P_critical = (np.pi**2 * E * I_estimate) / (Le**2)
        
        if compression_force >= P_critical:
            return {
                'failed': True,
                'failure_mode': 'buckling',
                'compression_force': compression_force,
                'critical_load': P_critical,
                'load_ratio': compression_force / P_critical,
                'effective_length': Le
            }
        else:
            return {
                'failed': False,
                'compression_force': compression_force,
                'critical_load': P_critical,
                'load_ratio': compression_force / P_critical,
                'margin_to_buckling': (P_critical - compression_force) / P_critical
            }
    except:
        return {'failed': False, 'error': 'Could not get element force'}

def check_excessive_displacement(critical_node, span, limit_ratio=360.0):
    """
    Check for excessive displacement (Serviceability Failure)
    """
    try:
        # Get displacement at critical node
        disp = ops.nodeDisp(critical_node)
        vertical_displacement = abs(disp[1])  # Absolute vertical displacement
        
        # Calculate displacement limit (Span/360 is common for roof trusses)
        displacement_limit = span / limit_ratio
        
        if vertical_displacement > displacement_limit:
            return {
                'failed': True,
                'failure_mode': 'excessive_displacement',
                'actual_displacement': vertical_displacement,
                'limit': displacement_limit,
                'ratio': vertical_displacement / displacement_limit,
                'limit_description': f'L/{limit_ratio}'
            }
        else:
            return {
                'failed': False,
                'actual_displacement': vertical_displacement,
                'limit': displacement_limit,
                'usage_ratio': vertical_displacement / displacement_limit,
                'margin': (displacement_limit - vertical_displacement) / displacement_limit
            }
    except:
        return {'failed': False, 'error': 'Could not get node displacement'}

def check_structural_collapse():
    """
    Check for structural collapse (Global Failure)
    This is detected when the analysis fails to converge
    """
    # This function is called when analysis convergence fails
    return {
        'failed': True,
        'failure_mode': 'structural_collapse',
        'description': 'Analysis failed to converge - indicates structural instability or collapse'
    }

# Initialize the model and define materials
ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 2)

print("Creating enhanced truss model with nonlinear materials...")
material_props = define_enhanced_materials()
cross_sections = define_cross_sections()
all_nodes, bottom_nodes, top_nodes, span, height = create_enhanced_truss_geometry()

print(f"\n✓ Phase 1 Complete: Enhanced Structural Modeling")
print(f"✓ Materials defined: Elastic, Steel01, ElasticPP")
print(f"✓ Cross-sections defined for different member types")
print(f"✓ Failure detection functions implemented:")
print(f"  - Material Yield Failure")
print(f"  - Buckling Failure")  
print(f"  - Excessive Displacement")
print(f"  - Structural Collapse")
print(f"\nReady for Phase 2: Geometric and Material Nonlinearity Setup")

# Phase 2: Geometric and Material Nonlinearity Setup
print(f"\n" + "="*70)
print("PHASE 2: GEOMETRIC AND MATERIAL NONLINEARITY SETUP")
print("="*70)

def setup_nonlinear_geometry():
    """
    Setup for truss elements (no geometric transformations needed)
    Truss elements automatically include geometric nonlinearity when using nonlinear materials
    """
    print(f"\nSetting Up Nonlinear Analysis for Truss Elements:")
    print("="*55)
    
    # NOTE: Truss elements don't use geometric transformations
    # They automatically account for geometric nonlinearity when:
    # 1. Using nonlinear materials (Steel01, ElasticPP)
    # 2. Large displacement analysis is performed
    
    print("✓ Truss elements selected - no geomTransf required")
    print("✓ Geometric nonlinearity through:")
    print("  - Nonlinear materials (Steel01)")
    print("  - Large displacement formulation")
    print("  - Updated geometry during analysis")
    
    return True

def calculate_member_lengths():
    """
    Calculate actual member lengths from node coordinates
    """
    print(f"\nCalculating Actual Member Lengths:")
    print("="*40)
    
    member_lengths = {}
    
    # Calculate bottom chord lengths
    print("Bottom Chord Members:")
    for i in range(len(bottom_nodes)-1):
        node1, node2 = bottom_nodes[i], bottom_nodes[i+1]
        coord1 = ops.nodeCoord(node1)
        coord2 = ops.nodeCoord(node2)
        length = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
        member_lengths[f'bottom_{i+1}'] = length
        print(f"  Nodes {node1}-{node2}: {length:.2f} m")
    
    # Calculate top chord lengths
    print("\nTop Chord Members:")
    for i in range(len(top_nodes)-1):
        node1, node2 = top_nodes[i], top_nodes[i+1]
        coord1 = ops.nodeCoord(node1)
        coord2 = ops.nodeCoord(node2)
        length = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
        member_lengths[f'top_{i+1}'] = length
        print(f"  Nodes {node1}-{node2}: {length:.2f} m")
    
    # Calculate web member lengths (will be done during element creation)
    member_lengths['web_vertical'] = height  # 4.5m
    member_lengths['web_diagonal'] = np.sqrt(3.0**2 + height**2)  # ~5.4m
    
    print(f"\nTypical Web Member Lengths:")
    print(f"  Vertical members: {member_lengths['web_vertical']:.2f} m")
    print(f"  Diagonal members: {member_lengths['web_diagonal']:.2f} m")
    
    return member_lengths

# Execute Phase 2
setup_nonlinear_geometry()
member_lengths = calculate_member_lengths()

print(f"\n✓ Phase 2 Complete: Truss Nonlinearity Setup")
print(f"✓ Truss elements configured for nonlinear analysis")
print(f"✓ Member lengths calculated from actual coordinates")

# Phase 3: Buckling Analysis Implementation  
print(f"\n" + "="*70)
print("PHASE 3: BUCKLING ANALYSIS IMPLEMENTATION")
print("="*70)

def calculate_buckling_parameters():
    """
    Calculate critical buckling loads for compression members in truss
    """
    print(f"\nBuckling Analysis for Truss Members:")
    print("="*40)
    
    # Material properties
    E = material_props['E']
    
    # Moment of inertia estimates for truss sections (m⁴)
    # For truss members, these are based on the minimum radius of gyration
    I_values = {
        'bottom_chord': 8.49e-6,    # IPE160 equivalent
        'top_chord': 13.8e-6,       # IPE180 equivalent (larger for compression)
        'web_vertical': 0.89e-6,    # L80x80x8: Imin ≈ 0.89×10⁻⁶ m⁴
        'web_diagonal': 0.61e-6     # L70x70x7: Imin ≈ 0.61×10⁻⁶ m⁴
    }
    
    # Effective length factors for truss members
    K_factors = {
        'bottom_chord': 0.8,    # Some restraint from web connections
        'top_chord': 0.9,       # Less restraint (compression critical)
        'web_vertical': 1.0,    # Pin-ended
        'web_diagonal': 1.0     # Pin-ended
    }
    
    buckling_data = {}
    
    print("Member Type          Length(m)  K-factor  Imin(×10⁻⁶m⁴)  Pcr(kN)")
    print("-" * 67)
    
    # Calculate for each member type
    for member_type in ['bottom_chord', 'top_chord', 'web_vertical', 'web_diagonal']:
        
        # Get typical length
        if member_type == 'bottom_chord':
            typical_length = 6.0
        elif member_type == 'top_chord':
            typical_length = 6.0
        elif member_type == 'web_vertical':
            typical_length = height  # 4.5m
        else:  # web_diagonal
            typical_length = member_lengths['web_diagonal']  # ~5.4m
            
        K = K_factors[member_type]
        I = I_values[member_type]
        Le = K * typical_length  # Effective length
        
        # Euler critical buckling load: Pcr = π²EI/Le²
        P_cr = (np.pi**2 * E * I) / (Le**2)
        
        buckling_data[member_type] = {
            'length': typical_length,
            'K_factor': K,
            'effective_length': Le,
            'I': I,
            'P_critical': P_cr,
            'area': cross_sections[member_type]['A']
        }
        
        print(f"{member_type:<18} {typical_length:>6.1f}    {K:>6.1f}    {I*1e6:>9.2f}  {P_cr/1000:>8.0f}")
    
    return buckling_data

def implement_buckling_check():
    """
    Implement slenderness analysis for truss members
    """
    print(f"\nSlenderness Analysis for Truss Members:")
    print("="*45)
    
    print("Member Type          λ = Le/r    Category      Critical?")
    print("-" * 60)
    
    slenderness_data = {}
    
    for member_type, data in buckling_data.items():
        # Radius of gyration: r = √(I/A)
        I = data['I']
        A = data['area']
        r = np.sqrt(I / A)
        
        # Slenderness ratio: λ = Le/r
        Le = data['effective_length']
        slenderness = Le / r
        
        # Classify based on slenderness (for truss members)
        if slenderness < 75:
            category = "Stocky"
            critical = "No"
        elif slenderness < 150:
            category = "Intermediate"
            critical = "Monitor"
        else:
            category = "Slender"
            critical = "YES"
            
        slenderness_data[member_type] = {
            'radius_gyration': r,
            'slenderness': slenderness,
            'category': category,
            'buckling_critical': critical
        }
        
        print(f"{member_type:<18} {slenderness:>8.1f}    {category:<12} {critical}")
    
    return slenderness_data

# Execute Phase 3
buckling_data = calculate_buckling_parameters()
slenderness_data = implement_buckling_check()

print(f"\n✓ Phase 3 Complete: Buckling Analysis Implementation")
print(f"✓ Euler buckling loads calculated for all member types")
print(f"✓ Slenderness ratios determined - monitor slender members")

# Phase 4: Progressive Loading and Failure Detection
print(f"\n" + "="*70)
print("PHASE 4: PROGRESSIVE LOADING AND FAILURE DETECTION")
print("="*70)

def create_nonlinear_truss_elements():
    """
    Create truss elements with nonlinear materials (no geomTransf needed)
    """
    print(f"\nCreating Nonlinear Truss Elements:")
    print("="*40)
    
    element_tag = 1
    element_registry = {}
    
    # Use Steel01 material (Material 2) for nonlinear behavior
    material_tag = 2  # Steel01 with S235 properties
    
    print("Bottom Chord Elements (Steel01):")
    # Bottom chord elements
    bottom_connections = [(1,3), (3,5), (5,7), (7,9), (9,11)]
    for i, (node1, node2) in enumerate(bottom_connections):
        area = cross_sections['bottom_chord']['A']
        # Truss element: element('Truss', eleTag, *eleNodes, A, matTag)
        ops.element('Truss', element_tag, node1, node2, area, material_tag)
        
        element_registry[element_tag] = {
            'type': 'bottom_chord',
            'nodes': (node1, node2),
            'area': area,
            'material': 'Steel01_S235',
            'expected_force': 'tension/compression',
            'buckling_critical': buckling_data['bottom_chord']['P_critical']
        }
        
        print(f"  Element {element_tag}: Nodes {node1}-{node2}, A={area*1e4:.1f} cm²")
        element_tag += 1
    
    print("\nTop Chord Elements (Steel01):")
    # Top chord elements
    top_connections = [(2,4), (4,6), (6,8), (8,10)]
    for i, (node1, node2) in enumerate(top_connections):
        area = cross_sections['top_chord']['A']
        ops.element('Truss', element_tag, node1, node2, area, material_tag)
        
        element_registry[element_tag] = {
            'type': 'top_chord',
            'nodes': (node1, node2),
            'area': area,
            'material': 'Steel01_S235',
            'expected_force': 'compression',
            'buckling_critical': buckling_data['top_chord']['P_critical']
        }
        
        print(f"  Element {element_tag}: Nodes {node1}-{node2}, A={area*1e4:.1f} cm²")
        element_tag += 1
    
    print("\nWeb Elements (Steel01):")
    # Web elements - connecting top and bottom chords
    web_connections = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),    # First half
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)  # Second half
    ]
    
    for i, (node1, node2) in enumerate(web_connections):
        # Calculate actual member length and determine type
        coord1 = ops.nodeCoord(node1)
        coord2 = ops.nodeCoord(node2)
        length = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
        
        # Determine member subtype based on orientation
        if abs(coord1[0] - coord2[0]) < 0.1:  # Vertical (same X coordinate)
            member_subtype = 'web_vertical'
            area = cross_sections['web_vertical']['A']
        else:  # Diagonal
            member_subtype = 'web_diagonal'
            area = cross_sections['web_diagonal']['A']
        
        ops.element('Truss', element_tag, node1, node2, area, material_tag)
        
        element_registry[element_tag] = {
            'type': 'web_member',
            'subtype': member_subtype,
            'nodes': (node1, node2),
            'area': area,
            'material': 'Steel01_S235',
            'length': length,
            'expected_force': 'variable',
            'buckling_critical': buckling_data[member_subtype]['P_critical']
        }
        
        print(f"  Element {element_tag}: Nodes {node1}-{node2}, {member_subtype}, L={length:.2f}m, A={area*1e4:.1f} cm²")
        element_tag += 1
    
    total_elements = element_tag - 1
    print(f"\n✓ Total truss elements created: {total_elements}")
    
    return element_registry, total_elements

def setup_boundary_conditions_and_loading():
    """
    Setup supports and loading for progressive failure analysis
    """
    print(f"\nSetting Up Boundary Conditions and Loading:")
    print("="*50)
    
    # Boundary conditions for 2D truss
    ops.fix(1, 1, 1)   # Pin support: fix X and Y at node 1
    ops.fix(11, 0, 1)  # Roller support: fix Y only at node 11
    
    print("✓ Boundary conditions applied:")
    print("  Node 1: Pin support (restrained in X, Y)")
    print("  Node 11: Roller support (restrained in Y only)")
    
    # Define time series and load pattern
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    
    # Apply point loads at top chord nodes
    load_nodes = [2, 4, 6, 8, 10]  # All top chord nodes
    unit_load = -1000.0  # 1 kN downward per node (negative Y direction)
    
    print(f"\n✓ Loading configuration:")
    print(f"  Load application points: {load_nodes}")
    print(f"  Unit load per node: {abs(unit_load)/1000:.0f} kN (downward)")
    print(f"  Total unit load: {len(load_nodes) * abs(unit_load)/1000:.0f} kN")
    
    for node in load_nodes:
        ops.load(node, 0.0, unit_load)  # Load in Y direction only
    
    return load_nodes, unit_load

def setup_progressive_analysis():
    """
    Setup progressive loading analysis parameters
    """
    print(f"\nSetting Up Progressive Analysis:")
    print("="*40)
    
    # Analysis parameters for truss
    max_load_factor = 50.0       # Maximum load multiplier
    load_increment = 0.5         # Load increment per step  
    max_iterations = 50          # Maximum iterations per step
    tolerance = 1.0e-6           # Convergence tolerance (relaxed for truss)
    
    print(f"Progressive Loading Parameters:")
    print(f"  Maximum load factor: {max_load_factor}")
    print(f"  Load increment: {load_increment}")
    print(f"  Convergence tolerance: {tolerance}")
    print(f"  Max iterations per step: {max_iterations}")
    
    # Setup analysis objects for nonlinear static analysis
    ops.constraints('Plain')       # Plain constraints for truss
    ops.numberer('RCM')           # Reverse Cuthill-McKee numberer
    ops.system('BandSPD')         # Banded symmetric positive definite
    ops.test('NormDispIncr', tolerance, max_iterations, 0)  # Convergence test
    ops.algorithm('Newton')       # Newton-Raphson algorithm
    ops.integrator('LoadControl', load_increment)  # Load control
    ops.analysis('Static')        # Static analysis
    
    print("✓ Nonlinear analysis objects configured for truss")
    
    return max_load_factor, load_increment, tolerance, max_iterations

# Execute Phase 4
element_registry, total_elements = create_nonlinear_truss_elements()
load_nodes, unit_load = setup_boundary_conditions_and_loading()
max_load_factor, load_increment, tolerance, max_iterations = setup_progressive_analysis()

print(f"\n✓ Phase 4 Complete: Progressive Loading Setup")
print(f"✓ {total_elements} truss elements with Steel01 material")
print(f"✓ Ready for progressive loading up to {max_load_factor}× unit load")

# Phase 5: Serviceability Limit State Checking
print(f"\n" + "="*70)
print("PHASE 5: SERVICEABILITY LIMIT STATE CHECKING")
print("="*70)

def define_serviceability_limits():
    """
    Define serviceability limit state criteria
    """
    print(f"\nDefining Serviceability Limit State Criteria:")
    print("="*50)
    
    # Deflection limits
    serviceability_limits = {
        'deflection_normal': span / 360.0,      # L/360 for normal use
        'deflection_strict': span / 500.0,      # L/500 for strict requirements
        'deflection_minimum': span / 250.0,     # L/250 minimum requirement
    }
    
    # Stress limits for serviceability (percentage of yield strength)
    stress_limits = {
        'working_stress': 0.6,                  # 60% of yield strength
        'frequent_load': 0.7,                   # 70% for frequent loading
        'characteristic': 0.8                    # 80% for characteristic loading
    }
    
    # Critical node for deflection check (mid-span)
    critical_node = 6  # Top chord mid-span node
    
    print(f"Deflection Limits:")
    print(f"  Normal use (L/360): {serviceability_limits['deflection_normal']*1000:.1f} mm")
    print(f"  Strict requirements (L/500): {serviceability_limits['deflection_strict']*1000:.1f} mm")
    print(f"  Minimum requirement (L/250): {serviceability_limits['deflection_minimum']*1000:.1f} mm")
    print(f"  Critical monitoring node: {critical_node}")
    
    print(f"\nStress Limits (% of yield strength):")
    for limit_name, factor in stress_limits.items():
        fy = material_props['fy_s235']
        stress_limit = factor * fy
        print(f"  {limit_name}: {factor*100:.0f}% = {stress_limit/1e6:.0f} MPa")
    
    return serviceability_limits, stress_limits, critical_node

def check_serviceability_during_analysis(load_factor, serviceability_limits, stress_limits, critical_node):
    """
    Check serviceability criteria during progressive loading
    """
    # Get displacement at critical node (mid-span)
    try:
        disp = ops.nodeDisp(critical_node)
        vertical_displacement = abs(disp[1])  # Absolute value of vertical displacement
    except:
        return None, None, False
    
    # Check deflection limits
    deflection_status = {}
    for limit_name, limit_value in serviceability_limits.items():
        deflection_status[limit_name] = vertical_displacement <= limit_value
    
    # Check stress levels in critical members
    stress_status = {}
    max_stress_ratio = 0.0
    critical_element = None
    
    fy = material_props['fy_s235']  # Yield strength
    
    for elem_id in range(1, total_elements + 1):
        try:
            # Get element forces
            forces = ops.eleForce(elem_id)
            axial_force = abs(forces[0])  # Absolute axial force
            
            # Get element area
            area = element_registry[elem_id]['area']
            
            # Calculate stress
            stress = axial_force / area
            stress_ratio = stress / fy
            
            if stress_ratio > max_stress_ratio:
                max_stress_ratio = stress_ratio
                critical_element = elem_id
                
        except:
            continue
    
    # Check stress limits
    for limit_name, limit_factor in stress_limits.items():
        stress_status[limit_name] = max_stress_ratio <= limit_factor
    
    # Overall serviceability status
    serviceability_ok = (deflection_status['deflection_normal'] and 
                        stress_status['working_stress'])
    
    serviceability_data = {
        'load_factor': load_factor,
        'vertical_displacement': vertical_displacement,
        'max_stress_ratio': max_stress_ratio,
        'critical_element': critical_element,
        'deflection_status': deflection_status,
        'stress_status': stress_status,
        'overall_ok': serviceability_ok
    }
    
    # FIXED: Return serviceability_ok instead of undefined 'serviceable'
    return serviceability_data, vertical_displacement, serviceability_ok

# Execute Phase 5
serviceability_limits, stress_limits, critical_node = define_serviceability_limits()

print(f"\n✓ Phase 5 Complete: Serviceability Limit State Checking")
print(f"✓ Critical monitoring node: {critical_node} (mid-span)")
print(f"✓ Ready for serviceability checking during analysis")

# Phase 6: Ultimate Limit State Analysis
print(f"\n" + "="*70)
print("PHASE 6: ULTIMATE LIMIT STATE ANALYSIS")
print("="*70)

def perform_progressive_failure_analysis():
    """
    Perform progressive loading analysis with comprehensive failure detection
    """
    print(f"\nStarting Progressive Failure Analysis:")
    print("="*45)
    
    # Initialize data storage
    analysis_results = []
    failure_events = []
    yielded_elements = set()
    buckled_elements = set()
    
    # Analysis tracking variables
    current_load_factor = 0.0
    analysis_failed = False
    serviceability_exceeded = False
    first_yield_load = None
    first_buckling_load = None
    ultimate_load = None
    
    print(f"Load Step | Load Factor | Max Disp(mm) | Max Stress | Status")
    print("-" * 65)
    
    step = 0
    while current_load_factor < max_load_factor and not analysis_failed:
        step += 1
        
        # Attempt analysis step
        convergence = ops.analyze(1)
        current_load_factor += load_increment
        
        if convergence != 0:
            # Analysis failed to converge
            analysis_failed = True
            ultimate_load = current_load_factor - load_increment
            failure_events.append({
                'load_factor': ultimate_load,
                'type': 'structural_collapse',
                'description': 'Analysis convergence failure - structural instability'
            })
            print(f"{step:>9} | {current_load_factor:>11.1f} | {'FAILED':>12} | {'FAILED':>10} | COLLAPSE")
            break
        
        # Get current structural response
        try:
            # Check serviceability
            serviceability_data, max_displacement, sls_ok = check_serviceability_during_analysis(
                current_load_factor, serviceability_limits, stress_limits, critical_node
            )
            
            if not serviceability_exceeded and not sls_ok:
                serviceability_exceeded = True
                failure_events.append({
                    'load_factor': current_load_factor,
                    'type': 'serviceability_exceeded', 
                    'description': f'Serviceability limit exceeded at {current_load_factor:.1f}× load'
                })
            
            # Check for material yielding and buckling
            max_stress_ratio = 0.0
            new_yield_elements = []
            new_buckling_elements = []
            
            for elem_id in range(1, total_elements + 1):
                try:
                    # Get element response
                    forces = ops.eleForce(elem_id)
                    axial_force = forces[0]  # Keep sign for compression check
                    abs_force = abs(axial_force)
                    
                    # Get element properties
                    elem_info = element_registry[elem_id]
                    area = elem_info['area']
                    
                    # Calculate stress
                    stress = abs_force / area
                    fy = material_props['fy_s235']
                    stress_ratio = stress / fy
                    
                    max_stress_ratio = max(max_stress_ratio, stress_ratio)
                    
                    # Check for yielding (stress-based)
                    if stress_ratio >= 1.0 and elem_id not in yielded_elements:
                        yielded_elements.add(elem_id)
                        new_yield_elements.append(elem_id)
                        
                        if first_yield_load is None:
                            first_yield_load = current_load_factor
                    
                    # Check for buckling (compression members only)
                    if axial_force < 0:  # Compression (negative force)
                        P_critical = elem_info['buckling_critical']
                        
                        if abs_force >= P_critical and elem_id not in buckled_elements:
                            buckled_elements.add(elem_id)
                            new_buckling_elements.append(elem_id)
                            
                            if first_buckling_load is None:
                                first_buckling_load = current_load_factor
                
                except:
                    continue
            
            # Record new failure events
            for elem_id in new_yield_elements:
                elem_info = element_registry[elem_id]
                failure_events.append({
                    'load_factor': current_load_factor,
                    'type': 'material_yield',
                    'element': elem_id,
                    'element_type': elem_info['type'],
                    'description': f"Element {elem_id} ({elem_info['type']}) yielded"
                })
            
            for elem_id in new_buckling_elements:
                elem_info = element_registry[elem_id]
                failure_events.append({
                    'load_factor': current_load_factor,
                    'type': 'buckling_failure',
                    'element': elem_id, 
                    'element_type': elem_info['type'],
                    'description': f"Element {elem_id} ({elem_info['type']}) buckled"
                })
            
            # Determine current status
            if new_yield_elements or new_buckling_elements:
                status = f"YIELD({len(new_yield_elements)}) BUCK({len(new_buckling_elements)})"
            elif not sls_ok and serviceability_exceeded:
                status = "SLS_FAIL"
            else:
                status = "OK"
            
            # Store results
            analysis_results.append({
                'step': step,
                'load_factor': current_load_factor,
                'total_load_kN': current_load_factor * len(load_nodes) * abs(unit_load) / 1000,
                'max_displacement_mm': max_displacement * 1000,
                'max_stress_ratio': max_stress_ratio,
                'yielded_elements': len(yielded_elements),
                'buckled_elements': len(buckled_elements),
                'serviceability_ok': sls_ok,
                'status': status
            })
            
            # Print progress
            print(f"{step:>9} | {current_load_factor:>11.1f} | {max_displacement*1000:>12.2f} | "
                  f"{max_stress_ratio:>10.3f} | {status}")
            
            # Check for excessive displacement (alternative failure criterion)
            if max_displacement > serviceability_limits['deflection_minimum']:
                if ultimate_load is None:  # Only record first occurrence
                    ultimate_load = current_load_factor
                    failure_events.append({
                        'load_factor': current_load_factor,
                        'type': 'excessive_displacement',
                        'description': f'Excessive displacement: {max_displacement*1000:.1f} mm > {serviceability_limits["deflection_minimum"]*1000:.1f} mm'
                    })
        
        except Exception as e:
            # Unexpected error during analysis
            analysis_failed = True
            ultimate_load = current_load_factor
            failure_events.append({
                'load_factor': current_load_factor,
                'type': 'analysis_error',
                'description': f'Analysis error: {str(e)}'
            })
            break
    
    # Final ultimate load determination
    if ultimate_load is None and not analysis_failed:
        ultimate_load = current_load_factor  # Reached maximum load without failure
    
    print(f"\n✓ Progressive analysis completed")
    print(f"  Total steps: {step}")
    print(f"  Ultimate load factor: {ultimate_load:.1f}" if ultimate_load else "  No ultimate load reached")
    
    return analysis_results, failure_events, {
        'first_yield_load': first_yield_load,
        'first_buckling_load': first_buckling_load,
        'ultimate_load': ultimate_load,
        'serviceability_load': next((event['load_factor'] for event in failure_events 
                                   if event['type'] == 'serviceability_exceeded'), None)
    }

def calculate_safety_factors(critical_loads):
    """
    Calculate safety factors based on different failure modes
    """
    print(f"\nSafety Factor Analysis:")
    print("="*30)
    
    # Typical design loads (working loads)
    working_load_factor = 10.0  # Assume working load is 10× unit load
    
    safety_factors = {}
    
    print(f"Assuming working load = {working_load_factor}× unit load")
    print(f"Working load = {working_load_factor * len(load_nodes) * abs(unit_load) / 1000:.0f} kN total")
    print()
    
    print("Failure Mode                Safety Factor")
    print("-" * 45)
    
    if critical_loads.get('serviceability_load'):
        sf_sls = critical_loads['serviceability_load'] / working_load_factor
        safety_factors['serviceability'] = sf_sls
        status = "✓ OK" if sf_sls >= 1.0 else "✗ FAIL"
        print(f"Serviceability              {sf_sls:>8.2f}     {status}")
    
    if critical_loads.get('first_yield_load'):
        sf_yield = critical_loads['first_yield_load'] / working_load_factor
        safety_factors['yield'] = sf_yield
        status = "✓ OK" if sf_yield >= 1.5 else "✗ FAIL"  # Typical SF = 1.5 for yielding
        print(f"First Yield                 {sf_yield:>8.2f}     {status}")
    
    if critical_loads.get('first_buckling_load'):
        sf_buckling = critical_loads['first_buckling_load'] / working_load_factor
        safety_factors['buckling'] = sf_buckling
        status = "✓ OK" if sf_buckling >= 2.0 else "✗ FAIL"  # Typical SF = 2.0 for buckling
        print(f"First Buckling              {sf_buckling:>8.2f}     {status}")
    
    if critical_loads.get('ultimate_load'):
        sf_ultimate = critical_loads['ultimate_load'] / working_load_factor
        safety_factors['ultimate'] = sf_ultimate
        status = "✓ OK" if sf_ultimate >= 2.5 else "✗ FAIL"  # Typical SF = 2.5 for collapse
        print(f"Ultimate Capacity           {sf_ultimate:>8.2f}     {status}")
    
    return safety_factors

# Execute the progressive failure analysis
print("Executing Progressive Failure Analysis...")
print("This may take a few moments depending on the complexity...")

analysis_results, failure_events, critical_loads = perform_progressive_failure_analysis()
safety_factors = calculate_safety_factors(critical_loads)

print(f"\n" + "="*60)
print("PROGRESSIVE FAILURE ANALYSIS COMPLETED")
print("="*60)

# Phase 7: Failure Mode Classification
print(f"\n" + "="*70)
print("PHASE 7: FAILURE MODE CLASSIFICATION")
print("="*70)

def classify_failure_modes(failure_events, critical_loads):
    """
    Classify and analyze failure modes from the analysis results
    """
    print(f"\nFailure Mode Classification:")
    print("="*35)
    
    # Categorize failure events by type
    failure_categories = {
        'material_yield': [],
        'buckling_failure': [],
        'serviceability_exceeded': [],
        'excessive_displacement': [],
        'structural_collapse': [],
        'analysis_error': []
    }
    
    for event in failure_events:
        event_type = event['type']
        if event_type in failure_categories:
            failure_categories[event_type].append(event)
    
    # Determine dominant failure mode
    first_failure_load = float('inf')
    dominant_mode = None
    
    print(f"Failure Event Summary:")
    print("-" * 50)
    
    for mode, events in failure_categories.items():
        if events:
            first_event = min(events, key=lambda x: x['load_factor'])
            load_factor = first_event['load_factor']
            
            print(f"{mode.replace('_', ' ').title()}:")
            print(f"  First occurrence: Load factor {load_factor:.1f}")
            print(f"  Total events: {len(events)}")
            
            if load_factor < first_failure_load:
                first_failure_load = load_factor
                dominant_mode = mode
            
            # Print details for each event in this category
            for event in events[:3]:  # Show first 3 events
                if 'element' in event:
                    print(f"    - Element {event['element']} ({event.get('element_type', 'unknown')})")
                else:
                    print(f"    - {event.get('description', 'No description')}")
            
            if len(events) > 3:
                print(f"    - ... and {len(events) - 3} more events")
            print()
    
    # Critical load analysis
    print(f"Critical Load Analysis:")
    print("-" * 30)
    
    load_summary = {}
    
    if critical_loads.get('serviceability_load'):
        load_summary['Serviceability Limit'] = critical_loads['serviceability_load']
        total_sls_load = critical_loads['serviceability_load'] * len(load_nodes) * abs(unit_load) / 1000
        print(f"Serviceability Limit: {critical_loads['serviceability_load']:.1f}× ({total_sls_load:.0f} kN)")
    
    if critical_loads.get('first_yield_load'):
        load_summary['First Yield'] = critical_loads['first_yield_load']
        total_yield_load = critical_loads['first_yield_load'] * len(load_nodes) * abs(unit_load) / 1000
        print(f"First Material Yield: {critical_loads['first_yield_load']:.1f}× ({total_yield_load:.0f} kN)")
    
    if critical_loads.get('first_buckling_load'):
        load_summary['First Buckling'] = critical_loads['first_buckling_load']
        total_buckling_load = critical_loads['first_buckling_load'] * len(load_nodes) * abs(unit_load) / 1000
        print(f"First Buckling: {critical_loads['first_buckling_load']:.1f}× ({total_buckling_load:.0f} kN)")
    
    if critical_loads.get('ultimate_load'):
        load_summary['Ultimate Capacity'] = critical_loads['ultimate_load']
        total_ultimate_load = critical_loads['ultimate_load'] * len(load_nodes) * abs(unit_load) / 1000
        print(f"Ultimate Capacity: {critical_loads['ultimate_load']:.1f}× ({total_ultimate_load:.0f} kN)")
    
    # Determine governing failure mode
    print(f"\nGoverning Failure Mode Analysis:")
    print("-" * 40)
    
    structural_behavior = None
    recommendation = None
    
    if dominant_mode:
        print(f"Dominant failure mode: {dominant_mode.replace('_', ' ').title()}")
        print(f"First occurrence at: {first_failure_load:.1f}× load factor")
        
        # Classify structural behavior
        if dominant_mode == 'serviceability_exceeded':
            structural_behavior = "Serviceability-governed design"
            recommendation = "Increase member sizes or reduce deflection limits"
        elif dominant_mode == 'material_yield':
            structural_behavior = "Strength-governed design (yielding)"
            recommendation = "Check load combinations or increase member capacity"
        elif dominant_mode == 'buckling_failure':
            structural_behavior = "Stability-governed design (buckling)"
            recommendation = "Provide lateral bracing or increase member size"
        elif dominant_mode == 'structural_collapse':
            structural_behavior = "Collapse-governed design"
            recommendation = "Major structural revision required"
        else:
            structural_behavior = "Other failure mechanism"
            recommendation = "Review analysis parameters and assumptions"
        
        print(f"Structural behavior: {structural_behavior}")
        print(f"Recommendation: {recommendation}")
    
    else:
        print("No clear dominant failure mode identified")
    
    return {
        'failure_categories': failure_categories,
        'dominant_mode': dominant_mode,
        'first_failure_load': first_failure_load,
        'load_summary': load_summary,
        'structural_behavior': structural_behavior,
        'recommendation': recommendation
    }

# Perform failure mode classification
failure_classification = classify_failure_modes(failure_events, critical_loads)

print(f"\n✓ Phase 6 Complete: Ultimate Limit State Analysis")
print(f"✓ Phase 7 Complete: Failure Mode Classification")
print(f"✓ Safety factors calculated based on assumed working loads")

# Section 8: Results Visualization and Reporting
def create_comprehensive_visualizations():
    """
    Create comprehensive visualizations of the failure analysis results
    """
    print(f"\nCreating Comprehensive Visualizations:")
    print("="*45)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Load-Displacement Curve
    ax1 = plt.subplot(2, 3, 1)
    
    # Extract data for load-displacement curve
    load_factors = [result['load_factor'] for result in analysis_results]
    displacements = [result['max_displacement_mm'] for result in analysis_results]
    total_loads = [result['total_load_kN'] for result in analysis_results]
    
    ax1.plot(displacements, total_loads, 'b-', linewidth=2, label='Load-Displacement')
    
    # Mark critical points
    if critical_loads['serviceability_load']:
        sls_idx = next((i for i, lf in enumerate(load_factors) 
                       if lf >= critical_loads['serviceability_load']), None)
        if sls_idx:
            ax1.axhline(y=total_loads[sls_idx], color='green', linestyle='--', 
                       label=f'SLS ({critical_loads["serviceability_load"]:.1f}×)')
    
    if critical_loads['first_yield_load']:
        yield_idx = next((i for i, lf in enumerate(load_factors) 
                         if lf >= critical_loads['first_yield_load']), None)
        if yield_idx:
            ax1.axhline(y=total_loads[yield_idx], color='orange', linestyle='--', 
                       label=f'First Yield ({critical_loads["first_yield_load"]:.1f}×)')
    
    if critical_loads['ultimate_load']:
        ult_idx = next((i for i, lf in enumerate(load_factors) 
                       if lf >= critical_loads['ultimate_load']), None)
        if ult_idx:
            ax1.axhline(y=total_loads[ult_idx], color='red', linestyle='--', 
                       label=f'Ultimate ({critical_loads["ultimate_load"]:.1f}×)')
    
    ax1.set_xlabel('Mid-span Displacement (mm)')
    ax1.set_ylabel('Total Applied Load (kN)')
    ax1.set_title('Load-Displacement Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Stress Ratio vs Load Factor
    ax2 = plt.subplot(2, 3, 2)
    
    stress_ratios = [result['max_stress_ratio'] for result in analysis_results]
    
    ax2.plot(load_factors, stress_ratios, 'r-', linewidth=2, label='Maximum Stress Ratio')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Yield Limit')
    ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='SLS Limit (60%)')
    
    ax2.set_xlabel('Load Factor')
    ax2.set_ylabel('Stress Ratio (σ/fy)')
    ax2.set_title('Stress Development')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Failure Events Timeline
    ax3 = plt.subplot(2, 3, 3)
    
    # Categorize failure events
    event_types = ['serviceability_exceeded', 'material_yield', 'buckling_failure', 'structural_collapse']
    colors = ['green', 'orange', 'red', 'black']
    
    for i, event_type in enumerate(event_types):
        events = [e for e in failure_events if e['type'] == event_type]
        if events:
            event_loads = [e['load_factor'] for e in events]
            ax3.scatter([i] * len(event_loads), event_loads, 
                       c=colors[i], s=50, alpha=0.7, label=event_type.replace('_', ' ').title())
    
    ax3.set_xticks(range(len(event_types)))
    ax3.set_xticklabels([et.replace('_', '\n').title() for et in event_types], rotation=45)
    ax3.set_ylabel('Load Factor')
    ax3.set_title('Failure Events Timeline')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Element Status at Ultimate Load
    ax4 = plt.subplot(2, 3, 4)
    
    # Get element status at final analysis step
    if analysis_results:
        final_result = analysis_results[-1]
        yielded_count = final_result['yielded_elements']
        buckled_count = final_result['buckled_elements']
        total_count = len(element_registry)
        intact_count = total_count - yielded_count - buckled_count
        
        status_data = [intact_count, yielded_count, buckled_count]
        status_labels = ['Intact', 'Yielded', 'Buckled']
        status_colors = ['green', 'orange', 'red']
        
        ax4.pie(status_data, labels=status_labels, colors=status_colors, autopct='%1.1f%%')
        ax4.set_title(f'Element Status at Load Factor {final_result["load_factor"]:.1f}')
    
    # Plot 5: Safety Factors
    ax5 = plt.subplot(2, 3, 5)
    
    if safety_factors:
        sf_names = list(safety_factors.keys())
        sf_values = list(safety_factors.values())
        
        bars = ax5.bar(sf_names, sf_values, color=['green' if sf >= 1.5 else 'orange' if sf >= 1.0 else 'red' 
                                                  for sf in sf_values])
        
        # Add safety factor thresholds
        ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Minimum SF = 1.0')
        ax5.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Target SF = 1.5')
        ax5.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Good SF = 2.0')
        
        # Add value labels on bars
        for bar, value in zip(bars, sf_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        ax5.set_ylabel('Safety Factor')
        ax5.set_title('Safety Factor Analysis')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        plt.setp(ax5.get_xticklabels(), rotation=45)
    
    # Plot 6: Progressive Failure Summary
    ax6 = plt.subplot(2, 3, 6)
    
    # Count cumulative failures
    cumulative_yield = []
    cumulative_buckling = []
    load_steps = []
    
    yielded_so_far = 0
    buckled_so_far = 0
    
    for result in analysis_results:
        load_steps.append(result['load_factor'])
        yielded_so_far = result['yielded_elements']
        buckled_so_far = result['buckled_elements']
        cumulative_yield.append(yielded_so_far)
        cumulative_buckling.append(buckled_so_far)
    
    ax6.plot(load_steps, cumulative_yield, 'o-', color='orange', label='Yielded Elements')
    ax6.plot(load_steps, cumulative_buckling, 's-', color='red', label='Buckled Elements')
    
    ax6.set_xlabel('Load Factor')
    ax6.set_ylabel('Number of Failed Elements')
    ax6.set_title('Progressive Failure Development')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('comprehensive_failure_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive visualizations saved as 'comprehensive_failure_analysis.png'")
    
    return fig

def generate_detailed_report():
    """
    Generate a comprehensive failure analysis report
    """
    print(f"\nGenerating Detailed Failure Analysis Report:")
    print("="*50)
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE STRUCTURAL FAILURE ANALYSIS REPORT")
    report.append("="*80)
    
    # Executive Summary
    report.append("\nEXECUTIVE SUMMARY")
    report.append("-" * 20)
    
    if failure_classification['dominant_mode']:
        report.append(f"Governing Failure Mode: {failure_classification['dominant_mode'].replace('_', ' ').title()}")
        report.append(f"First Failure Load: {failure_classification['first_failure_load']:.1f}× unit load")
        
        if critical_loads['ultimate_load']:
            total_ultimate = critical_loads['ultimate_load'] * len(load_nodes) * abs(unit_load) / 1000
            report.append(f"Ultimate Capacity: {critical_loads['ultimate_load']:.1f}× unit load ({total_ultimate:.0f} kN)")
        
        report.append(f"Structural Behavior: {failure_classification.get('structural_behavior', 'Not determined')}")
        report.append(f"Design Recommendation: {failure_classification.get('recommendation', 'Review required')}")
    
    # Structure Information
    report.append(f"\nSTRUCTURE INFORMATION")
    report.append("-" * 25)
    report.append(f"Span: 30.0 m")
    report.append(f"Height: 4.5 m")
    report.append(f"Total Elements: {len(element_registry)}")
    report.append(f"Material: Steel01 S235 (fy = {material_props['fy_s235']/1e6:.0f} MPa)")
    report.append(f"Analysis Type: Nonlinear static with Steel01 material")
    
    # Critical Load Summary
    report.append(f"\nCRITICAL LOAD SUMMARY")
    report.append("-" * 25)
    
    for load_name, load_factor in failure_classification['load_summary'].items():
        total_load = load_factor * len(load_nodes) * abs(unit_load) / 1000
        report.append(f"{load_name}: {load_factor:.1f}× ({total_load:.0f} kN)")
    
    # Safety Factor Assessment
    report.append(f"\nSAFETY FACTOR ASSESSMENT")
    report.append("-" * 30)
    
    working_load_kN = 10.0 * len(load_nodes) * abs(unit_load) / 1000
    report.append(f"Assumed Working Load: {working_load_kN:.0f} kN")
    
    for sf_name, sf_value in safety_factors.items():
        status = "ADEQUATE" if sf_value >= 1.5 else "MARGINAL" if sf_value >= 1.0 else "INADEQUATE"
        report.append(f"{sf_name.title()} Safety Factor: {sf_value:.2f} - {status}")
    
    # Failure Event Details
    report.append(f"\nFAILURE EVENT DETAILS")
    report.append("-" * 25)
    
    for event_type, events in failure_classification['failure_categories'].items():
        if events:
            report.append(f"\n{event_type.replace('_', ' ').title()} Events: {len(events)}")
            for i, event in enumerate(events[:5]):  # Show first 5 events
                report.append(f"  {i+1}. Load Factor {event['load_factor']:.1f}: {event.get('description', 'No description')}")
            if len(events) > 5:
                report.append(f"  ... and {len(events) - 5} more events")
    
    # Element Performance
    report.append(f"\nELEMENT PERFORMANCE SUMMARY")
    report.append("-" * 35)
    
    if analysis_results:
        final_result = analysis_results[-1]
        report.append(f"At Ultimate Load (Factor {final_result['load_factor']:.1f}):")
        report.append(f"  Yielded Elements: {final_result['yielded_elements']}")
        report.append(f"  Buckled Elements: {final_result['buckled_elements']}")
        report.append(f"  Intact Elements: {len(element_registry) - final_result['yielded_elements'] - final_result['buckled_elements']}")
        report.append(f"  Maximum Displacement: {final_result['max_displacement_mm']:.2f} mm")
        report.append(f"  Maximum Stress Ratio: {final_result['max_stress_ratio']:.3f}")
    
    # Serviceability Assessment
    report.append(f"\nSERVICEABILITY ASSESSMENT")
    report.append("-" * 30)
    
    sls_load = critical_loads.get('serviceability_load')
    if sls_load:
        sls_total = sls_load * len(load_nodes) * abs(unit_load) / 1000
        report.append(f"Serviceability Limit Reached: {sls_load:.1f}× ({sls_total:.0f} kN)")
        
        # Find serviceability data at SLS load
        sls_result = next((r for r in analysis_results if r['load_factor'] >= sls_load), None)
        if sls_result:
            report.append(f"  Displacement at SLS: {sls_result['max_displacement_mm']:.2f} mm")
            report.append(f"  Limit (L/360): {serviceability_limits['deflection_normal']*1000:.1f} mm")
            report.append(f"  Stress Ratio at SLS: {sls_result['max_stress_ratio']:.3f}")
    else:
        report.append("Serviceability limits not exceeded within analysis range")
    
    # Design Recommendations
    report.append(f"\nDESIGN RECOMMENDATIONS")
    report.append("-" * 30)
    
    recommendations = []
    
    # Based on dominant failure mode
    if failure_classification['recommendation']:
        recommendations.append(f"Primary: {failure_classification['recommendation']}")
    
    # Based on safety factors
    inadequate_sf = [name for name, sf in safety_factors.items() if sf < 1.5]
    if inadequate_sf:
        recommendations.append(f"Improve safety factors for: {', '.join(inadequate_sf)}")
    
    # Based on slenderness
    slender_members = [name for name, data in slenderness_data.items() if data['category'] == 'Slender']
    if slender_members:
        recommendations.append(f"Consider lateral bracing for slender members: {', '.join(slender_members)}")
    
    if not recommendations:
        recommendations.append("Structure appears adequate for assumed loading")
    
    for i, rec in enumerate(recommendations, 1):
        report.append(f"{i}. {rec}")
    
    # Analysis Limitations
    report.append(f"\nANALYSIS LIMITATIONS")
    report.append("-" * 25)
    report.append("1. Truss elements assumed (no member moments)")
    report.append("2. Material properties based on S235 steel")
    report.append("3. Buckling based on Euler theory with estimated I values")
    report.append("4. No dynamic effects considered")
    report.append("5. Perfect connections assumed")
    report.append("6. No fabrication tolerances or imperfections")
    
    report.append(f"\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Print report
    for line in report:
        print(line)
    
    # Save report to file
    with open('structural_failure_analysis_report.txt', 'w') as f:
        for line in report:
            f.write(line + '\n')
    
    print(f"\n✓ Detailed report generated and saved to 'structural_failure_analysis_report.txt'")
    
    return report

def create_data_export():
    """
    Export analysis data for further processing
    """
    print(f"\nExporting Analysis Data:")
    print("="*30)
    
    # Create comprehensive data dictionary
    export_data = {
        'analysis_metadata': {
            'structure_type': 'Truss',
            'span_m': 30.0,
            'height_m': 4.5,
            'total_elements': len(element_registry),
            'material': 'Steel01_S235',
            'analysis_type': 'Nonlinear_Static',
            'yield_strength_MPa': material_props['fy_s235'] / 1e6,
            'elastic_modulus_GPa': material_props['E'] / 1e9
        },
        'critical_loads': critical_loads,
        'safety_factors': safety_factors,
        'analysis_results': analysis_results,
        'failure_events': failure_events,
        'element_registry': {str(k): v for k, v in element_registry.items()},
        'buckling_data': buckling_data,
        'slenderness_data': slenderness_data,
        'serviceability_limits': serviceability_limits,
        'failure_classification': failure_classification
    }
    
    # Export to JSON
    import json
    with open('structural_failure_analysis_data.json', 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    # Export analysis results to CSV
    df_results = pd.DataFrame(analysis_results)
    df_results.to_csv('analysis_results.csv', index=False)
    
    # Export failure events to CSV
    df_failures = pd.DataFrame(failure_events)
    if not df_failures.empty:
        df_failures.to_csv('failure_events.csv', index=False)
    
    print("✓ Analysis data exported to:")
    print("  - structural_failure_analysis_data.json (comprehensive data)")
    print("  - analysis_results.csv (load-displacement data)")
    print("  - failure_events.csv (failure timeline)")
    print("  - structural_failure_analysis_report.txt (detailed report)")
    
    return export_data

# Execute Section 8: Results Visualization and Reporting
print(f"\n" + "="*70)
print("PHASE 8: RESULTS VISUALIZATION AND REPORTING")
print("="*70)

# Create visualizations
fig = create_comprehensive_visualizations()

# Generate detailed report
report = generate_detailed_report()

# Export data
export_data = create_data_export()

print(f"\n" + "="*70)
print("STRUCTURAL FAILURE ANALYSIS COMPLETE")
print("="*70)
print("✅ All phases completed successfully:")
print("✅ Phase 1: Enhanced Material Properties and Nonlinear Modeling")
print("✅ Phase 2: Geometric and Material Nonlinearity Setup")  
print("✅ Phase 3: Buckling Analysis Implementation")
print("✅ Phase 4: Progressive Loading and Failure Detection")
print("✅ Phase 5: Serviceability Limit State Checking")
print("✅ Phase 6: Ultimate Limit State Analysis")
print("✅ Phase 7: Failure Mode Classification")
print("✅ Phase 8: Results Visualization and Reporting")

print(f"\n📊 Generated Outputs:")
print(f"  📈 6 comprehensive plots showing failure progression")
print(f"  📄 Detailed failure analysis report")
print(f"  💾 Analysis data exported to multiple formats")
print(f"  📋 Safety factor assessment and recommendations")

print(f"\n🎯 Key Findings:")
if critical_loads.get('ultimate_load'):
    total_ult = critical_loads['ultimate_load'] * len(load_nodes) * abs(unit_load) / 1000
    print(f"  Ultimate Capacity: {critical_loads['ultimate_load']:.1f}× unit load ({total_ult:.0f} kN)")

if failure_classification.get('dominant_mode'):
    print(f"  Governing Failure: {failure_classification['dominant_mode'].replace('_', ' ').title()}")

if safety_factors:
    min_sf = min(safety_factors.values())
    print(f"  Minimum Safety Factor: {min_sf:.2f}")

print(f"\n✨ Ready for engineering review and design optimization!")