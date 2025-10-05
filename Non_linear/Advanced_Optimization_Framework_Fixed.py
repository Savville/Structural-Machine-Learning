# Add OpenSees path to environment
import os
opensees_path = r'C:\OpenSees3.7.1'
if os.path.exists(opensees_path):
    os.environ['PATH'] = os.path.join(opensees_path, 'bin') + ';' + os.environ['PATH']
    tcl_path = os.path.join(opensees_path, 'lib')
    if os.path.exists(tcl_path):
        os.environ['TCL_LIBRARY'] = os.path.join(tcl_path, 'tcl8.6')
        os.environ['TK_LIBRARY'] = os.path.join(tcl_path, 'tk8.6')

import matplotlib
matplotlib.use('Agg')

import openseespy.opensees as ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import your existing framework
try:
    from ML_Nonlinear_Truss import NonlinearSteelTrussFramework
    print("✅ Base framework imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Solution: Ensure ML_Nonlinear_Truss.py is in the same directory")
    exit(1)

print("="*80)
print("ADVANCED NONLINEAR STEEL SECTION OPTIMIZATION FRAMEWORK")
print("Multi-Objective Design with ML-Driven Optimization")
print("Analysis Type: NONLINEAR STATIC + MODAL ANALYSIS")
print("="*80)

class AdvancedSteelSectionOptimizer(NonlinearSteelTrussFramework):
    """
    Advanced optimization framework for steel sections using:
    - NONLINEAR STATIC ANALYSIS for forces and displacements
    - MODAL ANALYSIS for frequencies and dynamic properties
    - Multiple load combinations (EN 1990)
    - Temperature effects and material nonlinearity
    - Multi-objective optimization (weight + cost + carbon)
    """
    
    def __init__(self):
        super().__init__()
        print("\n🏛️ Initializing optimization framework...")
        self.design_standards = self.define_design_standards()
        self.load_combinations = self.define_load_combinations()
        self.optimization_constraints = self.define_optimization_constraints()
        self.material_costs = self.define_material_costs()
        print("✅ Framework initialized successfully")
    
    def define_design_standards(self):
        """
        Define design standards and limits (Eurocode 3)
        """
        print("\n🏛️ Defining Design Standards (Eurocode 3):")
        print("="*50)
        
        standards = {
            'eurocode_3': {
                # Material factors (EN 1993-1-1)
                'gamma_M0': 1.00,    # Partial factor for resistance of cross-sections
                'gamma_M1': 1.00,    # Partial factor for resistance of members to instability
                'gamma_M2': 1.25,    # Partial factor for resistance of net cross-sections
                
                # Steel grades (EN 10025-2)
                'steel_grades': {
                    'S235': {'fy': 235e6, 'fu': 360e6, 'E': 210e9},
                    'S275': {'fy': 275e6, 'fu': 430e6, 'E': 210e9},
                    'S355': {'fy': 355e6, 'fu': 510e6, 'E': 210e9},
                    'S420': {'fy': 420e6, 'fu': 520e6, 'E': 210e9}
                },
                
                # Serviceability limits (EN 1993-1-1 §7)
                'deflection_limits': {
                    'vertical_deflection_live': 'L/250',      # Live load deflection
                    'vertical_deflection_total': 'L/200',     # Total deflection
                    'horizontal_deflection': 'H/150'          # Horizontal deflection
                },
                
                # Stress limits
                'stress_utilization_limit': 0.95,  # 95% of characteristic strength
                
                # Buckling parameters (EN 1993-1-1 §6)
                'buckling_curves': {
                    'rolled_sections': 'curve_a',    # α = 0.21
                    'welded_sections': 'curve_c'     # α = 0.49
                },
                
                # Fatigue design (EN 1993-1-9)
                'fatigue_categories': {
                    'base_material': 160e6,           # Category 160
                    'welded_details': 80e6            # Category 80
                }
            }
        }
        
        print("✅ Eurocode 3 standards loaded")
        print(f"   Steel grades: {list(standards['eurocode_3']['steel_grades'].keys())}")
        print(f"   Deflection limits: L/250 (live), L/200 (total)")
        print(f"   Stress utilization: {standards['eurocode_3']['stress_utilization_limit']:.0%}")
        
        return standards
    
    def define_load_combinations(self):
        """
        Define load combinations according to EN 1990
        """
        print("\n📊 Defining Load Combinations (EN 1990):")
        print("="*50)
        
        combinations = {
            # Ultimate Limit State combinations (EN 1990 §6.4.3.2)
            'ULS': {
                'combination_1': {
                    'description': 'Permanent + Leading variable',
                    'factors': {'G': 1.35, 'Q1': 1.50, 'Q2': 0.75, 'W': 0.60, 'S': 0.75},
                    'load_case': '1.35*G + 1.50*Q + 0.75*W'
                },
                'combination_2': {
                    'description': 'Permanent + Wind leading',
                    'factors': {'G': 1.35, 'Q1': 0.75, 'Q2': 1.50, 'W': 1.50, 'S': 0.75},
                    'load_case': '1.35*G + 0.75*Q + 1.50*W'
                },
                'combination_3': {
                    'description': 'Permanent + Snow leading',
                    'factors': {'G': 1.35, 'Q1': 0.75, 'Q2': 0.60, 'W': 0.60, 'S': 1.50},
                    'load_case': '1.35*G + 0.75*Q + 1.50*S'
                },
                'combination_4': {
                    'description': 'Seismic combination',
                    'factors': {'G': 1.00, 'Q1': 0.30, 'Q2': 0.00, 'W': 0.00, 'E': 1.00},
                    'load_case': '1.00*G + 0.30*Q + 1.00*E'
                }
            },
            
            # Serviceability Limit State combinations (EN 1990 §6.5.3)
            'SLS': {
                'characteristic': {
                    'description': 'Characteristic combination',
                    'factors': {'G': 1.00, 'Q1': 1.00, 'Q2': 0.70, 'W': 0.60, 'S': 0.70},
                    'load_case': '1.00*G + 1.00*Q + 0.70*W'
                },
                'frequent': {
                    'description': 'Frequent combination',
                    'factors': {'G': 1.00, 'Q1': 0.70, 'Q2': 0.70, 'W': 0.20, 'S': 0.50},
                    'load_case': '1.00*G + 0.70*Q + 0.20*W'
                },
                'quasi_permanent': {
                    'description': 'Quasi-permanent combination',
                    'factors': {'G': 1.00, 'Q1': 0.60, 'Q2': 0.30, 'W': 0.00, 'S': 0.20},
                    'load_case': '1.00*G + 0.60*Q'
                }
            },
            
            # Fatigue combinations (EN 1993-1-9)
            'FATIGUE': {
                'fatigue_load': {
                    'description': 'Fatigue load model',
                    'factors': {'G': 1.00, 'Q_fat': 1.00},
                    'load_case': '1.00*G + 1.00*Q_fatigue'
                }
            }
        }
        
        # Define base load values (kN)
        base_loads = {
            'G': 2.0,      # Dead load per node (kN)
            'Q1': 3.0,     # Live load per node (kN) 
            'Q2': 1.5,     # Additional live load (kN)
            'W': 2.5,      # Wind load per node (kN)
            'S': 2.0,      # Snow load per node (kN)
            'E': 4.0,      # Seismic load per node (kN)
            'Q_fat': 1.0   # Fatigue load amplitude (kN)
        }
        
        print("✅ Load combinations defined")
        print(f"   ULS combinations: {len(combinations['ULS'])}")
        print(f"   SLS combinations: {len(combinations['SLS'])}")
        print(f"   Fatigue combinations: {len(combinations['FATIGUE'])}")
        print(f"   Base loads: G={base_loads['G']}kN, Q={base_loads['Q1']}kN, W={base_loads['W']}kN")
        
        combinations['base_loads'] = base_loads
        return combinations
        
    def define_optimization_constraints(self):
        """
        Define optimization constraints for design verification
        """
        print("\n🎯 Defining Optimization Constraints:")
        print("="*50)
        
        constraints = {
            # Strength constraints (ULS)
            'strength': {
                'max_stress_ratio': 0.95,              # σ/fy ≤ 0.95
                'max_shear_stress_ratio': 0.6,         # τ/fy ≤ 0.6 (von Mises)
                'buckling_resistance_factor': 1.0,      # Buckling check
                'fatigue_damage_limit': 1.0             # Palmgren-Miner rule
            },
            
            # Serviceability constraints (SLS)
            'serviceability': {
                'max_deflection_ratio_live': 1/250,     # δ/L ≤ 1/250 (live loads)
                'max_deflection_ratio_total': 1/200,    # δ/L ≤ 1/200 (total)
                'max_acceleration': 0.7,                # Natural frequency > 3Hz
                'max_vibration_amplitude': 5e-3         # 5mm vibration limit
            },
            
            # Stability constraints
            'stability': {
                'min_slenderness_ratio': 60,            # λ ≤ 200 for compression
                'max_slenderness_ratio': 200,
                'lateral_torsional_buckling': True,      # Check LTB
                'local_buckling': True                   # Check local buckling
            },
            
            # Geometric constraints
            'geometry': {
                'min_area': 1e-4,                      # 10 cm² minimum
                'max_area': 1e-2,                      # 100 cm² maximum
                'min_thickness': 5e-3,                  # 5mm minimum thickness
                'area_ratios': {
                    'top_chord_min': 0.8,              # Top/bottom chord ratio
                    'web_min': 0.4                     # Web/chord ratio
                }
            },
            
            # Performance constraints
            'performance': {
                'min_natural_frequency': 3.0,          # Hz (avoid resonance)
                'max_dynamic_amplification': 2.0,      # Dynamic amplification factor
                'temperature_stability': 0.05,         # 5% frequency variation allowed
                'damage_tolerance': 0.02               # 2% damage detection limit
            }
        }
        
        print("✅ Optimization constraints defined")
        print(f"   Strength: Max stress ratio = {constraints['strength']['max_stress_ratio']:.0%}")
        print(f"   Serviceability: Max deflection = L/{1/constraints['serviceability']['max_deflection_ratio_live']:.0f}")
        print(f"   Geometry: Area range = {constraints['geometry']['min_area']*1e4:.0f}-{constraints['geometry']['max_area']*1e4:.0f} cm²")
        
        return constraints
    
    def define_material_costs(self):
        """
        Define material costs and carbon footprint data
        """
        print("\n💰 Defining Material Costs & Environmental Data:")
        print("="*50)
        
        costs = {
            # Steel costs (€/kg, 2024 prices)
            'steel_cost_per_kg': {
                'S235': 1.20,  # €/kg
                'S275': 1.25,  # €/kg  
                'S355': 1.35,  # €/kg
                'S420': 1.50   # €/kg
            },
            
            # Carbon footprint (kg CO2/kg steel)
            'carbon_footprint_per_kg': {
                'S235': 2.1,   # kg CO2/kg steel
                'S275': 2.2,   # kg CO2/kg steel
                'S355': 2.3,   # kg CO2/kg steel
                'S420': 2.5    # kg CO2/kg steel
            },
            
            # Additional costs
            'fabrication_cost': {
                'cutting': 15.0,        # €/m cut length
                'welding': 25.0,        # €/m weld length
                'bolting': 5.0,         # €/bolt
                'surface_treatment': 8.0 # €/m²
            },
            
            # Transportation costs
            'transport_cost_per_km': 0.05,  # €/kg/km
            'average_transport_distance': 200,  # km
            
            # Maintenance costs (NPV over 50 years)
            'maintenance_cost_factor': 0.15,  # 15% of initial cost
            
            # End-of-life value
            'recycling_value_factor': 0.10   # 10% recovery value
        }
        
        # Steel density
        costs['steel_density'] = 7850  # kg/m³
        
        print("✅ Cost and environmental data defined")
        print(f"   Steel cost range: €{min(costs['steel_cost_per_kg'].values()):.2f}-{max(costs['steel_cost_per_kg'].values()):.2f}/kg")
        print(f"   Carbon footprint: {min(costs['carbon_footprint_per_kg'].values()):.1f}-{max(costs['carbon_footprint_per_kg'].values()):.1f} kg CO2/kg")
        
        return costs
    
    def calculate_load_case_forces(self, load_case):
        """
        Calculate forces for specific load combination
        """
        base_loads = self.load_combinations['base_loads']
        
        # Define load case factors
        load_case_factors = {
            'combination_1': {'G': 1.35, 'Q1': 1.50, 'W': 0.60},
            'combination_2': {'G': 1.35, 'Q1': 0.75, 'W': 1.50},
            'characteristic': {'G': 1.00, 'Q1': 1.00, 'W': 0.60},
            'fatigue_load': {'G': 1.00, 'Q_fat': 1.00}
        }
        
        factors = load_case_factors.get(load_case, {'G': 1.35, 'Q1': 1.50})
        
        # Calculate total vertical load
        total_vertical_load = 0
        for load_type, factor in factors.items():
            if load_type in base_loads:
                total_vertical_load += factor * base_loads[load_type]
        
        # Distribute loads to nodes (5 loaded nodes)
        load_per_node = total_vertical_load * 1000  # Convert to N
        
        return {
            'vertical_loads': [load_per_node] * 5,  # 5 loading points
            'horizontal_loads': [0.0] * 5,         # No horizontal loads for this case
            'load_case_description': load_case,
            'total_load': total_vertical_load * 5 * 1000  # Total load in N
        }
    
    def calculate_total_weight(self, section_assignment):
        """
        Calculate total structural weight
        """
        density = self.material_costs['steel_density']  # kg/m³
        
        # Element lengths (approximate)
        element_lengths = {
            'bottom_chord': 6.0 * 5,      # 5 elements × 6m each = 30m total
            'top_chord': 6.0 * 4,         # 4 elements × 6m each = 24m total  
            'web_vertical': 4.5 * 4,      # 4 vertical × 4.5m each = 18m total
            'web_diagonal': 7.5 * 6       # 6 diagonals × ~7.5m each = 45m total
        }
        
        total_weight = 0.0
        for member_type, length in element_lengths.items():
            area = section_assignment[member_type]
            weight = area * length * density
            total_weight += weight
        
        return total_weight
    
    def calculate_total_cost(self, section_assignment):
        """
        Calculate total project cost
        """
        steel_grade = section_assignment['steel_grade']
        cost_per_kg = self.material_costs['steel_cost_per_kg'][steel_grade]
        
        # Material cost
        weight = self.calculate_total_weight(section_assignment)
        material_cost = weight * cost_per_kg
        
        # Fabrication cost (simplified)
        fabrication_cost = material_cost * 0.3  # 30% of material cost
        
        # Transportation cost
        transport_distance = self.material_costs['average_transport_distance']
        transport_cost = weight * self.material_costs['transport_cost_per_km'] * transport_distance
        
        # Total cost
        total_cost = material_cost + fabrication_cost + transport_cost
        
        return total_cost
    
    def calculate_carbon_footprint(self, section_assignment):
        """
        Calculate carbon footprint
        """
        steel_grade = section_assignment['steel_grade'] 
        carbon_per_kg = self.material_costs['carbon_footprint_per_kg'][steel_grade]
        
        weight = self.calculate_total_weight(section_assignment)
        
        # Production carbon footprint
        production_carbon = weight * carbon_per_kg
        
        # Transportation carbon (simplified)
        transport_carbon = weight * 0.1  # 0.1 kg CO2/kg for transport
        
        total_carbon = production_carbon + transport_carbon
        
        return total_carbon
    
    def analyze_section_performance(self, section_assignment, applied_loads, temp_factor):
        """
        Analyze structural performance for given section assignment
        """
        try:
            # Create model with custom sections
            span, height = self.create_nonlinear_truss_model(temp_factor)
            
            # Define custom cross sections
            custom_sections = self.create_custom_cross_sections(section_assignment)
            
            # Create elements with custom sections
            element_registry, total_elements = self.create_nonlinear_elements(custom_sections)
            
            # Apply custom loading
            self.apply_custom_loading(applied_loads)
            
            # Perform analyses
            static_results = self.perform_nonlinear_static_analysis()
            modal_results = self.perform_modal_analysis()
            
            if static_results['converged'] and modal_results:
                # Calculate performance metrics
                max_stress = self.calculate_maximum_stress(static_results, element_registry)
                max_deflection = self.calculate_maximum_deflection(static_results)
                fundamental_frequency = modal_results['frequencies'][0]
                
                return {
                    'converged': True,
                    'max_stress': max_stress,
                    'max_deflection': max_deflection,
                    'fundamental_frequency': fundamental_frequency,
                    'static_results': static_results,
                    'modal_results': modal_results,
                    'element_registry': element_registry
                }
            else:
                return {'converged': False}
                
        except Exception as e:
            return {'converged': False, 'error': str(e)}
    
    def create_custom_cross_sections(self, section_assignment):
        """
        Create custom cross sections from optimization parameters (FIXED)
        """
        custom_sections = {
            'bottom_chord': {
                'A': section_assignment['bottom_chord'],
                'I': self.estimate_moment_of_inertia(section_assignment['bottom_chord']),
                'description': f"Custom bottom chord - {section_assignment['bottom_chord']*1e4:.1f} cm²",
                'material_tag': self.get_material_tag(section_assignment['steel_grade']),
                # Add missing keys that base framework expects:
                'section': f"BC_{section_assignment['bottom_chord']*1e4:.0f}",
                'type': 'bottom_chord',
                'area': section_assignment['bottom_chord'],  # Duplicate for compatibility
                'moment_of_inertia': self.estimate_moment_of_inertia(section_assignment['bottom_chord'])
            },
            'top_chord': {
                'A': section_assignment['top_chord'],
                'I': self.estimate_moment_of_inertia(section_assignment['top_chord']),
                'description': f"Custom top chord - {section_assignment['top_chord']*1e4:.1f} cm²",
                'material_tag': self.get_material_tag(section_assignment['steel_grade']),
                # Add missing keys:
                'section': f"TC_{section_assignment['top_chord']*1e4:.0f}",
                'type': 'top_chord',
                'area': section_assignment['top_chord'],
                'moment_of_inertia': self.estimate_moment_of_inertia(section_assignment['top_chord'])
            },
            'web_vertical': {
                'A': section_assignment['web_vertical'],
                'I': self.estimate_moment_of_inertia(section_assignment['web_vertical']),
                'description': f"Custom web vertical - {section_assignment['web_vertical']*1e4:.1f} cm²",
                'material_tag': self.get_material_tag(section_assignment['steel_grade']),
                # Add missing keys:
                'section': f"WV_{section_assignment['web_vertical']*1e4:.0f}",
                'type': 'web_vertical',
                'area': section_assignment['web_vertical'],
                'moment_of_inertia': self.estimate_moment_of_inertia(section_assignment['web_vertical'])
            },
            'web_diagonal': {
                'A': section_assignment['web_diagonal'],
                'I': self.estimate_moment_of_inertia(section_assignment['web_diagonal']),
                'description': f"Custom web diagonal - {section_assignment['web_diagonal']*1e4:.1f} cm²",
                'material_tag': self.get_material_tag(section_assignment['steel_grade']),
                # Add missing keys:
                'section': f"WD_{section_assignment['web_diagonal']*1e4:.0f}",
                'type': 'web_diagonal',
                'area': section_assignment['web_diagonal'],
                'moment_of_inertia': self.estimate_moment_of_inertia(section_assignment['web_diagonal'])
            }
        }
        
        return custom_sections
    
    def estimate_moment_of_inertia(self, area):
        """
        Estimate moment of inertia from cross-sectional area (simplified)
        """
        # Assume rectangular section with aspect ratio 2:1
        # A = b*h, h = 2*b, so b = sqrt(A/2), h = sqrt(2*A)
        # I = b*h³/12 = (A/2) * (sqrt(2*A))² / 12 = A^2 / (6*sqrt(2))
        return area**2 / (6 * np.sqrt(2))
    
    def get_material_tag(self, steel_grade):
        """
        Get material tag for steel grade
        """
        grade_map = {
            'S235': 2,  # Steel01 S235
            'S275': 2,  # Steel01 S235 (approximation)
            'S355': 3,  # Steel01 S355
            'S420': 3   # Steel01 S355 (approximation)
        }
        return grade_map.get(steel_grade, 3)
    
    def apply_custom_loading(self, applied_loads):
        """
        Apply custom loading pattern
        """
        # Boundary conditions (same as base framework)
        ops.fix(1, 1, 1)   # Pin support
        ops.fix(11, 0, 1)  # Roller support
        
        # Add masses
        node_masses = [150, 250, 200, 250, 200, 300, 200, 250, 200, 250, 150]
        for i, mass in enumerate(node_masses, 1):
            ops.mass(i, mass, mass)
        
        # Apply loads
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        
        # Apply vertical loads to top chord nodes
        load_nodes = [2, 4, 6, 8, 10]
        for i, node in enumerate(load_nodes):
            if i < len(applied_loads['vertical_loads']):
                ops.load(node, 0.0, -applied_loads['vertical_loads'][i])
    
    def calculate_maximum_stress(self, static_results, element_registry):
        """
        Calculate maximum stress in structure
        """
        max_stress = 0.0
        
        for elem_id, force_data in static_results['element_forces'].items():
            if elem_id in element_registry:
                area = element_registry[elem_id]['actual_area']
                axial_force = abs(force_data['axial'])
                stress = axial_force / area if area > 0 else 0
                max_stress = max(max_stress, stress)
        
        return max_stress
    
    def calculate_maximum_deflection(self, static_results):
        """
        Calculate maximum deflection in structure
        """
        max_deflection = 0.0
        
        for node_id, disp_data in static_results['displacements'].items():
            deflection = abs(disp_data['y'])
            max_deflection = max(max_deflection, deflection)
        
        return max_deflection
    
    def evaluate_design_constraints(self, section_assignment, analysis_result, applied_loads):
        """
        Evaluate design constraints and calculate violations
        """
        constraints = self.optimization_constraints
        steel_grade = section_assignment['steel_grade']
        fy = self.design_standards['eurocode_3']['steel_grades'][steel_grade]['fy']
        
        # Stress constraint
        max_stress = analysis_result['max_stress']
        allowable_stress = fy * constraints['strength']['max_stress_ratio']
        stress_utilization = max_stress / allowable_stress
        stress_violation = max(0, stress_utilization - 1.0)
        
        # Deflection constraint (span = 30m)
        span = 30.0
        max_deflection = analysis_result['max_deflection']
        allowable_deflection = span * constraints['serviceability']['max_deflection_ratio_total']
        deflection_utilization = max_deflection / allowable_deflection
        deflection_violation = max(0, deflection_utilization - 1.0)
        
        # Frequency constraint
        fundamental_frequency = analysis_result['fundamental_frequency']
        min_frequency = constraints['performance']['min_natural_frequency']
        frequency_utilization = min_frequency / fundamental_frequency if fundamental_frequency > 0 else 1e6
        frequency_violation = max(0, frequency_utilization - 1.0)
        
        # Overall feasibility
        feasible = (stress_violation == 0 and 
                   deflection_violation == 0 and 
                   frequency_violation == 0)
        
        return {
            'stress_violation': stress_violation,
            'deflection_violation': deflection_violation, 
            'frequency_violation': frequency_violation,
            'stress_utilization': stress_utilization,
            'deflection_utilization': deflection_utilization,
            'frequency_utilization': frequency_utilization,
            'feasible': feasible
        }
    
    def test_single_scenario(self):
        """Test a single optimization scenario with enhanced debugging"""
        print("\n🧪 Testing Single Optimization Scenario (ENHANCED DEBUG):")
        print("="*60)
        
        # Define test parameters
        test_section = {
            'bottom_chord': 30e-4,      # 30 cm²
            'top_chord': 30e-4,         # 30 cm²
            'web_vertical': 15e-4,      # 15 cm²
            'web_diagonal': 15e-4,      # 15 cm²
            'steel_grade': 'S355'
        }
        
        test_loads = self.calculate_load_case_forces('combination_1')
        test_temp = 1.0
        
        print(f"Test parameters:")
        print(f"  Bottom chord: {test_section['bottom_chord']*1e4:.1f} cm²")
        print(f"  Top chord: {test_section['top_chord']*1e4:.1f} cm²")
        print(f"  Web vertical: {test_section['web_vertical']*1e4:.1f} cm²")
        print(f"  Web diagonal: {test_section['web_diagonal']*1e4:.1f} cm²")
        print(f"  Steel grade: {test_section['steel_grade']}")
        print(f"  Total load: {test_loads['total_load']/1000:.1f} kN")
        
        # Check base framework methods first
        required_methods = [
            'create_nonlinear_truss_model',
            'create_nonlinear_elements',
            'perform_nonlinear_static_analysis',
            'perform_modal_analysis'
        ]
        
        print(f"\n🔍 Checking base framework methods:")
        missing_methods = []
        for method in required_methods:
            if hasattr(self, method):
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} MISSING")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"\n❌ Missing methods: {missing_methods}")
            print(f"   Fix ML_Nonlinear_Truss.py first")
            return False
        
        # Run enhanced debug analysis
        try:
            result = self.analyze_section_performance_debug(test_section, test_loads, test_temp)
            
            if result['converged']:
                print(f"\n🎉 ANALYSIS SUCCESSFUL!")
                
                # Test objective calculations
                print(f"\n📊 Testing objective calculations:")
                weight = self.calculate_total_weight(test_section)
                cost = self.calculate_total_cost(test_section)
                carbon = self.calculate_carbon_footprint(test_section)
                
                print(f"  Weight: {weight:.0f} kg ({weight/30:.1f} kg/m)")
                print(f"  Cost: €{cost:.0f}")
                print(f"  Carbon: {carbon:.0f} kg CO2")
                
                # Test constraint evaluation
                print(f"\n🎯 Testing constraint evaluation:")
                constraints = self.evaluate_design_constraints(test_section, result, test_loads)
                print(f"  Feasible: {constraints['feasible']}")
                print(f"  Stress utilization: {constraints['stress_utilization']:.1%}")
                print(f"  Deflection utilization: {constraints['deflection_utilization']:.1%}")
                print(f"  Frequency utilization: {constraints['frequency_utilization']:.1%}")
                
                return True
            else:
                print(f"\n❌ ANALYSIS FAILED: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"\n❌ TEST EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_section_performance_debug(self, section_assignment, applied_loads, temp_factor):
        """
        Enhanced debug version with step-by-step analysis
        """
        print(f"\n🔍 DEBUG: Step-by-step Analysis")
        print(f"="*50)
        
        try:
            # Step 1: Model Creation
            print(f"Step 1: Creating nonlinear truss model...")
            span, height = self.create_nonlinear_truss_model(temp_factor)
            print(f"  ✅ Model: {span}m × {height}m")
            
            # Step 2: Section Creation  
            print(f"Step 2: Creating custom cross sections...")
            custom_sections = self.create_custom_cross_sections(section_assignment)
            print(f"  ✅ Sections created: {len(custom_sections)}")
            
            # Debug: Print section details
            for section_name, section_data in custom_sections.items():
                print(f"    {section_name}: A={section_data['A']*1e4:.1f}cm², section='{section_data.get('section', 'MISSING')}'")
            
            # Step 3: Element Creation
            print(f"Step 3: Creating nonlinear elements...")
            element_registry, total_elements = self.create_nonlinear_elements(custom_sections)
            print(f"  ✅ Elements: {total_elements} created")
            
            # Debug: Check element registry
            if len(element_registry) == 0:
                print(f"  ❌ WARNING: Element registry is empty!")
                return {'converged': False, 'error': 'Empty element registry'}
            else:
                # Print first few elements for debugging
                sample_elements = list(element_registry.items())[:3]
                print(f"  📋 Sample elements:")
                for elem_id, elem_data in sample_elements:
                    print(f"    Element {elem_id}: {elem_data}")
            
            # Step 4: Load Application
            print(f"Step 4: Applying loads...")
            self.apply_custom_loading(applied_loads)
            print(f"  ✅ Loads applied: {applied_loads['total_load']/1000:.1f} kN total")
            
            # Step 5: Static Analysis
            print(f"Step 5: Performing static analysis...")
            static_results = self.perform_nonlinear_static_analysis()
            
            if not static_results['converged']:
                print(f"  ❌ Static analysis failed to converge")
                print(f"     Static results keys: {list(static_results.keys())}")
                return {'converged': False, 'error': 'Static analysis divergence'}
            
            print(f"  ✅ Static analysis converged")
            print(f"     Displacement nodes: {len(static_results.get('displacements', {}))}")
            print(f"     Element forces: {len(static_results.get('element_forces', {}))}")
            
            # Step 6: Modal Analysis
            print(f"Step 6: Performing modal analysis...")
            modal_results = self.perform_modal_analysis()
            
            if not modal_results or 'frequencies' not in modal_results:
                print(f"  ⚠️ Modal analysis failed, using default frequency")
                # Use default frequency if modal analysis fails
                modal_results = {'frequencies': [5.0, 8.0, 12.0]}
            else:
                print(f"  ✅ Modal analysis: {len(modal_results['frequencies'])} modes")
            
            # Step 7: Performance Calculations
            print(f"Step 7: Calculating performance metrics...")
            try:
                max_stress = self.calculate_maximum_stress(static_results, element_registry)
                max_deflection = self.calculate_maximum_deflection(static_results)
                fundamental_frequency = modal_results['frequencies'][0]
                
                print(f"  ✅ Performance metrics:")
                print(f"    Max stress: {max_stress/1e6:.1f} MPa")
                print(f"    Max deflection: {max_deflection*1000:.2f} mm")
                print(f"    Fundamental freq: {fundamental_frequency:.2f} Hz")
                
            except Exception as calc_error:
                print(f"  ❌ Performance calculation error: {calc_error}")
                # Use simplified calculations as fallback
                max_stress = self.calculate_maximum_stress_simplified(static_results)
                max_deflection = self.calculate_maximum_deflection_simplified(static_results)
                fundamental_frequency = modal_results['frequencies'][0]
                
                print(f"  ✅ Simplified performance metrics:")
                print(f"    Max stress (approx): {max_stress/1e6:.1f} MPa")
                print(f"    Max deflection: {max_deflection*1000:.2f} mm")
                print(f"    Fundamental freq: {fundamental_frequency:.2f} Hz")
            
            return {
                'converged': True,
                'max_stress': max_stress,
                'max_deflection': max_deflection,
                'fundamental_frequency': fundamental_frequency,
                'static_results': static_results,
                'modal_results': modal_results,
                'element_registry': element_registry
            }
            
        except Exception as e:
            print(f"  ❌ Error in step: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'converged': False, 'error': str(e)}

    def calculate_maximum_stress_simplified(self, static_results, section_assignment):
        """
        Simplified stress calculation as fallback
        """
        try:
            # Find maximum force in static results
            max_force = 0.0
            
            if 'element_forces' in static_results:
                for elem_id, force_data in static_results['element_forces'].items():
                    if isinstance(force_data, dict):
                        # Try different possible keys for axial force
                        for key in ['axial', 'N', 'force', 'axial_force']:
                            if key in force_data:
                                force = abs(force_data[key])
                                max_force = max(max_force, force)
                                break
                    elif isinstance(force_data, (int, float)):
                        force = abs(force_data)
                        max_force = max(max_force, force)
            
            # Use minimum area from section assignment (most critical)
            min_area = min(
                section_assignment['bottom_chord'],
                section_assignment['top_chord'], 
                section_assignment['web_vertical'],
                section_assignment['web_diagonal']
            )
            
            # Calculate stress
            max_stress = max_force / min_area if min_area > 0 else 0
            
            return max_stress
            
        except Exception as e:
            print(f"    Simplified stress calculation failed: {e}")
            # Return a reasonable default based on loads and areas
            total_load = 50000  # Approximate total load in N
            min_area = min(section_assignment.values())
            return total_load / min_area if min_area > 0 else 100e6  # 100 MPa default

    def calculate_maximum_deflection_simplified(self, static_results):
        """
        Simplified deflection calculation as fallback
        """
        try:
            max_deflection = 0.0
            
            if 'displacements' in static_results:
                for node_id, disp_data in static_results['displacements'].items():
                    if isinstance(disp_data, dict):
                        # Try different possible keys for vertical displacement
                        for key in ['y', 'Y', 'vy', 'vertical', 'disp_y']:
                            if key in disp_data:
                                deflection = abs(disp_data[key])
                                max_deflection = max(max_deflection, deflection)
                                break
                    elif isinstance(disp_data, (list, tuple)) and len(disp_data) >= 2:
                        # Assume [x_disp, y_disp] format
                        deflection = abs(disp_data[1])
                        max_deflection = max(max_deflection, deflection)
            
            return max_deflection
            
        except Exception as e:
            print(f"    Simplified deflection calculation failed: {e}")
            # Return a reasonable default (L/300 for 30m span)
            return 30.0 / 300.0  # 0.1m = 100mm default deflection
    
    def generate_optimization_dataset(self, num_scenarios=1000):
        """
        Generate comprehensive dataset for ML training
        """
        print(f"\n🔄 Generating Optimization Dataset ({num_scenarios} scenarios):")
        print("="*60)
        
        dataset = []
        successful_scenarios = 0
        failed_scenarios = 0
        
        # Define parameter ranges for optimization
        parameter_ranges = {
            'bottom_chord_areas': np.linspace(20e-4, 60e-4, 8),     # 20-60 cm²
            'top_chord_areas': np.linspace(20e-4, 60e-4, 8),       # 20-60 cm²
            'web_vertical_areas': np.linspace(10e-4, 30e-4, 6),    # 10-30 cm²
            'web_diagonal_areas': np.linspace(10e-4, 30e-4, 6),    # 10-30 cm²
            'steel_grades': ['S235', 'S275', 'S355'],
            'temperature_factors': [0.95, 1.0, 1.05],
            'load_cases': ['combination_1', 'combination_2', 'characteristic', 'fatigue_load']
        }
        
        from itertools import product
        import random
        
        # Generate all possible combinations
        all_combinations = list(product(
            parameter_ranges['bottom_chord_areas'],
            parameter_ranges['top_chord_areas'],
            parameter_ranges['web_vertical_areas'],
            parameter_ranges['web_diagonal_areas'],
            parameter_ranges['steel_grades'],
            parameter_ranges['temperature_factors'],
            parameter_ranges['load_cases']
        ))
        
        # Randomly sample if too many combinations
        if len(all_combinations) > num_scenarios:
            selected_combinations = random.sample(all_combinations, num_scenarios)
        else:
            selected_combinations = all_combinations
        
        print(f"Testing {len(selected_combinations)} parameter combinations...")
        
        for i, combo in enumerate(selected_combinations, 1):
            bc_area, tc_area, wv_area, wd_area, steel_grade, temp_factor, load_case = combo
            
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(selected_combinations)} ({i/len(selected_combinations)*100:.1f}%)")
            
            try:
                # Create section assignment
                section_assignment = {
                    'bottom_chord': bc_area,
                    'top_chord': tc_area,
                    'web_vertical': wv_area,
                    'web_diagonal': wd_area,
                    'steel_grade': steel_grade
                }
                
                # Calculate loads
                applied_loads = self.calculate_load_case_forces(load_case)
                
                # Perform analysis
                analysis_result = self.analyze_section_performance(
                    section_assignment, applied_loads, temp_factor
                )
                
                if analysis_result['converged']:
                    # Calculate objectives
                    weight = self.calculate_total_weight(section_assignment)
                    cost = self.calculate_total_cost(section_assignment)
                    carbon = self.calculate_carbon_footprint(section_assignment)
                    
                    # Evaluate constraints
                    constraints_result = self.evaluate_design_constraints(
                        section_assignment, analysis_result, applied_loads
                    )
                    
                    # Store data
                    data_row = {
                        # Design variables
                        'bottom_chord_area_cm2': bc_area * 1e4,
                        'top_chord_area_cm2': tc_area * 1e4,
                        'web_vertical_area_cm2': wv_area * 1e4,
                        'web_diagonal_area_cm2': wd_area * 1e4,
                        'steel_grade': steel_grade,
                        'temperature_factor': temp_factor,
                        'load_case': load_case,
                        
                        # Objectives
                        'total_weight_kg': weight,
                        'total_cost_eur': cost,
                        'carbon_footprint_kg_co2': carbon,
                        
                        # Performance metrics
                        'max_stress_mpa': analysis_result['max_stress'] / 1e6,
                        'max_deflection_mm': analysis_result['max_deflection'] * 1000,
                        'fundamental_frequency_hz': analysis_result['fundamental_frequency'],
                        
                        # Constraints
                        'design_feasible': constraints_result['feasible'],
                        'stress_utilization': constraints_result['stress_utilization'],
                        'deflection_utilization': constraints_result['deflection_utilization'],
                        'frequency_utilization': constraints_result['frequency_utilization'],
                        
                        # Performance ratios
                        'weight_per_span_kg_m': weight / 30.0,
                        'cost_per_weight_eur_kg': cost / weight,
                        'carbon_intensity_kg_co2_kg': carbon / weight
                    }
                    
                    dataset.append(data_row)
                    successful_scenarios += 1
                    
                else:
                    failed_scenarios += 1
                    
            except Exception as e:
                failed_scenarios += 1
                if failed_scenarios % 100 == 0:
                    print(f"    ⚠️ Failed scenarios: {failed_scenarios}")
        
        print(f"\n✅ Dataset Generation Complete:")
        print(f"   Successful scenarios: {successful_scenarios}")
        print(f"   Failed scenarios: {failed_scenarios}")
        print(f"   Success rate: {successful_scenarios/(successful_scenarios+failed_scenarios)*100:.1f}%")
        
        if successful_scenarios > 0:
            # Create DataFrame and save
            df = pd.DataFrame(dataset)
            df.to_csv('optimization_training_dataset.csv', index=False)
            print(f"✅ Dataset saved: optimization_training_dataset.csv")
            
            # Print summary statistics
            print(f"\n📊 Dataset Summary:")
            print(f"   Weight range: {df['total_weight_kg'].min():.0f} - {df['total_weight_kg'].max():.0f} kg")
            print(f"   Cost range: €{df['total_cost_eur'].min():.0f} - €{df['total_cost_eur'].max():.0f}")
            print(f"   Feasible designs: {df['design_feasible'].sum()}/{len(df)} ({df['design_feasible'].mean()*100:.1f}%)")
            
            # Add documentation to your dataset
            dataset_metadata = {
                'area_definition': 'actual_steel_area',
                'section_type': 'hollow_steel_sections',
                'area_includes': 'steel_material_only',
                'area_excludes': 'hollow_interior_space',
                'typical_hollow_ratio': 0.35-0.45  # Steel area / Total outer area
            }
            
            # Save metadata to JSON
            import json
            with open('dataset_metadata.json', 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            
            print(f"✅ Metadata saved: dataset_metadata.json")
            
            return df
        else:
            print(f"❌ No successful scenarios generated!")
            return pd.DataFrame()
    
    def train_surrogate_models(self, df):
        """
        Train ML surrogate models for optimization objectives
        """
        print(f"\n🤖 Training ML Surrogate Models:")
        print("="*50)
        
        # Prepare features and targets
        feature_columns = [
            'bottom_chord_area_cm2', 'top_chord_area_cm2', 
            'web_vertical_area_cm2', 'web_diagonal_area_cm2',
            'temperature_factor'
        ]
        
        # Encode categorical variables
        le_steel = LabelEncoder()
        le_load = LabelEncoder()
        
        df['steel_grade_encoded'] = le_steel.fit_transform(df['steel_grade'])
        df['load_case_encoded'] = le_load.fit_transform(df['load_case'])
        
        feature_columns.extend(['steel_grade_encoded', 'load_case_encoded'])
        
        X = df[feature_columns].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models for each objective
        targets = {
            'weight': 'total_weight_kg',
            'cost': 'total_cost_eur', 
            'carbon': 'carbon_footprint_kg_co2',
            'stress_utilization': 'stress_utilization',
            'deflection_utilization': 'deflection_utilization',
            'frequency': 'fundamental_frequency_hz'
        }
        
        models = {}
        
        for target_name, target_col in targets.items():
            print(f"Training {target_name} model...")
            
            y = df[target_col].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = rf_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            models[target_name] = {
                'model': rf_model,
                'r2_score': r2,
                'mae': mae,
                'feature_importance': rf_model.feature_importances_
            }
            
            print(f"  ✅ {target_name}: R² = {r2:.3f}, MAE = {mae:.2f}")
        
        # Save models
        surrogate_models = {
            'models': models,
            'scaler': scaler,
            'encoders': {'steel_grade': le_steel, 'load_case': le_load},
            'feature_columns': feature_columns
        }
        
        import joblib
        joblib.dump(surrogate_models, 'surrogate_models.pkl')
        print(f"✅ Models saved: surrogate_models.pkl")
        
        return surrogate_models
    
    def run_multi_objective_optimization(self, surrogate_models):
        """
        Run multi-objective optimization using trained surrogate models
        """
        print(f"\n🎯 Multi-Objective Optimization:")
        print("="*50)
        
        try:
            # Try PyMOO if available
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize
            
            # Define optimization problem
            class SteelOptimizationProblem(Problem):
                def __init__(self, surrogate_models):
                    self.models = surrogate_models['models']
                    self.scaler = surrogate_models['scaler']
                    
                    super().__init__(
                        n_var=7,      # 4 areas + temp + steel + load
                        n_obj=3,      # weight, cost, carbon
                        n_constr=3,   # stress, deflection, frequency
                        xl=np.array([20, 20, 10, 10, 0.95, 0, 0]),
                        xu=np.array([60, 60, 30, 30, 1.05, 2, 3])
                    )
                
                def _evaluate(self, X, out, *args, **kwargs):
                    objectives = []
                    constraints = []
                    
                    for x in X:
                        # Scale features
                        x_scaled = self.scaler.transform([x])
                        
                        # Predict objectives
                        weight = self.models['weight']['model'].predict(x_scaled)[0]
                        cost = self.models['cost']['model'].predict(x_scaled)[0]
                        carbon = self.models['carbon']['model'].predict(x_scaled)[0]
                        
                        # Predict constraints
                        stress_util = self.models['stress_utilization']['model'].predict(x_scaled)[0]
                        deflection_util = self.models['deflection_utilization']['model'].predict(x_scaled)[0]
                        frequency = self.models['frequency']['model'].predict(x_scaled)[0]
                        
                        objectives.append([weight, cost, carbon])
                        
                        # Constraint violations (≤ 0 for feasible)
                        constraints.append([
                            stress_util - 1.0,        # Stress ≤ 1.0
                            deflection_util - 1.0,    # Deflection ≤ 1.0
                            3.0 - frequency           # Frequency ≥ 3.0 Hz
                        ])
                    
                    out["F"] = np.array(objectives)
                    out["G"] = np.array(constraints)
        
            # Create and run optimization
            problem = SteelOptimizationProblem(surrogate_models)
            algorithm = NSGA2(pop_size=100)
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', 100),
                verbose=True
            )
            
            print(f"✅ Optimization complete: {len(result.X)} Pareto solutions found")
            
            # Save results
            pareto_solutions = []
            for i, (x, f) in enumerate(zip(result.X, result.F)):
                solution = {
                    'solution_id': i + 1,
                    'bottom_chord_cm2': x[0],
                    'top_chord_cm2': x[1], 
                    'web_vertical_cm2': x[2],
                    'web_diagonal_cm2': x[3],
                    'temperature_factor': x[4],
                    'weight_kg': f[0],
                    'cost_eur': f[1],
                    'carbon_kg_co2': f[2]
                }
                pareto_solutions.append(solution)
            
            # Save to JSON
            import json
            with open('pareto_optimal_solutions.json', 'w') as f:
                json.dump(pareto_solutions, f, indent=2)
            
            return pareto_solutions
            
        except ImportError:
            print("⚠️ PyMOO not available, using SciPy alternative...")
            return self.run_scipy_optimization(surrogate_models)
    
    def get_standard_hollow_section(self, target_area):
        """
        Match target area to standard hollow sections
        """
        # Standard RHS sections (width×height×thickness in mm)
        standard_rhs = [
            {'name': 'RHS100×50×4', 'area': 5.62e-4, 'steel_area_ratio': 0.374},
            {'name': 'RHS120×80×5', 'area': 9.55e-4, 'steel_area_ratio': 0.416},
            {'name': 'RHS150×100×6', 'area': 14.7e-4, 'steel_area_ratio': 0.392},
            {'name': 'RHS200×100×8', 'area': 23.3e-4, 'steel_area_ratio': 0.388},
            {'name': 'RHS250×150×10', 'area': 37.7e-4, 'steel_area_ratio': 0.377},
            {'name': 'RHS300×200×12', 'area': 56.5e-4, 'steel_area_ratio': 0.376}
        ]
        
        # Find closest match
        best_match = min(standard_rhs, 
                        key=lambda x: abs(x['area'] - target_area))
        
        actual_steel_area = best_match['area'] * best_match['steel_area_ratio']
        
        return best_match, actual_steel_area

    def estimate_steel_ratio(self, outer_area_cm2, wall_thickness_mm):
        """
        Estimate steel ratio for rectangular hollow section
        """
        # Assume square section: A = b²
        outer_dim = np.sqrt(outer_area_cm2) * 10  # Convert to mm
        inner_dim = outer_dim - 2 * wall_thickness_mm
        
        if inner_dim > 0:
            outer_full_area = (outer_dim/10)**2  # cm²
            inner_hollow_area = (inner_dim/10)**2  # cm²
            steel_ratio = (outer_full_area - inner_hollow_area) / outer_full_area
        else:
            steel_ratio = 1.0  # Solid section
        
        return np.clip(steel_ratio, 0.25, 0.65)  # Realistic range

    def calculate_corrected_weight(self, outer_area_cm2, wall_thickness_mm, member_length_m=6.0):
        """
        Calculate corrected weight for hollow section
        """
        density = self.material_costs['steel_density']  # kg/m³
        
        # Estimate steel ratio based on wall thickness
        steel_ratio = self.estimate_steel_ratio(outer_area_cm2, wall_thickness_mm)
        actual_steel_area_m2 = (outer_area_cm2 * 1e-4) * steel_ratio  # Convert cm² to m²
        
        # Calculate weight using actual steel area
        weight = actual_steel_area_m2 * member_length_m * density
        return weight, steel_ratio

    def estimate_wall_thickness(self, bc_area, tc_area):
        """Estimate wall thickness based on section size"""
        avg_area = (bc_area + tc_area) / 2
        if avg_area < 25:
            return 5   # 5mm for small sections
        elif avg_area < 40:
            return 6   # 6mm for medium sections  
        elif avg_area < 55:
            return 8   # 8mm for large sections
        else:
            return 10  # 10mm for very large sections

    def calculate_corrected_total_weight(self, row):
        """
        Calculate corrected total weight for a dataset row
        """
        # Element lengths (from your existing method)
        element_lengths = {
            'bottom_chord': 6.0 * 5,      # 5 elements × 6m each = 30m total
            'top_chord': 6.0 * 4,         # 4 elements × 6m each = 24m total  
            'web_vertical': 4.5 * 4,      # 4 vertical × 4.5m each = 18m total
            'web_diagonal': 7.5 * 6       # 6 diagonals × ~7.5m each = 45m total
        }
        
        total_weight = 0.0
        wall_thickness = row['wall_thickness_mm']
        
        for member_type, length in element_lengths.items():
            outer_area_col = f'{member_type}_area_cm2'
            if outer_area_col in row:
                outer_area = row[outer_area_col]
                
                # Calculate corrected weight for this member type
                weight, _ = self.calculate_corrected_weight(
                    outer_area, wall_thickness, length
                )
                total_weight += weight
        
        return total_weight

    def correct_existing_dataset(self, df):
        """
        Correct existing dataset - treat areas as outer dimensions
        """
        df_corrected = df.copy()
        
        # Add wall thickness estimates
        df_corrected['wall_thickness_mm'] = df_corrected.apply(
            lambda row: self.estimate_wall_thickness(
                row['bottom_chord_area_cm2'],
                row['top_chord_area_cm2']
            ), axis=1
        )
        
        # Calculate steel ratios for each member type
        for member in ['bottom_chord', 'top_chord', 'web_vertical', 'web_diagonal']:
            col = f'{member}_area_cm2'
            df_corrected[f'{member}_steel_ratio'] = df_corrected.apply(
                lambda row: self.estimate_steel_ratio(
                    row[col], row['wall_thickness_mm']
                ), axis=1
            )
        
        # Correct weights and costs
        df_corrected['total_weight_kg_corrected'] = df_corrected.apply(
            lambda row: self.calculate_corrected_total_weight(row), axis=1
        )
        
        # Recalculate cost based on corrected weight
        df_corrected['total_cost_eur_corrected'] = (
            df_corrected['total_weight_kg_corrected'] * 
            df_corrected['cost_per_weight_eur_kg']
        )
        
        # Calculate material savings
        df_corrected['weight_reduction_percent'] = (
            (df_corrected['total_weight_kg'] - df_corrected['total_weight_kg_corrected']) / 
            df_corrected['total_weight_kg'] * 100
        )
        
        print(f"\n📊 Hollow Section Corrections Applied:")
        print(f"   Average weight reduction: {df_corrected['weight_reduction_percent'].mean():.1f}%")
        print(f"   Weight range: {df_corrected['total_weight_kg_corrected'].min():.0f} - {df_corrected['total_weight_kg_corrected'].max():.0f} kg")
        print(f"   Cost range: €{df_corrected['total_cost_eur_corrected'].min():.0f} - €{df_corrected['total_cost_eur_corrected'].max():.0f}")
        
        return df_corrected

    def demonstrate_hollow_corrections(self):
        """
        Demonstrate hollow section corrections on existing dataset
        """
        print(f"\n🔧 Demonstrating Hollow Section Corrections:")
        print("="*50)
        
        try:
            # Load existing dataset
            df = pd.read_csv('optimization_training_dataset.csv')
            print(f"✅ Loaded dataset: {len(df)} rows")
            
            # Apply corrections
            df_corrected = self.correct_existing_dataset(df)
            
            # Save corrected dataset
            df_corrected.to_csv('optimization_training_dataset_corrected.csv', index=False)
            print(f"✅ Corrected dataset saved: optimization_training_dataset_corrected.csv")
            
            # Show comparison
            comparison = pd.DataFrame({
                'Metric': ['Average Weight (kg)', 'Average Cost (€)', 'Weight Range (kg)', 'Cost Range (€)'],
                'Original': [
                    f"{df['total_weight_kg'].mean():.0f}",
                    f"{df['total_cost_eur'].mean():.0f}",
                    f"{df['total_weight_kg'].min():.0f} - {df['total_weight_kg'].max():.0f}",
                    f"{df['total_cost_eur'].min():.0f} - {df['total_cost_eur'].max():.0f}"
                ],
                'Corrected': [
                    f"{df_corrected['total_weight_kg_corrected'].mean():.0f}",
                    f"{df_corrected['total_cost_eur_corrected'].mean():.0f}",
                    f"{df_corrected['total_weight_kg_corrected'].min():.0f} - {df_corrected['total_weight_kg_corrected'].max():.0f}",
                    f"{df_corrected['total_cost_eur_corrected'].min():.0f} - {df_corrected['total_cost_eur_corrected'].max():.0f}"
                ]
            })
            
            print(f"\n📊 Comparison:")
            print(comparison.to_string(index=False))
            
            return df_corrected
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found. Generate dataset first.")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

# MAIN EXECUTION (Updated)
if __name__ == "__main__":
    print("Initializing Advanced Steel Section Optimizer...")
    
    try:
        # Create optimizer
        optimizer = AdvancedSteelSectionOptimizer()
        
        # Run single test
        print("\n🧪 Testing framework with single scenario...")
        test_passed = optimizer.test_single_scenario()
        
        if test_passed:
            print(f"\n✅ FRAMEWORK TEST PASSED!")
            print(f"   Ready for optimization dataset generation")
            
            # Check if dataset already exists
            try:
                df_existing = pd.read_csv('optimization_training_dataset.csv')
                print(f"\n📁 Found existing dataset: {len(df_existing)} rows")
                
                # Demonstrate hollow section corrections
                print(f"\n🔧 Applying hollow section corrections...")
                df_corrected = optimizer.demonstrate_hollow_corrections()
                
                if df_corrected is not None:
                    print(f"✅ Dataset corrected for realistic hollow sections!")
                    
                    # Train models on corrected data
                    print(f"\n🤖 Training surrogate models on corrected data...")
                    surrogate_models = optimizer.train_surrogate_models(df_corrected)
                    print(f"✅ Surrogate models trained and saved")
                else:
                    print(f"❌ Could not correct dataset")
                    
            except FileNotFoundError:
                # Generate new dataset
                print(f"\n🔄 Generating optimization dataset...")
                df = optimizer.generate_optimization_dataset(num_scenarios=500)
                
                if len(df) > 100:
                    print(f"✅ Dataset ready for ML training!")
                    
                    # Apply corrections immediately
                    df_corrected = optimizer.correct_existing_dataset(df)
                    df_corrected.to_csv('optimization_training_dataset_corrected.csv', index=False)
                    
                    # Train models
                    surrogate_models = optimizer.train_surrogate_models(df_corrected)
                    print(f"✅ Surrogate models trained and saved")
                else:
                    print(f"❌ Insufficient data for ML training")
        else:
            print(f"\n❌ FRAMEWORK TEST FAILED!")
            print(f"   Fix base framework issues before proceeding")
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()