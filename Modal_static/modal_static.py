import os
import sys

# Add OpenSees path to environment to avoid Tcl conflicts
opensees_path = r'C:\OpenSees3.7.1'
if os.path.exists(opensees_path):
    os.environ['PATH'] = os.path.join(opensees_path, 'bin') + ';' + os.environ['PATH']
    # Set Tcl library paths
    tcl_path = os.path.join(opensees_path, 'lib')
    if os.path.exists(tcl_path):
        os.environ['TCL_LIBRARY'] = os.path.join(tcl_path, 'tcl8.6')
        os.environ['TK_LIBRARY'] = os.path.join(tcl_path, 'tk8.6')

import openseespy.opensees as ops
import numpy as np
import pandas as pd

# Fix matplotlib import and backend setting
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import json
import ast
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class StaticModalAnalysisFramework:
    """
    Unified framework for analyzing relationships between static and modal data
    with area reduction effects
    """
    
    def __init__(self):
        self.element_info = {}
        self.analysis_data = []
        self.baseline_data = {}
        
    def create_truss_model(self, temperature_factor=1.0):
        """Create the baseline truss model"""
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 2)

        # Define nodes
        ops.node(1, 0.0, 0.0)
        ops.node(3, 6.0, 0.0)
        ops.node(5, 12.0, 0.0)
        ops.node(7, 18.0, 0.0)
        ops.node(9, 24.0, 0.0)
        ops.node(11, 30.0, 0.0)

        height = 4.5
        ops.node(2, 3.0, height)
        ops.node(4, 9.0, height)
        ops.node(6, 15.0, height)
        ops.node(8, 21.0, height)
        ops.node(10, 27.0, height)

        # Material properties
        E_base = 200000.0e6  # Pa
        E = E_base * temperature_factor
        A_chord = 0.01  # m²
        A_web = 0.005   # m²

        ops.uniaxialMaterial('Elastic', 1, E)
        return A_chord, A_web, E_base

    def create_elements(self, A_chord, A_web, area_reductions=None):
        """
        Create elements with area reductions
        area_reductions: dict {element_id: reduction_percentage}
        """
        element_tag = 1
        element_info = {}
        
        if area_reductions is None:
            area_reductions = {}

        # Bottom chord elements
        bottom_connections = [(1,3), (3,5), (5,7), (7,9), (9,11)]
        for connection in bottom_connections:
            area = A_chord
            reduction = area_reductions.get(element_tag, 0.0)
            area *= (1 - reduction/100.0)

            ops.element('Truss', element_tag, connection[0], connection[1], area, 1)
            element_info[element_tag] = {
                'nodes': connection,
                'type': 'bottom_chord',
                'original_area': A_chord,
                'actual_area': area,
                'area_reduction_pct': reduction,
                'length': self.calculate_element_length(connection[0], connection[1])
            }
            element_tag += 1

        # Top chord elements
        top_connections = [(2,4), (4,6), (6,8), (8,10)]
        for connection in top_connections:
            area = A_chord
            reduction = area_reductions.get(element_tag, 0.0)
            area *= (1 - reduction/100.0)

            ops.element('Truss', element_tag, connection[0], connection[1], area, 1)
            element_info[element_tag] = {
                'nodes': connection,
                'type': 'top_chord',
                'original_area': A_chord,
                'actual_area': area,
                'area_reduction_pct': reduction,
                'length': self.calculate_element_length(connection[0], connection[1])
            }
            element_tag += 1

        # Web elements
        web_connections = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                          (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        for connection in web_connections:
            area = A_web
            reduction = area_reductions.get(element_tag, 0.0)
            area *= (1 - reduction/100.0)

            ops.element('Truss', element_tag, connection[0], connection[1], area, 1)
            element_info[element_tag] = {
                'nodes': connection,
                'type': 'web_member',
                'original_area': A_web,
                'actual_area': area,
                'area_reduction_pct': reduction,
                'length': self.calculate_element_length(connection[0], connection[1])
            }
            element_tag += 1

        return element_tag - 1, element_info

    def calculate_element_length(self, node1, node2):
        """Calculate element length from node coordinates"""
        coord1 = ops.nodeCoord(node1)
        coord2 = ops.nodeCoord(node2)
        return np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)

    def apply_boundary_and_loads(self, load_factor=1.0):
        """Apply supports and loads"""
        # Boundary conditions
        ops.fix(1, 1, 1)  # Pin
        ops.fix(11, 0, 1)  # Roller

        # Define loads for static analysis
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)

        # Apply point loads
        load_nodes = [2, 4, 6, 8, 10]
        unit_load = -10000.0  # 10 kN downward per node
        
        for node in load_nodes:
            ops.load(node, 0.0, load_factor * unit_load)

        # Add masses for modal analysis
        node_mass = 100.0  # kg per node
        all_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for node in all_nodes:
            ops.mass(node, node_mass, node_mass)

        return load_nodes, unit_load

    def perform_static_analysis(self):
        """Perform static analysis and extract results"""
        # Setup analysis
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1.0e-6, 50)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.1)
        ops.analysis('Static')

        # Run analysis with load stepping
        success = True
        for step in range(10):  # 10 load steps
            result = ops.analyze(1)
            if result != 0:
                success = False
                break

        if not success:
            return None

        # Extract static response
        all_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        # Node displacements
        displacements = {}
        max_displacement = 0
        for node in all_nodes:
            disp = ops.nodeDisp(node)
            displacements[node] = {'x': disp[0], 'y': disp[1]}
            max_displacement = max(max_displacement, abs(disp[0]), abs(disp[1]))

        # Element forces and stresses
        element_data = {}
        max_stress = 0
        total_strain_energy = 0
        
        for elem_id in range(1, len(self.element_info) + 1):
            try:
                force = ops.eleForce(elem_id)
                axial_force = force[0]
                
                area = self.element_info[elem_id]['actual_area']
                length = self.element_info[elem_id]['length']
                
                # Calculate stress and strain energy
                stress = abs(axial_force) / area if area > 0 else 0
                strain_energy = (axial_force**2 * length) / (2 * 200000.0e6 * area) if area > 0 else 0
                
                element_data[elem_id] = {
                    'axial_force': axial_force,
                    'stress': stress,
                    'strain_energy': strain_energy
                }
                
                max_stress = max(max_stress, stress)
                total_strain_energy += strain_energy
                
            except:
                element_data[elem_id] = {'axial_force': 0, 'stress': 0, 'strain_energy': 0}

        # Support reactions
        reactions = {}
        for node in [1, 11]:
            reaction = ops.nodeReaction(node)
            reactions[node] = {'rx': reaction[0], 'ry': reaction[1]}

        return {
            'success': True,
            'displacements': displacements,
            'max_displacement': max_displacement,
            'element_forces': element_data,
            'max_stress': max_stress,
            'total_strain_energy': total_strain_energy,
            'reactions': reactions
        }

    def perform_modal_analysis(self, num_modes=6):
        """Perform modal analysis and extract frequencies and mode shapes"""
        eigenvalues = ops.eigen(num_modes)

        if not eigenvalues or any(ev <= 0 for ev in eigenvalues):
            return None

        frequencies = []
        mode_shapes = {}
        all_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for i, eigenval in enumerate(eigenvalues):
            omega = eigenval**0.5
            frequency = omega / (2 * np.pi)
            frequencies.append(frequency)

            # Get mode shapes
            mode_shapes[i+1] = {}
            for node in all_nodes:
                try:
                    shape = ops.nodeEigenvector(node, i+1)
                    mode_shapes[i+1][node] = {'x': shape[0], 'y': shape[1]}
                except:
                    mode_shapes[i+1][node] = {'x': 0.0, 'y': 0.0}

        return {
            'success': True,
            'frequencies': frequencies,
            'mode_shapes': mode_shapes
        }

    def generate_combined_dataset(self):
        """Generate dataset with both static and modal data"""
        print("Generating Combined Static-Modal Dataset...")
        print("="*50)
        
        dataset = []
        total_elements = 0
        
        # Get baseline (healthy) structure
        print("Analyzing healthy structure...")
        A_chord, A_web, E_base = self.create_truss_model()
        total_elements, self.element_info = self.create_elements(A_chord, A_web)
        self.apply_boundary_and_loads()

        # Get baseline data
        baseline_static = self.perform_static_analysis()
        baseline_modal = self.perform_modal_analysis()
        
        if baseline_static and baseline_modal:
            self.baseline_data = {
                'static': baseline_static,
                'modal': baseline_modal
            }
            print("✓ Baseline analysis completed")
        else:
            print("❌ Baseline analysis failed")
            return None

        # Generate scenarios with area reductions
        area_reduction_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]  # Percentage reductions
        
        # Single element reductions
        print("\nGenerating single element area reduction scenarios...")
        for element_id in range(1, total_elements + 1):
            for reduction_pct in area_reduction_levels:
                if reduction_pct == 0:  # Skip 0% for individual elements (already have baseline)
                    continue
                    
                print(f"Analyzing Element {element_id}, {reduction_pct}% reduction...")
                
                # Create model with area reduction
                ops.wipe()
                A_chord, A_web, _ = self.create_truss_model()
                _, element_info = self.create_elements(A_chord, A_web, 
                                                    area_reductions={element_id: reduction_pct})
                self.apply_boundary_and_loads()

                # Perform analyses
                static_results = self.perform_static_analysis()
                modal_results = self.perform_modal_analysis()

                if static_results and modal_results:
                    # Create comprehensive data row
                    data_row = self.create_data_row(
                        reduction_type='single_element',
                        affected_elements=[element_id],
                        reduction_percentages=[reduction_pct],
                        static_results=static_results,
                        modal_results=modal_results,
                        element_info=element_info
                    )
                    dataset.append(data_row)

        # Multiple element reductions (selected combinations)
        print("\nGenerating multiple element area reduction scenarios...")
        selected_pairs = [(1, 6), (3, 8), (5, 10), (2, 7), (4, 9)]  # Representative pairs
        
        for elem1, elem2 in selected_pairs:
            for reduction1 in [10, 20, 30]:
                for reduction2 in [10, 20, 30]:
                    print(f"Analyzing Elements {elem1},{elem2}, {reduction1}%,{reduction2}% reduction...")
                    
                    # Create model with area reductions
                    ops.wipe()
                    A_chord, A_web, _ = self.create_truss_model()
                    _, element_info = self.create_elements(A_chord, A_web,
                                                        area_reductions={elem1: reduction1, elem2: reduction2})
                    self.apply_boundary_and_loads()

                    # Perform analyses
                    static_results = self.perform_static_analysis()
                    modal_results = self.perform_modal_analysis()

                    if static_results and modal_results:
                        data_row = self.create_data_row(
                            reduction_type='two_elements',
                            affected_elements=[elem1, elem2],
                            reduction_percentages=[reduction1, reduction2],
                            static_results=static_results,
                            modal_results=modal_results,
                            element_info=element_info
                        )
                        dataset.append(data_row)

        # Add baseline case
        baseline_row = self.create_data_row(
            reduction_type='healthy',
            affected_elements=[],
            reduction_percentages=[],
            static_results=baseline_static,
            modal_results=baseline_modal,
            element_info=self.element_info
        )
        dataset.append(baseline_row)

        print(f"\n✓ Dataset generation completed: {len(dataset)} scenarios")
        return dataset

    def create_data_row(self, reduction_type, affected_elements, reduction_percentages, 
                       static_results, modal_results, element_info):
        """Create a comprehensive data row combining static and modal data"""
        
        data_row = {
            'reduction_type': reduction_type,
            'affected_elements': affected_elements,
            'reduction_percentages': reduction_percentages,
            'case_description': f"{reduction_type}: Elements {affected_elements}, Reductions {reduction_percentages}%"
        }

        # Static response features
        data_row['max_displacement'] = static_results['max_displacement']
        data_row['max_stress'] = static_results['max_stress']
        data_row['total_strain_energy'] = static_results['total_strain_energy']

        # Displacement features for key nodes
        critical_nodes = [2, 4, 6, 8, 10]  # Top chord nodes
        for node in critical_nodes:
            disp = static_results['displacements'][node]
            data_row[f'disp_node_{node}_x'] = disp['x']
            data_row[f'disp_node_{node}_y'] = disp['y']
            data_row[f'disp_node_{node}_magnitude'] = np.sqrt(disp['x']**2 + disp['y']**2)

        # Element force and stress features
        for elem_id, elem_data in static_results['element_forces'].items():
            data_row[f'force_elem_{elem_id}'] = elem_data['axial_force']
            data_row[f'stress_elem_{elem_id}'] = elem_data['stress']
            data_row[f'strain_energy_elem_{elem_id}'] = elem_data['strain_energy']

        # Modal features
        frequencies = modal_results['frequencies']
        for i, freq in enumerate(frequencies):
            data_row[f'frequency_{i+1}'] = freq

        # Mode shape features for critical nodes
        for mode in range(1, len(frequencies) + 1):
            for node in critical_nodes:
                shape = modal_results['mode_shapes'][mode][node]
                data_row[f'mode_{mode}_node_{node}_x'] = shape['x']
                data_row[f'mode_{mode}_node_{node}_y'] = shape['y']
                data_row[f'mode_{mode}_node_{node}_magnitude'] = np.sqrt(shape['x']**2 + shape['y']**2)

        # Calculate changes relative to baseline (if baseline exists)
        if hasattr(self, 'baseline_data') and self.baseline_data:
            # Static changes
            baseline_static = self.baseline_data['static']
            data_row['max_displacement_change'] = static_results['max_displacement'] - baseline_static['max_displacement']
            data_row['max_stress_change'] = static_results['max_stress'] - baseline_static['max_stress']
            data_row['strain_energy_change'] = static_results['total_strain_energy'] - baseline_static['total_strain_energy']

            # Frequency changes
            baseline_frequencies = self.baseline_data['modal']['frequencies']
            for i, (freq, baseline_freq) in enumerate(zip(frequencies, baseline_frequencies)):
                data_row[f'freq_change_{i+1}'] = freq - baseline_freq
                data_row[f'freq_change_pct_{i+1}'] = ((freq - baseline_freq) / baseline_freq) * 100 if baseline_freq != 0 else 0

        # Area reduction summary
        total_area_reduced = sum(reduction_percentages) if reduction_percentages else 0
        data_row['total_area_reduction'] = total_area_reduced
        data_row['num_elements_reduced'] = len(affected_elements)
        
        # Calculate effective stiffness reduction (approximate)
        if element_info and affected_elements:
            total_original_area = sum(element_info[elem_id]['original_area'] for elem_id in affected_elements)
            total_reduced_area = sum(element_info[elem_id]['actual_area'] for elem_id in affected_elements)
            data_row['effective_stiffness_reduction_pct'] = ((total_original_area - total_reduced_area) / total_original_area) * 100 if total_original_area > 0 else 0
        else:
            data_row['effective_stiffness_reduction_pct'] = 0.0

        return data_row

    def save_dataset(self, dataset, filename='static_modal_combined_dataset.csv'):
        """Save the combined dataset"""
        df = pd.DataFrame(dataset)
        df.to_csv(filename, index=False)
        print(f"✓ Combined dataset saved: {filename}")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Features per sample: {len(df.columns)}")
        return df

    def analyze_static_modal_correlations(self, df):
        """Analyze correlations between static and modal parameters"""
        print("\n" + "="*60)
        print("STATIC-MODAL CORRELATION ANALYSIS")
        print("="*60)

        # Define static and modal feature groups
        static_features = [col for col in df.columns if any(x in col for x in 
                          ['displacement', 'stress', 'force', 'strain_energy']) and 'change' not in col]
        modal_features = [col for col in df.columns if 'frequency' in col and 'change' not in col]
        
        print(f"Static features: {len(static_features)}")
        print(f"Modal features: {len(modal_features)}")

        # Calculate correlation matrix between static and modal features
        correlation_data = []
        
        for static_feat in static_features[:10]:  # Limit for visualization
            for modal_feat in modal_features:
                try:
                    corr_coef, p_value = pearsonr(df[static_feat], df[modal_feat])
                    correlation_data.append({
                        'static_feature': static_feat,
                        'modal_feature': modal_feat,
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                except:
                    continue

        corr_df = pd.DataFrame(correlation_data)
        
        # Visualize strongest correlations using matplotlib
        plt.figure(figsize=(12, 8))
        
        # Filter for significant correlations
        significant_corr = corr_df[corr_df['significant'] & (abs(corr_df['correlation']) > 0.5)]
        
        if not significant_corr.empty:
            # Create pivot table for heatmap
            pivot_corr = significant_corr.pivot(index='static_feature', 
                                              columns='modal_feature', 
                                              values='correlation')
            
            # Use matplotlib imshow instead of seaborn heatmap
            im = plt.imshow(pivot_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im, label='Correlation Coefficient')
            
            # Set ticks and labels
            plt.xticks(range(len(pivot_corr.columns)), pivot_corr.columns, rotation=45, ha='right')
            plt.yticks(range(len(pivot_corr.index)), pivot_corr.index)
            
            # Add correlation values as text
            for i in range(len(pivot_corr.index)):
                for j in range(len(pivot_corr.columns)):
                    value = pivot_corr.iloc[i, j]
                    if not np.isnan(value):
                        plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                               color='white' if abs(value) > 0.7 else 'black')
            
            plt.title('Significant Static-Modal Correlations (|r| > 0.5, p < 0.05)')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to save memory
            
            print(f"Found {len(significant_corr)} significant correlations")
        else:
            print("No significant strong correlations found")

        return corr_df

    def build_prediction_models(self, df):
        """Build ML models to predict static from modal and vice versa"""
        print("\n" + "="*60)
        print("STATIC-MODAL PREDICTION MODELS")
        print("="*60)

        # Define feature sets
        modal_features = [col for col in df.columns if 'frequency' in col and 'change' not in col]
        static_targets = ['max_displacement', 'max_stress', 'total_strain_energy']
        
        results = {}

        # Model 1: Predict static response from modal properties
        print("\n1. Predicting Static Response from Modal Properties")
        print("-" * 50)
        
        X_modal = df[modal_features].values
        
        for target in static_targets:
            if target in df.columns:
                y = df[target].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_modal, y, test_size=0.3, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Evaluate
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                results[f'modal_to_{target}'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'model': model,
                    'scaler': scaler
                }
                
                print(f"  {target}: R² = {r2:.3f}, MSE = {mse:.2e}")

        # Model 2: Predict modal properties from static response
        print("\n2. Predicting Modal Properties from Static Response")
        print("-" * 50)
        
        static_features = ['max_displacement', 'max_stress', 'total_strain_energy']
        static_features = [f for f in static_features if f in df.columns]
        X_static = df[static_features].values
        
        for i, freq_col in enumerate(modal_features[:3]):  # First 3 frequencies
            y = df[freq_col].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_static, y, test_size=0.3, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            results[f'static_to_{freq_col}'] = {
                'r2_score': r2,
                'mse': mse,
                'model': model,
                'scaler': scaler
            }
            
            print(f"  {freq_col}: R² = {r2:.3f}, MSE = {mse:.2e}")

        return results

    def create_comprehensive_plots(self, df):
        """Create comprehensive visualization plots"""
        print("\nCreating Comprehensive Visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Frequency vs Displacement
        axes[0,0].scatter(df['max_displacement'], df['frequency_1'], alpha=0.6, c=df['total_area_reduction'], cmap='viridis')
        axes[0,0].set_xlabel('Max Displacement (m)')
        axes[0,0].set_ylabel('1st Natural Frequency (Hz)')
        axes[0,0].set_title('Frequency vs Displacement')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Frequency vs Stress  
        axes[0,1].scatter(df['max_stress'], df['frequency_1'], alpha=0.6, c=df['total_area_reduction'], cmap='viridis')
        axes[0,1].set_xlabel('Max Stress (Pa)')
        axes[0,1].set_ylabel('1st Natural Frequency (Hz)')
        axes[0,1].set_title('Frequency vs Stress')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Area Reduction vs Frequency Change
        if 'freq_change_1' in df.columns:
            axes[0,2].scatter(df['total_area_reduction'], df['freq_change_1'], alpha=0.6)
            axes[0,2].set_xlabel('Total Area Reduction (%)')
            axes[0,2].set_ylabel('Frequency Change (Hz)')
            axes[0,2].set_title('Area Reduction vs Frequency Change')
            axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Strain Energy vs Frequencies
        axes[1,0].scatter(df['total_strain_energy'], df['frequency_1'], alpha=0.6, c=df['total_area_reduction'], cmap='viridis')
        axes[1,0].set_xlabel('Total Strain Energy (J)')
        axes[1,0].set_ylabel('1st Natural Frequency (Hz)')
        axes[1,0].set_title('Strain Energy vs Frequency')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Multiple Frequency Comparison
        freq_cols = [col for col in df.columns if 'frequency_' in col and 'change' not in col][:3]
        for i, col in enumerate(freq_cols):
            axes[1,1].scatter(df['total_area_reduction'], df[col], alpha=0.6, label=f'Mode {i+1}')
        axes[1,1].set_xlabel('Total Area Reduction (%)')
        axes[1,1].set_ylabel('Natural Frequencies (Hz)')
        axes[1,1].set_title('Area Reduction vs Natural Frequencies')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Static vs Modal Relationship Summary
        static_modal_ratio = df['max_displacement'] * df['frequency_1']
        axes[1,2].scatter(df['total_area_reduction'], static_modal_ratio, alpha=0.6)
        axes[1,2].set_xlabel('Total Area Reduction (%)')
        axes[1,2].set_ylabel('Displacement × Frequency')
        axes[1,2].set_title('Static-Modal Product vs Area Reduction')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('static_modal_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("STATIC-MODAL COMBINED ANALYSIS FRAMEWORK")
    print("="*70)
    
    # Initialize framework
    framework = StaticModalAnalysisFramework()
    
    # Generate combined dataset
    dataset = framework.generate_combined_dataset()
    
    if dataset:
        # Save dataset
        df = framework.save_dataset(dataset)
        
        # Analyze correlations
        correlation_df = framework.analyze_static_modal_correlations(df)
        
        # Build prediction models
        prediction_results = framework.build_prediction_models(df)
        
        # Create visualizations
        framework.create_comprehensive_plots(df)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("Generated files:")
        print("1. static_modal_combined_dataset.csv - Complete dataset")
        print("2. static_modal_analysis_plots.png - Visualization plots")
        
        print(f"\nDataset Summary:")
        print(f"- Total scenarios: {len(df)}")
        print(f"- Features per sample: {len(df.columns)}")
        print(f"- Static features: {len([col for col in df.columns if any(x in col for x in ['displacement', 'stress', 'force', 'strain_energy'])])}")
        print(f"- Modal features: {len([col for col in df.columns if 'frequency' in col or 'mode_' in col])}")
        
        print("\n🎯 Key Insights Available:")
        print("✓ Static response vs modal properties correlations")
        print("✓ Area reduction effects on both static and modal behavior")
        print("✓ ML models for cross-prediction (static ↔ modal)")
        print("✓ Comprehensive visualization of relationships")
        
    else:
        print("❌ Dataset generation failed!")