# OpenSees Analysis of a Two-Degree-of-Freedom (2DOF) System

This document explains how to use **OpenSees** to analyze a two-degree-of-freedom (2DOF) system consisting of two masses connected by springs, as described in the provided problem. The system has two masses (\( M_1 = 2 \), \( M_2 = 1 \)) on a frictionless surface, connected by springs (\( K_1 = 3 \) between mass 1 and a fixed wall, \( K_2 = 2 \) between mass 1 and mass 2), with an external force \( F(t) \) applied to mass 2. The analysis includes computing natural frequencies, mode shapes, and dynamic responses (free vibration and forced response, including resonance) using OpenSees, matching the analytical results provided.

## Problem Description

- **System**: Two masses (\( M_1 = 2 \), \( M_2 = 1 \)) connected by springs (\( K_1 = 3 \), \( K_2 = 2 \)). An external force \( F(t) \) is applied to mass 2.
- **Coordinates**: \( X_1 \) (displacement of mass 1), \( X_2 \) (displacement of mass 2).
- **Stiffness Matrix**:
  \[
  K = \begin{bmatrix}
  K_1 + K_2 & -K_2 \\
  -K_2 & K_2
  \end{bmatrix} = \begin{bmatrix}
  5 & -2 \\
  -2 & 2
  \end{bmatrix}
  \]
- **Mass Matrix**:
  \[
  M = \begin{bmatrix}
  M_1 & 0 \\
  0 & M_2
  \end{bmatrix} = \begin{bmatrix}
  2 & 0 \\
  0 & 1
  \end{bmatrix}
  \]
- **Analytical Results**:
  - Eigenvalues: \( \omega_1^2 = \frac{9 - \sqrt{33}}{4} \approx 0.811 \), \( \omega_2^2 = \frac{9 + \sqrt{33}}{4} \approx 3.689 \)
  - Natural Frequencies: \( \omega_1 \approx 0.901 \) rad/s, \( \omega_2 \approx 1.9199 \) rad/s
  - Mode Shapes:
    - Mode 1 (symmetric): \( \begin{bmatrix} \frac{-1 + \sqrt{33}}{8} \\ 1 \end{bmatrix} \approx \begin{bmatrix} 0.593 \\ 1 \end{bmatrix} \)
    - Mode 2 (anti-symmetric): \( \begin{bmatrix} \frac{-1 - \sqrt{33}}{8} \\ 1 \end{bmatrix} \approx \begin{bmatrix} -0.843 \\ 1 \end{bmatrix} \)
- **Objectives**:
  1. Compute natural frequencies and mode shapes via eigenvalue analysis.
  2. Simulate free vibration with initial displacements (mode 1 shape).
  3. Simulate forced vibration with \( F(t) = 10 \sin(\omega_1 t) \) to observe resonance.

## OpenSees Implementation

OpenSees solves the eigenvalue problem \( |K - \omega^2 M| = 0 \) for frequencies and mode shapes and integrates the equations of motion \( M\ddot{x} + Kx = F(t) \) for dynamic responses. The following Tcl script sets up the model, performs eigenvalue analysis, and simulates both free and forced vibrations.

### Tcl Script for OpenSees

```tcl
# Clear previous model
wipe;

# Define 2D model with 2 DOFs per node (only x-direction used)
model Basic -ndm 2 -ndf 2;

# Define nodes (masses at x-coordinates for simplicity; y=0)
node 1 0.0 0.0;  # Mass 1
node 2 1.0 0.0;  # Mass 2
node 0 -1.0 0.0; # Fixed wall (for spring K1)

# Assign masses (M1=2, M2=1)
mass 1 2.0 0.0;  # Mass 1: 2 units in x, 0 in y
mass 2 1.0 0.0;  # Mass 2: 1 unit in x, 0 in y

# Fix the wall node (Node 0)
fix 0 1 1;  # Fixed in x and y

# Define material for springs (elastic, stiffness defined later)
uniaxialMaterial Elastic 1 1.0;  # Placeholder (stiffness set in elements)

# Define zero-length elements for springs
# K1=3 between wall (Node 0) and Mass 1 (Node 1)
element zeroLength 1 0 1 -mat 1 -dir 1 -doRayleigh 0;
set K1 3.0;
geomTransf Linear 1;
# K2=2 between Mass 1 (Node 1) and Mass 2 (Node 2)
element zeroLength 2 1 2 -mat 1 -dir 1 -doRayleigh 0;
set K2 2.0;

# Update element stiffness
element zeroLength 1 0 1 -mat 1 -dir 1 -doRayleigh 0 -factor $K1;
element zeroLength 2 1 2 -mat 1 -dir 1 -doRayleigh 0 -factor $K2;

# Eigenvalue analysis
set numModes 2;
set lambda [eigen -fullGenLapack $numModes];
set omega1 [expr sqrt([lindex $lambda 0])];
set omega2 [expr sqrt([lindex $lambda 1])];
puts "Natural Frequencies (rad/s): Omega1 = $omega1, Omega2 = $omega2";

# Get mode shapes
set mode1 [nodeEigenVec 1 1];
set mode2 [nodeEigenVec 1 2];
puts "Mode 1 Shape: [nodeEigenVec 1 1 1], [nodeEigenVec 2 1 1]";
puts "Mode 2 Shape: [nodeEigenVec 1 2 1], [nodeEigenVec 2 2 1]";

# Free Vibration: Mode 1 Initial Displacement
wipeAnalysis;
timeSeries Constant 1;
pattern Plain 1 1 {
    # No external loads
}
constraints Transformation;
numberer RCM;
system BandGeneral;
test NormDispIncr 1.0e-8 6;
algorithm Linear;
integrator Newmark 0.5 0.25;  # Newmark for dynamic analysis
analysis Transient;

# Set initial displacements for Mode 1 (scaled to X2=1)
set phi1_x1 [expr (-1 + sqrt(33))/8.0];  # ≈ 0.593
set phi1_x2 1.0;
setDisp 1 1 $phi1_x1;
setDisp 2 1 $phi1_x2;

# Run transient analysis for 10 seconds
set dt 0.01;
set nSteps 1000;
analyze $nSteps $dt;

# Record displacements
recorder Node -file mode1_disp.txt -time -node 1 2 -dof 1 disp;

# Forced Vibration: Resonance at Omega1
wipeAnalysis;
timeSeries Trig 2 0.0 10.0 $omega1;  # Sin(omega1*t) for 10s
pattern Plain 2 2 {
    load 2 10.0 0.0;  # 10N force on Mass 2 in x-direction
}
constraints Transformation;
numberer RCM;
system BandGeneral;
test NormDispIncr 1.0e-8 6;
algorithm Linear;
integrator Newmark 0.5 0.25;
analysis Transient;
recorder Node -file resonance_omega1.txt -time -node 1 2 -dof 1 disp;
analyze $nSteps $dt;
```

### Script Explanation

1. **Model Setup**:
   - Defines a 2D model with 2 DOFs per node (x, y), though only x-direction motion is active.
   - Nodes represent masses (\( M_1 \), \( M_2 \)) and the fixed wall.
   - Masses are assigned (\( M_1 = 2 \), \( M_2 = 1 \)).
   - The wall node is fixed.

2. **Springs**:
   - Uses `zeroLength` elements to model springs with stiffness \( K_1 = 3 \) (wall to mass 1) and \( K_2 = 2 \) (mass 1 to mass 2).
   - `-dir 1` ensures stiffness acts in the x-direction.

3. **Eigenvalue Analysis**:
   - The `eigen -fullGenLapack 2` command solves \( |K - \lambda M| = 0 \), computing eigenvalues (\( \lambda_i = \omega_i^2 \)).
   - Mode shapes are retrieved using `nodeEigenVec` for x-direction DOFs.

4. **Free Vibration**:
   - Sets initial displacements to mode 1 shape (\( X_1 \approx 0.593 \), \( X_2 = 1 \)).
   - Uses Newmark integration (`integrator Newmark 0.5 0.25`) to solve \( M\ddot{x} + Kx = 0 \) over 10 seconds.

5. **Forced Vibration**:
   - Applies a sinusoidal force \( F(t) = 10 \sin(\omega_1 t) \) to mass 2.
   - Records displacements to observe resonance (unbounded growth).

### Running in VS Code

1. **Save the Script**:
   - Save as `2DOF.tcl` in a directory (e.g., `C:\2DOF`).
   - Ensure UTF-8 encoding and Windows CRLF line endings in VS Code.

2. **Configure Task** (in `.vscode/tasks.json`):
   ```json
   {
       "version": "2.0.0",
       "tasks": [
           {
               "label": "Run OpenSees",
               "type": "shell",
               "command": "OpenSees",
               "args": ["${file}"],
               "group": {
                   "kind": "build",
                   "isDefault": true
               },
               "problemMatcher": []
           }
       ]
   }
   ```

3. **Run**:
   - Open `2DOF.tcl` in VS Code.
   - Press `Ctrl+Shift+B` to execute.
   - Check terminal for frequencies and mode shapes.
   - Open `mode1_disp.txt` and `resonance_omega1.txt` for displacement data.

### Expected Results

1. **Natural Frequencies**:
   - \( \omega_1 \approx 0.901 \) rad/s
   - \( \omega_2 \approx 1.920 \) rad/s
   - Matches analytical results.

2. **Mode Shapes**:
   - Mode 1: \( \begin{bmatrix} 0.593 \\ 1 \end{bmatrix} \)
   - Mode 2: \( \begin{bmatrix} -0.843 \\ 1 \end{bmatrix} \)
   - Matches analytical eigenvectors.

3. **Free Vibration**:
   - `mode1_disp.txt` shows both masses oscillating at \( \omega_1 \approx 0.901 \) rad/s, in phase, with amplitudes proportional to mode 1.

4. **Resonance**:
   - `resonance_omega1.txt` shows linearly growing displacements, confirming resonance at \( \omega_1 \).

### Visualization

To plot results (e.g., in Python):
```python
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('resonance_omega1.txt')
time = data[:, 0]
x1 = data[:, 1]  # Node 1, DOF 1
x2 = data[:, 2]  # Node 2, DOF 1
plt.plot(time, x1, label='Mass 1')
plt.plot(time, x2, label='Mass 2')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Resonance at Omega1')
plt.legend()
plt.show()
```

### Notes
- **No Damping**: The model assumes no damping, leading to unbounded resonance. Add Rayleigh damping (`rayleigh`) if needed.
- **Troubleshooting**: If errors like `invalid command name "."` occur, retype the script in VS Code and verify encoding.
- **Extensions**: Use the Tcl extension (`actboy168`) in VS Code for syntax highlighting.

This script replicates the analytical eigenvalue problem and dynamic simulations, providing a numerical framework for the 2DOF system in OpenSees.