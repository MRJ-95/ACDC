Aspirin-COX Binding Dynamics: A Molecular Simulation Study
A simplified molecular dynamics simulation to analyze the binding interactions between aspirin and the cyclooxygenase (COX) enzyme. This project serves as an educational model to illustrate key concepts in computational drug discovery.

About The Project
Aspirin is a widely used NSAID that functions by inhibiting cyclooxygenase (COX) enzymes, which are key to the inflammatory response. This simulation investigates the molecular mechanism of this interaction. Using a simplified model, we run a molecular dynamics simulation to characterize the binding pose, estimate binding affinity, and analyze the primary forces driving the interaction.


The project aims to:

Simulate the molecular interactions between aspirin and the COX enzyme binding site.
Identify and quantify the contributions of different interaction types (van der Waals, hydrophobic, etc.).
Analyze the structural basis for aspirin's known inhibitory mechanism.
Key Findings
Dominant Interactions: The simulation shows that binding is primarily driven by van der Waals (65.2%) and hydrophobic (15.2%) interactions.
Energy Stabilization: The system's potential energy stabilized rapidly, indicating the successful optimization of a binding pose.
Binding Affinity: The simulation yielded a strong calculated binding energy of -32.41 kcal/mol. The estimated Kd was physically unrealistic, which highlights the approximations of the model.
Mechanism Support: The results are consistent with the known mechanism of covalent modification at the Ser530 residue in the COX active site.

Simulation Results
The following plots summarize the key results from the 500-step molecular dynamics simulation.

Methodology
The simulation was built using a simplified model and classical force fields.

Molecular System: A 14-atom model for aspirin and a 13-atom model for the COX binding site were used.
Simulation Algorithm: A Velocity Verlet molecular dynamics algorithm was run for 500 steps.
Force Field: Interactions were modeled using a Lennard-Jones potential and Coulombic electrostatics.
Software: The simulation and analysis were performed using Python with NumPy, SciPy, and Matplotlib.
Limitations
This is a simplified educational model and has several limitations for research applications:

The simulation uses a simplified atomic model rather than the full protein context.
It relies on classical force field approximations.
The simulation time scale is limited.
Explicit solvent effects were not included in the model.
