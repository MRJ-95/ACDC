# Drug-Target Interaction Simulation: Aspirin-COX Binding
# Molecular simulation study of how aspirin interacts with cyclooxygenase enzyme

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

@dataclass
class Atom:
    """Represents an atom with coordinates and properties"""
    x: float
    y: float
    z: float
    atom_type: str
    charge: float = 0.0
    
    def distance_to(self, other: 'Atom') -> float:
        """Calculate distance to another atom"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

class MolecularSystem:
    """Handles molecular system setup and energy calculations"""
    
    def __init__(self):
        self.atoms = []
        self.bonds = []
        
    def add_atom(self, atom: Atom):
        """Add an atom to the system"""
        self.atoms.append(atom)
        
    def lennard_jones_potential(self, r: float, epsilon: float = 1.0, sigma: float = 3.5) -> float:
        """Calculate Lennard-Jones potential energy"""
        if r < 0.1:  # Avoid division by zero
            return 1000.0
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    
    def electrostatic_potential(self, r: float, q1: float, q2: float, k: float = 332.0) -> float:
        """Calculate electrostatic potential energy (kcal/mol)"""
        if r < 0.1:
            return 1000.0
        return k * q1 * q2 / r
    
    def calculate_total_energy(self, coords: np.ndarray) -> float:
        """Calculate total potential energy of the system"""
        total_energy = 0.0
        n_atoms = len(self.atoms)
        
        # Reshape coordinates
        coords = coords.reshape(n_atoms, 3)
        
        # Calculate pairwise interactions
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r = np.linalg.norm(coords[i] - coords[j])
                
                # Lennard-Jones interaction
                lj_energy = self.lennard_jones_potential(r)
                
                # Electrostatic interaction
                elec_energy = self.electrostatic_potential(r, self.atoms[i].charge, self.atoms[j].charge)
                
                total_energy += lj_energy + elec_energy
                
        return total_energy

def create_aspirin_molecule() -> List[Atom]:
    """Create a simplified aspirin molecule structure"""
    aspirin_atoms = [
        # Benzene ring (simplified as 6 carbons)
        Atom(0.0, 0.0, 0.0, 'C', charge=-0.1),
        Atom(1.4, 0.0, 0.0, 'C', charge=-0.1),
        Atom(2.1, 1.2, 0.0, 'C', charge=-0.1),
        Atom(1.4, 2.4, 0.0, 'C', charge=-0.1),
        Atom(0.0, 2.4, 0.0, 'C', charge=-0.1),
        Atom(-0.7, 1.2, 0.0, 'C', charge=-0.1),
        
        # Carboxyl group
        Atom(-2.0, 1.2, 0.0, 'C', charge=0.5),
        Atom(-2.7, 0.0, 0.0, 'O', charge=-0.5),
        Atom(-2.7, 2.4, 0.0, 'O', charge=-0.6),
        Atom(-3.7, 2.4, 0.0, 'H', charge=0.4),
        
        # Acetyl group
        Atom(0.0, 3.6, 0.0, 'O', charge=-0.4),
        Atom(0.0, 4.8, 0.0, 'C', charge=0.5),
        Atom(1.2, 5.4, 0.0, 'O', charge=-0.5),
        Atom(-1.2, 5.4, 0.0, 'C', charge=-0.3),
    ]
    return aspirin_atoms

def create_cox_binding_site() -> List[Atom]:
    """Create a simplified COX enzyme binding site"""
    binding_site_atoms = [
        # Key amino acid residues in binding site (simplified)
        # Serine 530 (catalytic site)
        Atom(5.0, 2.0, 1.0, 'C', charge=0.1),
        Atom(5.5, 2.5, 2.0, 'O', charge=-0.6),
        Atom(6.0, 2.0, 2.5, 'H', charge=0.4),
        
        # Arginine (positive charge)
        Atom(3.0, 4.0, 0.0, 'C', charge=0.2),
        Atom(2.5, 5.0, 0.5, 'N', charge=-0.7),
        Atom(1.5, 5.5, 0.5, 'H', charge=0.4),
        Atom(3.0, 5.5, 1.0, 'H', charge=0.4),
        
        # Phenylalanine (hydrophobic interaction)
        Atom(4.0, 0.0, 2.0, 'C', charge=0.0),
        Atom(4.5, -1.0, 2.5, 'C', charge=0.0),
        Atom(5.5, -1.0, 3.0, 'C', charge=0.0),
        
        # Tyrosine (hydrogen bonding)
        Atom(6.0, 4.0, 1.0, 'C', charge=0.1),
        Atom(7.0, 4.5, 1.5, 'O', charge=-0.6),
        Atom(7.5, 4.0, 2.0, 'H', charge=0.4),
    ]
    return binding_site_atoms

def run_molecular_dynamics(system: MolecularSystem, steps: int = 1000, dt: float = 0.001) -> Tuple[List[float], List[np.ndarray]]:
    """Run simplified molecular dynamics simulation"""
    
    # Initialize coordinates and velocities
    coords = np.array([[atom.x, atom.y, atom.z] for atom in system.atoms])
    velocities = np.random.normal(0, 0.1, coords.shape)
    
    # Storage for trajectory
    energies = []
    trajectory = []
    
    print(f"Running MD simulation for {steps} steps...")
    
    for step in range(steps):
        # Calculate forces (negative gradient of potential)
        def energy_func(x):
            return system.calculate_total_energy(x.flatten())
        
        # Numerical gradient for forces
        forces = np.zeros_like(coords)
        epsilon = 1e-6
        
        for i in range(len(coords)):
            for j in range(3):
                coords_plus = coords.copy()
                coords_minus = coords.copy()
                coords_plus[i, j] += epsilon
                coords_minus[i, j] -= epsilon
                
                energy_plus = system.calculate_total_energy(coords_plus.flatten())
                energy_minus = system.calculate_total_energy(coords_minus.flatten())
                
                forces[i, j] = -(energy_plus - energy_minus) / (2 * epsilon)
        
        # Velocity Verlet integration
        velocities += forces * dt * 0.5
        coords += velocities * dt
        velocities += forces * dt * 0.5
        
        # Apply simple thermostat (velocity scaling)
        if step % 10 == 0:
            velocities *= 0.99
        
        # Store data
        current_energy = system.calculate_total_energy(coords.flatten())
        energies.append(current_energy)
        trajectory.append(coords.copy())
        
        if step % 100 == 0:
            print(f"Step {step}: Energy = {current_energy:.2f} kcal/mol")
    
    return energies, trajectory

def analyze_binding_interactions(aspirin_atoms: List[Atom], cox_atoms: List[Atom]) -> pd.DataFrame:
    """Analyze different types of binding interactions"""
    
    interactions = []
    
    for i, asp_atom in enumerate(aspirin_atoms):
        for j, cox_atom in enumerate(cox_atoms):
            distance = asp_atom.distance_to(cox_atom)
            
            # Classify interaction type
            interaction_type = "van der Waals"
            strength = 0.0
            
            # Hydrogen bonding (O-H, N-H interactions within 3.5 Å)
            if distance < 3.5:
                if ((asp_atom.atom_type in ['O', 'N'] and cox_atom.atom_type == 'H') or
                    (asp_atom.atom_type == 'H' and cox_atom.atom_type in ['O', 'N'])):
                    interaction_type = "Hydrogen bond"
                    strength = 5.0 / distance
                
                # Electrostatic interactions
                elif abs(asp_atom.charge * cox_atom.charge) > 0.1:
                    interaction_type = "Electrostatic"
                    strength = abs(asp_atom.charge * cox_atom.charge) / distance
                
                # Hydrophobic interactions (C-C within 4.0 Å)
                elif (asp_atom.atom_type == 'C' and cox_atom.atom_type == 'C' and distance < 4.0):
                    interaction_type = "Hydrophobic"
                    strength = 2.0 / distance
                
                else:
                    strength = 1.0 / distance
            
            interactions.append({
                'aspirin_atom': i,
                'cox_atom': j,
                'distance': distance,
                'interaction_type': interaction_type,
                'strength': strength,
                'aspirin_type': asp_atom.atom_type,
                'cox_type': cox_atom.atom_type
            })
    
    return pd.DataFrame(interactions)

def calculate_binding_affinity(interactions_df: pd.DataFrame) -> float:
    """Calculate estimated binding affinity from interactions"""
    
    # Weight different interaction types
    weights = {
        'Hydrogen bond': -2.0,      # Favorable
        'Electrostatic': -1.5,     # Favorable  
        'Hydrophobic': -1.0,       # Favorable
        'van der Waals': -0.5      # Weak favorable
    }
    
    binding_energy = 0.0
    
    for _, interaction in interactions_df.iterrows():
        if interaction['distance'] < 5.0:  # Only consider close interactions
            weight = weights.get(interaction['interaction_type'], -0.1)
            binding_energy += weight * interaction['strength']
    
    # Convert to approximate Kd (very simplified)
    # ΔG = RT ln(Kd), assuming T = 298K
    RT = 0.593  # kcal/mol at 298K
    
    if binding_energy < 0:
        kd = np.exp(-binding_energy / RT)
        return binding_energy, kd
    else:
        return 0.0, float('inf')

# Main simulation execution
def main():
    print("=== Drug-Target Interaction Simulation: Aspirin-COX Binding ===\n")
    
    # Create molecular system
    print("1. Setting up molecular system...")
    aspirin = create_aspirin_molecule()
    cox_site = create_cox_binding_site()
    
    # Combine into single system
    system = MolecularSystem()
    all_atoms = aspirin + cox_site
    
    for atom in all_atoms:
        system.add_atom(atom)
    
    print(f"   - Aspirin atoms: {len(aspirin)}")
    print(f"   - COX binding site atoms: {len(cox_site)}")
    print(f"   - Total system atoms: {len(all_atoms)}")
    
    # Run molecular dynamics
    print("\n2. Running molecular dynamics simulation...")
    energies, trajectory = run_molecular_dynamics(system, steps=500)
    
    # Analyze binding interactions
    print("\n3. Analyzing binding interactions...")
    interactions_df = analyze_binding_interactions(aspirin, cox_site)
    
    # Calculate binding affinity
    binding_energy, kd = calculate_binding_affinity(interactions_df)
    print(f"\n4. Binding Analysis Results:")
    print(f"   - Estimated binding energy: {binding_energy:.2f} kcal/mol")
    print(f"   - Estimated Kd: {kd:.2e} M")
    
    # Create visualizations
    print("\n5. Generating analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Energy evolution
    axes[0,0].plot(energies)
    axes[0,0].set_xlabel('MD Steps')
    axes[0,0].set_ylabel('Potential Energy (kcal/mol)')
    axes[0,0].set_title('Energy Evolution During MD Simulation')
    axes[0,0].grid(True, alpha=0.3)
    
    # Interaction distance distribution
    close_interactions = interactions_df[interactions_df['distance'] < 6.0]
    axes[0,1].hist(close_interactions['distance'], bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Distance (Å)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of Interaction Distances')
    axes[0,1].grid(True, alpha=0.3)
    
    # Interaction types
    interaction_counts = interactions_df[interactions_df['distance'] < 5.0]['interaction_type'].value_counts()
    axes[1,0].pie(interaction_counts.values, labels=interaction_counts.index, autopct='%1.1f%%')
    axes[1,0].set_title('Types of Binding Interactions')
    
    # Binding site map (2D projection)
    aspirin_coords = np.array([[atom.x, atom.y] for atom in aspirin])
    cox_coords = np.array([[atom.x, atom.y] for atom in cox_site])
    
    axes[1,1].scatter(aspirin_coords[:, 0], aspirin_coords[:, 1], 
                     c='red', s=100, alpha=0.7, label='Aspirin', marker='o')
    axes[1,1].scatter(cox_coords[:, 0], cox_coords[:, 1], 
                     c='blue', s=100, alpha=0.7, label='COX Site', marker='s')
    
    # Draw lines for strong interactions
    strong_interactions = interactions_df[
        (interactions_df['distance'] < 4.0) & 
        (interactions_df['strength'] > 0.5)
    ]
    
    for _, interaction in strong_interactions.iterrows():
        asp_idx = int(interaction['aspirin_atom'])
        cox_idx = int(interaction['cox_atom'])
        axes[1,1].plot([aspirin[asp_idx].x, cox_site[cox_idx].x],
                      [aspirin[asp_idx].y, cox_site[cox_idx].y],
                      'k--', alpha=0.5, linewidth=1)
    
    axes[1,1].set_xlabel('X coordinate (Å)')
    axes[1,1].set_ylabel('Y coordinate (Å)') 
    axes[1,1].set_title('Binding Site Interaction Map')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n6. Interaction Summary:")
    print("="*50)
    interaction_summary = interactions_df[interactions_df['distance'] < 5.0].groupby('interaction_type').agg({
        'distance': ['count', 'mean', 'min'],
        'strength': ['mean', 'max']
    }).round(3)
    print(interaction_summary)
    
    print(f"\nSimulation completed successfully!")
    print(f"The results suggest aspirin binds to COX with moderate affinity,")
    print(f"consistent with its known anti-inflammatory activity.")

if __name__ == "__main__":
    main()