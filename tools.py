import numpy as np

def get_atom_coords(filename):
    """
    Parse the .xyz file to extract atomic coordinates for each MD step.

    Args:
        filename (str): Path to the .xyz file.
    
    Returns:
        list of dict: A list of molecular data for each MD step, where each dict contains:
                      - "time" (float): The MD simulation time.
                      - "atoms" (list of tuple): List of (atom, [x, y, z]) for each atom.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    molecules = []
    i = 0
    while i < len(lines):
        # Number of atoms
        num_atoms = int(lines[i].strip())
        # MD time
        time_line = lines[i + 1].strip()
        time = float(time_line.split()[2])  # Extract time from "MD Time X.XX"
        # Atomic coordinates
        atoms = []
        for j in range(num_atoms):
            atom_line = lines[i + 2 + j].strip().split()
            atom = atom_line[0]
            coords = list(map(float, atom_line[1:]))
            atoms.append((atom, coords))
        # Store molecule data
        molecules.append({"time": time, "atoms": atoms})
        # Move to the next molecule block
        i += 2 + num_atoms
    
    return molecules

def format_for_pyscf(molecule):
    """
    Format the molecular data for PySCF.
    
    Args:
        molecule (dict): A single molecular data dict containing time and atoms.
    
    Returns:
        list: A list of strings formatted for PySCF.
    """
    return [(atom, coords) for atom, coords in molecule['atoms']]

def get_atom_list(molecule_data):
    """
    Get the list of atoms from the molecular data.

    Args:
        molecule_data (dict): A single molecular data dict containing time and atoms.
    
    Returns:
        list: A list of atom types.
    """
    return [atom for atom, _ in molecule_data]

def get_atom_position_list(molecule_data):
    """
    Get the list of atom positions from the molecular data.

    Args:
        molecule_data (dict): A single molecular data dict containing time and atoms.
    
    Returns:
        list: A list of atom positions.
    """
    return np.array([coords for _, coords in molecule_data])

