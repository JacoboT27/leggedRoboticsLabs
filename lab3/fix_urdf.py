import xml.etree.ElementTree as ET

def scale_urdf_mass(input_file, output_file, target_total_mass=5.5):
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # 1. Calculate current total mass
    current_mass = 0.0
    mass_elements = []
    
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is not None:
            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                m = float(mass_elem.get('value'))
                current_mass += m
                mass_elements.append((inertial, m))
    
    if current_mass == 0:
        print("Error: Could not find any mass elements in URDF.")
        return

    scale_factor = target_total_mass / current_mass
    print(f"Current Mass: {current_mass:.4f} kg")
    print(f"Target Mass:  {target_total_mass:.4f} kg")
    print(f"Scale Factor: {scale_factor:.6f}")

    # 2. Apply scaling to Mass and Inertia
    for inertial, original_mass in mass_elements:
        # Scale Mass
        mass_elem = inertial.find('mass')
        new_mass = original_mass * scale_factor
        mass_elem.set('value', str(new_mass))
        
        # Scale Inertia (Inertia scales linearly with mass if geometry is constant)
        inertia_elem = inertial.find('inertia')
        if inertia_elem is not None:
            for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                if inertia_elem.get(attr):
                    val = float(inertia_elem.get(attr))
                    inertia_elem.set(attr, str(val * scale_factor))

    tree.write(output_file)
    print(f"Successfully saved fixed model to: {output_file}")

if __name__ == "__main__":
    scale_urdf_mass("lab3/urdf/nao.urdf", "lab3/urdf/nao_fixed.urdf")