import numpy as np

# For testing this heres some parameters
array_size = 50
# Array spacing from simulations 0.057 for 3k? and 0.0858 for 2k. so those are essentially cm
spacing = 0.2 # meter

# Function for Microphone placements
def array_place(array_size, spacing):
    # Find midpoint of array
    zed = 1
    middle = array_size/2
    positions = np.array([[middle-((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed],[middle-((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed],[middle-((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed],[middle-((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed], 
                          [middle,middle+(4*spacing),zed],[middle,middle+(3*spacing),zed],[middle,middle+(2*spacing),zed],[middle,middle+spacing,zed], 
                          [middle+((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed],[middle+((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed],[middle+((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed],[middle+((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed]])
    return positions


array_place(array_size,spacing)