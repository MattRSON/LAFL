import numpy as np

def unwrap_phase(phase):
    """
    Unwrap phase values to extend them beyond a 2π range.
    
    Args:
    - phase: numpy array of phase values
    
    Returns:
    - unwrapped_phase: numpy array of unwrapped phase values
    """
    unwrapped_phase = np.unwrap(phase)
    return unwrapped_phase

# Example usage
if __name__ == "__main__":
    # Generate example phase values with phase wraps
    phase_wrapped = np.array([0.1, 6.2, 5.9, 10, 2.7, 0.5])

    # Unwrap phase values
    phase_unwrapped = unwrap_phase(phase_wrapped)

    print("Wrapped phase:", phase_wrapped)
    print("Unwrapped phase:", phase_unwrapped)