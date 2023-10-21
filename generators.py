import numpy as np
import random

def two_torus_observable_func(state, R=1/2):
    theta1 = state[0]
    theta2 = state[1]
    return np.array([
        (1 + R * np.cos(theta2)) * np.cos(theta1),
        (1 + R * np.cos(theta2)) * np.sin(theta1),
        R * np.sin(theta2)
    ])

def two_torus_evolution_function(state, alpha=np.sqrt(3), step=2*np.pi/100):
    return [
        angle_evolution(state[0], alpha=1, step=step),
        angle_evolution(state[1], alpha=alpha, step=step),
    ]

def full_state_observable(state):
    return state

def unit_circle_observable_function(state):
    return np.exp(1j * state)

def angle_evolution(state, alpha=np.sqrt(3)):
    return (state + alpha) % (2*np.pi)

def n_torus_initial_points(n=10):
    frequencies = np.array([random.uniform(0.1, 10) for _ in range(n)])
    initial_points = np.array([random.uniform(0, 1) for _ in range(n)])
    return frequencies, initial_points

def n_torus_evolution_function(current_points, frequencies):
    next_points = (current_points + frequencies) % 1
    return next_points

def flat_torus_observable_function(state):
    return np.exp(1j * state * 2*np.pi)

def twoD_state_to_complex(state):
    assert len(state) == 2
    return [state[0] + 1j*state[1]]
