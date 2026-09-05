import numpy as np

from cartpole import CartPoleSystem
from lqr_controller import LQRController


def test_matrix_dimensions():
    system = CartPoleSystem()
    A, B = system.get_linearized_matrices()
    
    assert A.shape == (4, 4), f"Expected A shape (4, 4), got {A.shape}"
    assert B.shape == (4, 1), f"Expected B shape (4, 1), got {B.shape}"
    
    lqr = LQRController(A, B)
    assert lqr.K.shape == (1, 4), f"Expected K shape (1, 4), got {lqr.K.shape}"

def test_lqr_convergence():
    system = CartPoleSystem()
    A, B = system.get_linearized_matrices()
    lqr = LQRController(A, B)
    
    # Initial state: slightly perturbed pole
    state = np.array([0.0, 0.0, 0.2, 0.0])
    dt = 0.02
    max_time = 3.0
    steps = int(max_time / dt)
    
    for _ in range(steps):
        u = lqr.compute_force(state)
        state = system.step_nonlinear(state, u, dt)
        
    # Check if |theta| < 0.01 rad within 3.0 seconds
    theta = state[2]
    assert abs(theta) < 0.01, f"Expected |theta| < 0.01 rad, got {abs(theta):.4f} rad"

def test_actuator_saturation():
    system = CartPoleSystem()
    state = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Input heavily exceeding limits
    state_dot_excess_positive = system._dynamics(state, 1000.0)
    state_dot_max_positive = system._dynamics(state, 20.0)
    
    state_dot_excess_negative = system._dynamics(state, -1000.0)
    state_dot_max_negative = system._dynamics(state, -20.0)
    
    # Accelerations must perfectly match due to internal np.clip clipping u to +/- 20
    np.testing.assert_array_almost_equal(
        state_dot_excess_positive, 
        state_dot_max_positive, 
        err_msg="Saturation failed for positive u"
    )
    
    np.testing.assert_array_almost_equal(
        state_dot_excess_negative, 
        state_dot_max_negative, 
        err_msg="Saturation failed for negative u"
    )
