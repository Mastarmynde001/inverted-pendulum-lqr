import random

import numpy as np
import scipy.linalg


class LQRController:
    """
    Linear Quadratic Regulator (LQR) for the CartPole System.
    """
    def __init__(self, A: np.ndarray, B: np.ndarray):
        """
        Initializes the LQR controller by solving the Continuous Algebraic Riccati Equation (CARE).
        
        Args:
            A (np.ndarray): Linearized state matrix (4x4)
            B (np.ndarray): Linearized input matrix (4x1)
        """
        # Define cost matrices
        # Heavy penalty on pole deviation (theta) [index 2]
        self.Q = np.diag([10.0, 1.0, 100.0, 10.0])
        # Control effort penalty
        self.R = np.array([[0.01]])
        
        # Solve the Continuous Algebraic Riccati Equation:
        # A^T * P + P * A - P * B * R^-1 * B^T * P + Q = 0
        self.P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
        
        # Compute the optimal gain matrix K = R^-1 * B^T * P
        self.K = np.linalg.inv(self.R) @ B.T @ self.P

    def compute_force(self, x: np.ndarray) -> float:
        """
        Computes the optimal control effort (force) to apply to the cart.
        
        Args:
            x (np.ndarray): Current state vector [position, velocity, angle, angular_velocity]
            
        Returns:
            float: The control effort u = -K * x
        """
        # u = -K * x. We extract the scalar value from the 1x1 resulting array.
        u = -self.K @ x
        return float(u[0])

class PIDController:
    """
    Proportional-Integral-Derivative (PID) Controller acting as a benchmark 
    to stabilize the inverted pendulum. Targets theta = 0.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initializes the PID controller with given gains.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0

    def compute_force(self, x: np.ndarray, dt: float) -> float:
        """
        Computes the PID control effort to stabilize the pole angle.
        
        Args:
            x (np.ndarray): Current state vector [position, velocity, angle, angular_velocity]
            dt (float): Time step duration for the integral term calculation
            
        Returns:
            float: The control effort.
        """
        # We want theta to be 0. Error is (target - current) = 0 - theta = -theta
        theta = x[2]
        omega = x[3] # Derivative of theta is omega
        
        error = -theta
        derivative_error = -omega
        
        self.integral_error += error * dt
        
        # u = Kp * e + Ki * integral(e) + Kd * de/dt
        u = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error
        
        # Note: Depending on motor/system convention, you might need to reverse the sign.
        # If the cart moves in +x direction to catch a +theta lean, the force should be positive.
        # The benchmark PID typically needs to be tuned accordingly.
        return float(u)

def apply_disturbance(x: np.ndarray, magnitude: float, probability: float) -> np.ndarray:
    """
    Simulates random impulse forces (e.g., wind or physical bumps) by directly
    perturbing the velocities in the state vector.
    
    Args:
        x (np.ndarray): Current state vector [position, velocity, angle, angular_velocity]
        magnitude (float): The maximum magnitude of the impulse to apply
        probability (float): The probability (0.0 to 1.0) of a disturbance occurring
        
    Returns:
        np.ndarray: The modified state vector
    """
    x_new = np.copy(x)
    if random.random() < probability:
        # Randomly choose between a cart bump (velocity) or a wind gust (angular velocity)
        if random.random() < 0.5:
            # Perturb cart velocity
            x_new[1] += random.uniform(-magnitude, magnitude)
        else:
            # Perturb pole angular velocity
            x_new[3] += random.uniform(-magnitude, magnitude)
            
    return x_new
