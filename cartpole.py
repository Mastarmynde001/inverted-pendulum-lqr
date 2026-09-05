import numpy as np


class CartPoleSystem:
    def __init__(self):
        # System constants
        self.M = 1.0       # Cart mass (kg)
        self.m = 0.1       # Pole mass (kg)
        self.l = 0.5       # Pole half-length to center of mass (m)
        self.b = 0.1       # Friction coefficient (N/m/s)
        self.g = 9.81      # Gravity (m/s^2)
        
        # Moment of inertia of a uniform rod about its center of mass
        # Full length is 2*l, I_cm = (1/12) * m * (2l)^2 = (1/3) * m * l^2
        self.I = (1.0 / 3.0) * self.m * (self.l ** 2)

    def _dynamics(self, state, u):
        """
        Calculates the derivatives of the state variables.
        state = [p, v, theta, omega]^T
        u = applied force
        """
        _p, v, theta, omega = state
        
        # Apply saturation limits to input force
        u = np.clip(u, -20.0, 20.0)
        
        # Denominator for the equations
        # D = (M + m)*(I + m*l^2) - m^2 * l^2 * cos^2(theta)
        I_term = self.I + self.m * self.l**2
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        D = (self.M + self.m) * I_term - (self.m * self.l * cos_theta)**2
        
        # Friction and driving forces
        force_term = u - self.b * v + self.m * self.l * (omega**2) * sin_theta
        gravity_term = self.m * self.g * self.l * sin_theta
        
        # Accelerations
        p_ddot = (I_term * force_term - self.m * self.l * cos_theta * gravity_term) / D
        theta_ddot = (-self.m * self.l * cos_theta * force_term + (self.M + self.m) * gravity_term) / D
        
        return np.array([v, p_ddot, omega, theta_ddot])

    def step_nonlinear(self, state, u, dt):
        """
        Performs one integration step using 4th-order Runge-Kutta (RK4).
        """
        # RK4 integration
        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + 0.5 * dt * k1, u)
        k3 = self._dynamics(state + 0.5 * dt * k2, u)
        k4 = self._dynamics(state + dt * k3, u)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize angle to [-pi, pi]
        # new_state[2] = (new_state[2] + np.pi) % (2 * np.pi) - np.pi
        return new_state

    def get_linearized_matrices(self):
        """
        Returns the linearized A and B matrices expanded around the upright equilibrium (theta = 0).
        state = [p, v, theta, omega]^T
        """
        I_term = self.I + self.m * self.l**2
        
        # Evaluate D at theta = 0
        D0 = (self.M + self.m) * I_term - (self.m * self.l)**2
        
        A = np.zeros((4, 4))
        # p_dot = v
        A[0, 1] = 1.0
        
        # v_dot equation
        A[1, 1] = -(I_term * self.b) / D0
        A[1, 2] = -((self.m**2) * (self.l**2) * self.g) / D0
        
        # theta_dot = omega
        A[2, 3] = 1.0
        
        # omega_dot equation
        A[3, 1] = (self.m * self.l * self.b) / D0
        A[3, 2] = ((self.M + self.m) * self.m * self.g * self.l) / D0
        
        B = np.zeros((4, 1))
        B[1, 0] = I_term / D0
        B[3, 0] = -(self.m * self.l) / D0
        
        return A, B
