# Inverted Pendulum Digital Twin

A high-performance Python digital twin for an Inverted Pendulum (Cart-Pole) system. This simulation features a continuous-time non-linear physics engine built with 4th-order Runge-Kutta (RK4) integration, governed by a robust Linear Quadratic Regulator (LQR) or a benchmark PID controller. The engine pushes live telemetry to a non-blocking `pygame` and `matplotlib` GUI at 100 Hz.

## System Architecture

The software is modularized into four primary components:
1. **Physics Engine** (`cartpole.py`): Models the non-linear rigid body dynamics and derives the state-space linearization Jacobians ($A$, $B$).
2. **Controllers** (`lqr_controller.py`): Solves the Continuous Algebraic Riccati Equation (CARE) to derive the optimal LQR gain matrix $K$. Also contains a PID fallback and a randomized environmental disturbance injector.
3. **Rendering Engine** (`visualizer.py`): Orchestrates a `pygame` interactive viewport and a `matplotlib` time-series telemetry tracker running seamlessly without blocking the physics loop.
4. **Integration Layer** (`main.py`): The CLI executable binding the loop together at a locked 100 Hz timestep ($dt=0.01$).

## Mathematical Formulation

The physical state vector is defined as $x = [p, v, \theta, \omega]^T$ (Position, Velocity, Angle, Angular Velocity).

### State-Space Matrices (Linearization around $\theta = 0$)
The continuous linear model $\dot{x} = Ax + Bu$ utilizes the following Jacobians:

```math
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & -\frac{(I + m l^2)b}{D_0} & -\frac{m^2 l^2 g}{D_0} & 0 \\
0 & 0 & 0 & 1 \\
0 & \frac{m l b}{D_0} & \frac{(M+m)m g l}{D_0} & 0
\end{bmatrix}, \quad
B = \begin{bmatrix}
0 \\
\frac{I + m l^2}{D_0} \\
0 \\
-\frac{m l}{D_0}
\end{bmatrix}
```
Where $D_0 = (M+m)(I + m l^2) - (m l)^2$.

### Linear Quadratic Regulator (LQR)
The LQR controller calculates a feedback gain $K$ to minimize the infinite-horizon quadratic cost function $J$:

```math
J = \int_{0}^{\infty} \left( x^T Q x + u^T R u \right) dt
```

We utilize a heavy penalty on pole deviation ($Q[2,2] = 100.0$) and a lenient control effort penalty ($R = [0.01]$).

## LQR vs. PID Comparison

| Feature | LQR (Linear Quadratic Regulator) | PID (Proportional-Integral-Derivative) |
| :--- | :--- | :--- |
| **Control Approach** | Optimal state-space multivariable control (MIMO) | Classic heuristic error-feedback control (SISO) |
| **State Knowledge** | Requires full state vector $x$ observation | Only requires observation of target error $\theta$ |
| **Tuning Complexity** | Solved mathematically via CARE based on $Q, R$ weights | Empirically tuned via $K_p, K_i, K_d$ heuristics |
| **Disturbance Rejection** | Excellent (rapidly snaps back to equilibrium) | Moderate (prone to integral windup or overshooting) |
| **Energy Efficiency** | High (minimizes control effort cost $R$) | Low (pushes aggressively based solely on instantaneous error) |

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mastarmynde001/inverted-pendulum-lqr.git
   cd inverted-pendulum-lqr
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Simulation**:
   ```bash
   python main.py --mode lqr --disturbance
   ```
   
### CLI Options
- `--mode {lqr, pid}`: Switch between the LQR or PID controllers.
- `--disturbance`: Enable random physical impulse bumps (wind gusts).
- `--dt`: Set the precision of the RK4 integrator (default `0.01`s).

## CI/CD
This repository runs automated GitHub Actions testing via `pytest` and code linting via `ruff` on every push to the `main` branch.
