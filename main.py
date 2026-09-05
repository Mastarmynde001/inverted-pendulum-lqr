import argparse
import sys

import numpy as np

from cartpole import CartPoleSystem
from lqr_controller import LQRController, PIDController, apply_disturbance
from visualizer import SimulationVisualizer


def main():
    parser = argparse.ArgumentParser(description="Inverted Pendulum Digital Twin Simulation")
    parser.add_argument("--mode", type=str, choices=["lqr", "pid"], default="lqr", 
                        help="Control mode: 'lqr' or 'pid' (default: lqr)")
    parser.add_argument("--disturbance", action="store_true", 
                        help="Enable random impulse disturbances")
    parser.add_argument("--dt", type=float, default=0.01, 
                        help="Physics integration time step (default: 0.01s)")
    args = parser.parse_args()

    print(f"Starting Inverted Pendulum Digital Twin in {args.mode.upper()} mode...")
    
    # 1. Initialize Physics
    system = CartPoleSystem()
    state = np.array([0.0, 0.0, 0.15, 0.0]) # Initial tilt ~8.5 degrees
    
    # 2. Initialize Controllers
    if args.mode == "lqr":
        A, B = system.get_linearized_matrices()
        controller = LQRController(A, B)
    else:
        # Standard PID tuned roughly for inverted pendulum benchmark
        controller = PIDController(kp=150.0, ki=10.0, kd=30.0)

    # 3. Initialize Visualizer
    viz = SimulationVisualizer(enable_plot=True)
    
    time_elapsed = 0.0
    running = True

    try:
        while running:
            # Control Input calculation
            if isinstance(controller, LQRController):
                u = controller.compute_force(state)
            else:
                u = controller.compute_force(state, args.dt)

            # Apply random disturbance if enabled
            disturbance_active = False
            if args.disturbance:
                # Store old state to check if disturbance modified it
                old_state = np.copy(state)
                state = apply_disturbance(state, magnitude=2.0, probability=0.01)
                # Check if an impulse actually hit this frame
                if not np.array_equal(old_state, state):
                    disturbance_active = True

            # Advance Physics Step
            state = system.step_nonlinear(state, u, args.dt)
            time_elapsed += args.dt

            # Push state to PyGame
            # render() manages internal 100 Hz limiting via pygame.time.Clock
            running = viz.render(state, time_elapsed, u, disturbance_active)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C).")
    finally:
        print("Shutting down cleanly...")
        viz.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
