import time

import matplotlib.pyplot as plt
import numpy as np
import pygame


class SimulationVisualizer:
    def __init__(self, screen_width=800, screen_height=400, enable_plot=True):
        self.width = screen_width
        self.height = screen_height
        self.enable_plot = enable_plot
        
        # Pygame Initialization
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Inverted Pendulum Digital Twin")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15)
        
        # Visual parameters
        self.pixels_per_meter = 100
        self.cart_w = 60
        self.cart_h = 40
        self.pole_l = 100  # pixels (corresponds to 2 * 0.5m = 1.0m)
        self.track_y = self.height // 2 + 50
        
        # History for plotting
        self.t_history = []
        self.theta_history = []
        self.u_history = []
        
        if self.enable_plot:
            # Matplotlib Initialization (Interactive mode)
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
            self.fig.canvas.manager.set_window_title('Control & State Tracking')
            
            # Setup Theta subplot
            self.ax1.set_ylabel('Theta (deg)')
            self.ax1.grid(True)
            self.line_theta, = self.ax1.plot([], [], 'r-')
            
            # Setup Control Force subplot
            self.ax2.set_ylabel('Force u (N)')
            self.ax2.set_xlabel('Time (s)')
            self.ax2.grid(True)
            self.line_u, = self.ax2.plot([], [], 'b-')
            
            self.plot_last_update = time.time()
            self.plot_update_interval = 0.05  # update plot every 50ms (20 FPS) to prevent lag

    def render(self, state, time_elapsed, force, disturbance_active=False):
        """
        Renders the PyGame window and updates the Matplotlib plot.
        
        Args:
            state: [position, velocity, theta, omega]
            time_elapsed: current simulation time
            force: applied control effort (u)
            disturbance_active: boolean flag if an impulse was just injected
        """
        # --- PyGame Rendering ---
        # Handle events to prevent window freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to stop simulation

        # Clear screen
        self.screen.fill((240, 240, 240))
        
        pos, vel, theta, omega = state
        
        # Draw track
        pygame.draw.line(self.screen, (0, 0, 0), (0, self.track_y), (self.width, self.track_y), 2)
        # Draw boundary markers (approx +/- 2.5 meters)
        bound_px = int(2.5 * self.pixels_per_meter)
        cx = self.width // 2
        pygame.draw.rect(self.screen, (200, 50, 50), (cx - bound_px, self.track_y, 10, 20))
        pygame.draw.rect(self.screen, (200, 50, 50), (cx + bound_px, self.track_y, 10, 20))

        # Calculate Cart position
        # screen center is pos=0
        cart_x = cx + int(pos * self.pixels_per_meter)
        
        # Draw Cart
        cart_rect = pygame.Rect(cart_x - self.cart_w // 2, self.track_y - self.cart_h // 2, self.cart_w, self.cart_h)
        pygame.draw.rect(self.screen, (50, 100, 200), cart_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect, 2)
        
        # Draw Pole
        # Theta=0 is straight up. PyGame y-axis is inverted (0 is top).
        # Math: dx = L * sin(theta), dy = -L * cos(theta)
        pole_end_x = cart_x + int(self.pole_l * np.sin(theta))
        pole_end_y = self.track_y - int(self.pole_l * np.cos(theta))
        
        pygame.draw.line(self.screen, (200, 150, 50), (cart_x, self.track_y), (pole_end_x, pole_end_y), 8)
        # Draw pivot joint
        pygame.draw.circle(self.screen, (50, 50, 50), (cart_x, self.track_y), 5)
        
        # Disturbance Indicator
        if disturbance_active:
            # Flash screen edges or draw a warning text
            pygame.draw.rect(self.screen, (255, 100, 100), (0, 0, self.width, self.height), 5)
            warn_surface = self.font.render("DISTURBANCE INJECTED!", True, (255, 0, 0))
            self.screen.blit(warn_surface, (self.width // 2 - 100, 50))
            
        # Draw HUD Overlay
        fps = self.clock.get_fps()
        hud_lines = [
            f"Time Elapsed: {time_elapsed:.2f} s",
            f"FPS:          {fps:.1f}",
            f"Position (x): {pos:.2f} m",
            f"Angle (deg):  {np.degrees(theta):.2f}",
            f"Control (u):  {force:.2f} N"
        ]
        
        for i, text in enumerate(hud_lines):
            surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, 10 + i * 20))
            
        pygame.display.flip()
        self.clock.tick(100) # Limit to 100 FPS for physics sync
        
        # --- Matplotlib Rendering ---
        if self.enable_plot:
            self.t_history.append(time_elapsed)
            self.theta_history.append(np.degrees(theta))
            self.u_history.append(force)
            
            # Keep history limited to the last 10 seconds to avoid memory bloat
            if time_elapsed - self.t_history[0] > 10.0:
                self.t_history.pop(0)
                self.theta_history.pop(0)
                self.u_history.pop(0)

            # Throttle plot updates
            current_time = time.time()
            if current_time - self.plot_last_update > self.plot_update_interval:
                self.line_theta.set_data(self.t_history, self.theta_history)
                self.line_u.set_data(self.t_history, self.u_history)
                
                # Dynamic axis scaling
                self.ax1.set_xlim(self.t_history[0], self.t_history[-1] + 0.1)
                self.ax2.set_xlim(self.t_history[0], self.t_history[-1] + 0.1)
                
                min_t, max_t = min(self.theta_history), max(self.theta_history)
                self.ax1.set_ylim(min_t - 5, max_t + 5)
                
                min_u, max_u = min(self.u_history), max(self.u_history)
                self.ax2.set_ylim(min_u - 5, max_u + 5)
                
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                self.plot_last_update = current_time

        return True

    def close(self):
        pygame.quit()
        if self.enable_plot:
            plt.close('all')
