import math
import numpy as np

# Constants for Earth radius in meters
EARTH_RADIUS = 6378137

# Function to convert meters to latitude change
def meters_to_latitude(meters):
    return (meters / EARTH_RADIUS) * (180 / math.pi)

# Function to convert meters to longitude change
def meters_to_longitude(meters, latitude):
    return (meters / (EARTH_RADIUS * math.cos(math.radians(latitude)))) * (180 / math.pi)

class SixDOFSim():

    def __init__(self, lat, long, alt, psi, speed, vert_speed, dt) -> None:
        self.lat = lat
        self.long = long
        self.alt = alt
        self.track = psi
        self.speed = speed
        self.vert_speed = vert_speed
        self.dt = dt

    # Aircraft state update function
    def update_flight_state(self, lat, long, alt, psi, speed, vert_speed, dt):
        # Ensure vertical speed does not exceed total speed
        if abs(vert_speed) > speed:
            vert_speed = (0 if vert_speed == 0 else math.copysign(speed, vert_speed))

        # Calculate horizontal speed as the remaining component of the total speed
        horizontal_speed = math.sqrt(max(speed**2 - vert_speed**2, 0))
            
        # Calculate change in position (latitude and longitude)
        delta_lat = meters_to_latitude(horizontal_speed * dt * math.cos(math.radians(psi)))
        delta_long = meters_to_longitude(horizontal_speed * dt * math.sin(math.radians(psi)), lat)
        
        # Update altitude based on vertical speed
        delta_alt = vert_speed * dt
        
        # Update the aircraft's position
        new_lat = lat + delta_lat
        new_long = long + delta_long
        new_alt = alt + delta_alt
        
        return new_lat, new_long, new_alt

    # Autopilot to adjust towards target heading, speed, vertical speed, and altitude
    def autopilot(self, psi, speed, alt, vert_speed, target_psi, target_speed, target_vert_speed, target_alt, dt):
        # Control parameters
        heading_adjustment_rate = 7.55      # deg/s
        speed_adjustment_rate = 2.0         # m/s^2
        altitude_smooth_radius = 10.0      # Altitude smoothing within 10 meters of target

        # === Adjust heading towards target with wraparound handling ===
        heading_error = (target_psi - psi + 540) % 360 - 180  # Now in range [-180, 180]
        if heading_error != 0:
            psi += heading_adjustment_rate * dt * math.copysign(1, heading_error)
        psi = psi % 360  # Ensure psi stays in [0, 360)
        
        # Adjust speed towards target
        speed_error = target_speed - speed
        speed += speed_adjustment_rate * dt * (0 if speed_error == 0 else math.copysign(1, speed_error))

        # Altitude control logic with target vertical speed
        altitude_error = target_alt - alt

        # If we're far from the target altitude, use target vertical speed
        if abs(altitude_error) > altitude_smooth_radius:
            vert_speed = (0 if altitude_error == 0 else math.copysign(target_vert_speed, altitude_error))
        else:
            # If we're close, smoothly reduce vertical speed to capture the altitude
            vert_speed = (altitude_error / altitude_smooth_radius) * target_vert_speed
        
        # Clip dynamics
        vert_speed = np.clip(vert_speed, -500 / 196.9, 500 / 196.9)
        speed = np.clip(speed, 60 / 1.944, 138 / 1.944)

        return psi, speed, vert_speed

    def update_targets(self, target_psi, target_speed, target_vert_speed, target_alt):
        self.target_psi = target_psi
        self.target_speed = target_speed
        self.target_vert_speed = target_vert_speed
        self.target_alt = target_alt

    def simulate(self):
        # Adjust controls towards target heading, speed, vertical speed, and altitude
        self.track, self.speed, self.vert_speed = self.autopilot(self.track, self.speed, self.alt, self.vert_speed, self.target_psi, self.target_speed, self.target_vert_speed, self.target_alt, self.dt)
        
        # Update position based on current speed and orientation
        self.lat, self.long, self.alt = self.update_flight_state(self.lat, self.long, self.alt, self.track, self.speed, self.vert_speed, self.dt)
        
        return self.lat, self.long, self.alt, self.track, self.speed, self.vert_speed