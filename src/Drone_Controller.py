# drone_controller.py

import airsim
import numpy as np

class DroneController:
    def __init__(self, client, Kp=0.01, target_altitude=-6.0):
        
        #Initialize the drone controller
        #Kp: Proportional control gain for image error.
        #target_altitude: Desired flight altitude.

        self.client = client
        self.Kp = Kp
        self.target_altitude = target_altitude

    def update(self, wolf_pos, image_shape):
        image_h, image_w = image_shape[:2]
        image_center_x = image_w / 2
        image_center_y = image_h / 2

        wolf_x, wolf_y = wolf_pos

        error_x = wolf_x - image_center_x
        error_y = wolf_y - image_center_y

        vx = -self.Kp * error_y
        vy =  self.Kp * error_x

        # Altitude correction
        state = self.client.getMultirotorState()
        current_altitude = state.kinematics_estimated.position.z_val  # Z (down is positive)
        altitude_error = current_altitude - self.target_altitude

        # Simple proportional controller for Z
        Kp_z = 0.5
        vz = -Kp_z * altitude_error

        # Clip vertical speed
        vz = np.clip(vz, -1.0, 1.0)

        self.client.moveByVelocityAsync(
            vx, vy, vz, duration=0.2,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
        )

    def maintain_altitude(self):
        # Used in debugging
        self.client.moveToZAsync(self.target_altitude, 1)
