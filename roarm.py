import requests
import json
from tracker import MobileIOARTracker


class RoArm:
    def __init__(self, ip_address='10.0.0.112'):
        """
        Initialize RoArm controller with robot's IP address.

        Args:
            ip_address (str): IP address of the robot arm
        """
        self.base_url = f'http://{ip_address}/js'
        self._last_response = None

    def _send_command(self, payload):
        """
        Send a command to the robot arm and return the response.

        Args:
            payload (dict): Command payload to send

        Returns:
            dict: Response from the robot arm
        """
        try:
            print(f"sending {payload}")
            response = requests.get(self.base_url, params={'json': json.dumps(payload)})
            response.raise_for_status()
            self._last_response = response.json()
            return self._last_response
        except requests.RequestException as e:
            print(f"Error sending command: {e}")
            return None

    def set_led(self, brightness=255):
        """
        Set the LED brightness.

        Args:
            brightness (int): LED brightness (0-255)
        """
        return self._send_command({"T": 114, "led": brightness})

    def get_position(self):
        """
        Retrieve current arm position.

        Returns:
            dict: Current position and torque information
        """
        return self._send_command({"T": 105})

    def move_to_position(self, x, y, z, t, spd=.1, **kwargs):
        """
        Move to a specific end-effector position.

        Args:
            x (float, required): X coordinate
            y (float, required): Y coordinate
            z (float, required): Z coordinate
            t (float, required): Gripper position
            spd (float, optional): Movement speed
        """
        # Construct payload with only provided coordinates
        payload = {"T": 104}
        for coord, value in [('x', x), ('y', y), ('z', z), ('t', t), ('spd', spd)]:
            if value is not None:
                payload[coord] = value

        return self._send_command(payload)

    def home(self):
        """
        Return the arm to its home position.
        """
        return self.move_to_position(x=300, y=0, z=226, t=3.13)



def main():
    try:
        arm = RoArm()
        arm.set_led(180)
        initial_pos = arm.get_position()
        print("Arm Start Position:", initial_pos)
        arm_pos = initial_pos.copy()

        def position_update(pos):
            print(f"Relative Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            scale_factor = 800
            arm_pos['x'] = initial_pos['x'] - pos[0] * scale_factor
            arm_pos['y'] = initial_pos['y'] - pos[1] * scale_factor
            arm_pos['z'] = initial_pos['z'] + pos[2] * scale_factor
            arm_pos['spd'] = 2
            arm.move_to_position(**arm_pos)

        tracker = MobileIOARTracker()
        tracker.continuous_tracking(callback=position_update, duration=60)
        arm.set_led(0)

    except Exception as e:
        print(f"Error controlling robot arm: {e}")


if __name__ == "__main__":
    main()
