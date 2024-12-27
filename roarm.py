import requests
import json
import time
import serial
import threading
import queue
from tracker import MobileIOARTracker


class RoArm:
    def __init__(self, ip_address='10.0.0.112', serial_port='COM8', baudrate=115200, use_serial=True):
        """
        Initialize RoArm controller.

        Args:
            ip_address (str): IP address for web API.
            serial_port (str): COM port for serial communication (e.g., "COM7").
            baudrate (int): Baud rate for serial communication.
        """
        self.use_serial = use_serial
        self.response_queue = queue.Queue()
        self.queue_next_response = False
        self.serial_response_acked = False
        self.send_lock = threading.Lock()
        if self.use_serial:
            try:
                self.serial = serial.Serial(serial_port, baudrate=baudrate, timeout=1, dsrdtr=None)
                self.serial.setRTS(False)
                self.serial.setDTR(False)
                serial_recv_thread = threading.Thread(target=self._read_serial)
                serial_recv_thread.daemon = True
                serial_recv_thread.start()
            except serial.SerialException as e:
                print(f"Error opening serial port: {e}, falling back to web")
                self.use_serial = False
        self.base_url = f'http://{ip_address}/js'

    def _send_command(self, payload, response_expected=False):
        """
        Send a command to the robot arm (either via serial or web API).
        """
        if self.use_serial:
            return self._send_serial_command(payload, response_expected)
        else:
            return self._send_web_command(payload)

    def _read_serial(self):
        while True:
            data = self.serial.readline().decode('utf-8')
            if data and data != '\r\n':
                if not self.serial_response_acked:
                    self.response_queue.put(data)
                    self.serial_response_acked = True
                elif self.queue_next_response:
                    self.response_queue.put(data)
                    self.queue_next_response = False
                else:
                    print(f"Received: {data}", end='')

    def _send_serial_command(self, payload, response_expected):
        """
        Send a command via serial communication.
        """
        command_str = json.dumps(payload)
        with self.send_lock:
            self.queue_next_response = response_expected
            self.serial_response_acked = False
            self.serial.write(command_str.encode() + b'\n')
            print(f"Sent: {command_str.strip()}")
            ack = self.response_queue.get(timeout=100)
            print(f"ACK: {ack}")
            if response_expected:
                response = self.response_queue.get(timeout=100)
                print(f"Received: {response}")
                return json.loads(response)
            else:
                return None

    def _send_web_command(self, payload):
        print(f"Sending {payload}")
        response = requests.get(self.base_url, params={'json': json.dumps(payload)})
        response.raise_for_status()
        return response

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
        return self._send_command({"T": 105}, response_expected=True)

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
    def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def tracking_update(feedback):
        axes = feedback['axes']
        pos = feedback['relative_pos']
        buttons = feedback['buttons']
        print(f"Relative Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        scale_factor = 800
        arm_pos['x'] = max(210, initial_pos['x'] - pos[0] * scale_factor)
        arm_pos['y'] = initial_pos['y'] - pos[1] * scale_factor
        arm_pos['z'] = max(-100, initial_pos['z'] + pos[2] * scale_factor)
        arm_pos['t'] = clamp(
            arm_pos['t'] + (axes[1] + axes[7]) * .05 + .03 * buttons[1] - .03 * buttons[5],
            1.5, 3.3)
        arm_pos['spd'] = 6
        arm.move_to_position(**arm_pos)

    arm = RoArm()
    arm.set_led(180)
    try:
        initial_pos = arm.get_position()
        print("Arm Start Position:", initial_pos)
        arm_pos = initial_pos.copy()
        tracker = MobileIOARTracker()
        tracker.continuous_tracking(callback=tracking_update, duration=6000)
    finally:
        arm.set_led(0)


if __name__ == "__main__":
    main()
