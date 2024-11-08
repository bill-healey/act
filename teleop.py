import threading
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time
from pynput import keyboard

class TeleOpHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.joystick.init()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        self.joystick_count = pygame.joystick.get_count()
        self.joystick = pygame.joystick.Joystick(0) if self.joystick_count > 0 else None
        if self.joystick:
            self.joystick.init()

        self.running = True
        self.lock = threading.Lock()
        # Separate input variables for keyboard and joystick
        self.keyboard_input = {
            'waist_rotation': .0,
            'shoulder_elevation': .0,
            'wrist_elevation': .0,
            'gripper_rotation': .0
        }
        self.joystick_input = {
            'waist_rotation': .0,
            'shoulder_elevation': .0,
            'wrist_elevation': .0,
            'gripper_rotation': .0
        }

    @staticmethod
    def apply_deadzone_and_exponential(input_value, deadzone=0.2, exponent=1):
        # Apply deadzone
        if abs(input_value) < deadzone:
            return 0
        # Adjust for deadzone and apply exponential scaling
        adjusted_input = (abs(input_value) - deadzone) / (1 - deadzone)  # Normalize
        adjusted_input = adjusted_input ** exponent * (1 if input_value > 0 else -1)
        return adjusted_input

    def run(self):
        while self.running:
            pygame.event.pump()  # Handle internal events
            if self.joystick:
                with self.lock:
                    self.joystick_input['waist_rotation'] = self.apply_deadzone_and_exponential(
                        self.joystick.get_axis(4)) * -0.03
                    self.joystick_input['shoulder_elevation'] = self.apply_deadzone_and_exponential(
                        self.joystick.get_axis(3)) * -0.02
                    self.joystick_input['wrist_elevation'] = self.apply_deadzone_and_exponential(
                        self.joystick.get_axis(1)) * -0.05
                    self.joystick_input['gripper_rotation'] = self.apply_deadzone_and_exponential(
                        self.joystick.get_axis(0)) * 0.1
            time.sleep(0.01)  # Limit the polling rate

    def on_press(self, key):
        with self.lock:
            try:
                if key.vk == 100:  # Numpad 4
                    self.keyboard_input['waist_rotation'] = 0.02  # Start rotating left (waist)
                elif key.vk == 102:  # Numpad 6
                    self.keyboard_input['waist_rotation'] = -0.02  # Start rotating right (waist)
                elif key.vk == 104:  # Numpad 8
                    self.keyboard_input['shoulder_elevation'] = -0.02  # Shoulder up
                elif key.vk == 101:  # Numpad 5
                    self.keyboard_input['shoulder_elevation'] = 0.02  # Shoulder down
                elif key.vk == 105:  # Numpad 9
                    self.keyboard_input['wrist_elevation'] = -0.02  # Elbow up
                elif key.vk == 99:  # Numpad 3
                    self.keyboard_input['wrist_elevation'] = 0.02  # Elbow down
                elif key.vk == 111:  # Numpad /
                    self.keyboard_input['gripper_rotation'] = -0.04  # Gripper open
                elif key.vk == 106:  # Numpad *
                    self.keyboard_input['gripper_rotation'] = 0.04  # Gripper close
            except AttributeError:
                pass  # Ignore non-virtual key presses

    def on_release(self, key):
        with self.lock:
            try:
                if key.vk in [100, 102]:  # Numpad 4 or 6
                    self.keyboard_input['waist_rotation'] = 0  # Reset on release
                elif key.vk in [104, 101]: # Numpad 8 or 5
                    self.keyboard_input['shoulder_elevation'] = 0  # Stop moving shoulder
                elif key.vk in [105, 99]:  # Numpad 9 or 3
                    self.keyboard_input['wrist_elevation'] = 0  # Stop moving elbow
                elif key.vk in [111, 106]: # Numpad / or *
                    self.keyboard_input['gripper_rotation'] = 0  # Stop moving gripper
            except AttributeError:
                pass  # Ignore non-virtual key releases

    def get_actions(self):
        with self.lock:
            # Sum keyboard and joystick inputs for final actions
            return {key: self.keyboard_input[key] + self.joystick_input[key] for key in self.keyboard_input}

    def stop(self):
        self.running = False
        pygame.quit()
        self.listener.stop()
