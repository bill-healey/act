import threading
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
            'elbow_elevation': .0,
            'gripper_rotation': .0
        }
        self.joystick_input = {
            'waist_rotation': .0,
            'shoulder_elevation': .0,
            'elbow_elevation': .0,
            'gripper_rotation': .0
        }

    def run(self):
        while self.running:
            pygame.event.pump()  # Handle internal events
            if self.joystick:
                with self.lock:
                    self.joystick_input['waist_rotation'] = self.joystick.get_axis(0) * 0.01
                    self.joystick_input['shoulder_elevation'] = self.joystick.get_axis(1) * 0.01
            time.sleep(0.1)  # Limit the polling rate

    def on_press(self, key):
        with self.lock:
            try:
                if key.vk == 100:  # Numpad 4
                    self.keyboard_input['waist_rotation'] += 0.02  # Increment left rotation
                elif key.vk == 102:  # Numpad 6
                    self.keyboard_input['waist_rotation'] -= 0.02  # Increment right rotation
                # Additional key mappings...
            except AttributeError:
                pass  # Ignore non-virtual key presses

    def on_release(self, key):
        with self.lock:
            try:
                if key.vk in [100, 102]:  # Numpad 4 or 6
                    self.keyboard_input['waist_rotation'] = 0  # Reset on release
                # Additional resets for other controls
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
