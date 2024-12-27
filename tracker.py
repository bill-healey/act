import hebi
import time
import numpy as np


class MobileIOARTracker:
    def __init__(self):
        self._initial_position = None
        device = None

        while device is None:
            lookup = hebi.Lookup()
            time.sleep(0.25)
            for device in lookup.entrylist:
                break
            if device is None:
                print("No Device found, start Mobile IO app on iPhone")
                time.sleep(1)

        self.group = lookup.get_group_from_names([device.family], [device.name])
        self.group.feedback_frequency = 200  # 200Hz, max is likely 1kHz linux and 640Hz Win7
        self.fbk = hebi.GroupFeedback(self.group.size)

    def reset_initial_position(self):
        self._initial_position = None

    def _get_feedback(self):
        axes = {}
        buttons = {}
        self.fbk = self.group.get_next_feedback(reuse_fbk=self.fbk)
        if self.fbk is None:
            print("Could not get feedback")
        orient = self.fbk[0].ar_orientation
        # reorder w,x,y,z to x,y,z,w
        orient = [*orient[1:], orient[0]]
        pos = self.fbk[0].ar_position
        for i in range(1, 9):
            io_a = self.fbk[0].io.a
            io_b = self.fbk[0].io.b
            axes[i] = io_a.get_float(i)
            buttons[i] = io_b.get_int(i)
        #r = R.from_quat(orient)
        #rot_mat = r.as_matrix()
        #rot_mat = np.eye(3)
        if self._initial_position is None:
            self._initial_position = np.ndarray.copy(pos)

        return {
            'pos': pos,
            'relative_pos': pos - self._initial_position,
            'axes': axes,
            'buttons': buttons,
        }

    def continuous_tracking(self, callback=None, duration=None):
        start_time = time.time()
        try:
            while duration is None or time.time() - start_time < duration:
                feedback = self._get_feedback()
                if callback:
                    callback(feedback)
        except KeyboardInterrupt:
            print("Tracking stopped by user.")


def print_position(feedback):
    pos = feedback['pos']
    print(f"Relative Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} meters")


def main():
    try:
        tracker = MobileIOARTracker()
        tracker.continuous_tracking(callback=print_position, duration=60)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
