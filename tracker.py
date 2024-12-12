import hebi
import time
import numpy as np


class MobileIOARTracker:
    def __init__(self):
        lookup = hebi.Lookup()
        time.sleep(.25)
        device = None
        for entry in lookup.entrylist:
            print(entry)
            device = entry
        if not device:
            raise Exception("No Device found, start Mobile IO app on iphone")
        self.group = lookup.get_group_from_names([device.family], [device.name])
        self.fbk = hebi.GroupFeedback(self.group.size)
        self._initial_position = None
        self.reset_initial_position()

    def reset_initial_position(self):
        self._initial_position = np.ndarray.copy(self._get_pos())
        print("Initial position reset successfully.")

    def _get_pos(self):
        self.fbk = self.group.get_next_feedback(reuse_fbk=self.fbk)
        if self.fbk is None:
            print("Could not get feedback")
        orient = self.fbk[0].ar_orientation
        # reorder w,x,y,z to x,y,z,w
        orient = [*orient[1:], orient[0]]
        pos = self.fbk[0].ar_position
        #r = R.from_quat(orient)
        #rot_mat = r.as_matrix()
        #rot_mat = np.eye(3)

        return pos

    def get_relative_position(self):
        return self._get_pos() - self._initial_position

    def continuous_tracking(self, callback=None, duration=None):
        start_time = time.time()
        try:
            while duration is None or time.time() - start_time < duration:
                relative_pos = self.get_relative_position()
                if callback:
                    callback(relative_pos)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Tracking stopped by user.")


def print_position(pos):
    print(f"Relative Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} meters")


def main():
    try:
        tracker = MobileIOARTracker()
        tracker.reset_initial_position()
        tracker.continuous_tracking(callback=print_position, duration=60)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
