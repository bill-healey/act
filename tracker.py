import hebi
import time
import threading
import numpy as np


class MobileIOARTracker:
    def __init__(self):
        self._initial_position = None
        self._stop_event = threading.Event()
        self._thread = None
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
        self.group.feedback_frequency = 1600
        self.fbk = hebi.GroupFeedback(self.group.size)

    def reset_initial_position(self):
        self._initial_position = None

    def get_last_feedback(self):
        return self._last_feedback_dict

    def _get_feedback(self):
        self.fbk = self.group.get_next_feedback(reuse_fbk=self.fbk)
        if self.fbk is None:
            print("Could not get feedback")
            return None

        pos = self.fbk[0].ar_position
        if self._initial_position is None:
            self._initial_position = np.copy(pos)

        axes, buttons = {}, {}
        for i in range(1, 9):
            axes[i] = self.fbk[0].io.a.get_float(i)
            buttons[i] = self.fbk[0].io.b.get_int(i)

        self._last_feedback_dict = {
            'pos': pos,
            'relative_pos': pos - self._initial_position,
            'axes': axes,
            'buttons': buttons
        }
        return self._last_feedback_dict

    def continuous_tracking(self, callback=None, duration=None, blocking=True):
        """
        Starts AR tracking in a separate thread.
          - callback: function(feedback_dict) to handle each feedback.
          - duration: maximum seconds to run (None for indefinite).
          - blocking: if True, this call will not return until duration elapsed or stop_tracking() is called.
        """
        self._stop_event.clear()

        def _track_loop():
            start_time = time.time()
            while not self._stop_event.is_set():
                if duration is not None and (time.time() - start_time) > duration:
                    break
                feedback = self._get_feedback()
                if feedback and callback:
                    callback(feedback)

        self._thread = threading.Thread(target=_track_loop, daemon=True)
        self._thread.start()

        if blocking:
            self._thread.join()

    def stop_tracking(self):
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()


def print_position(feedback):
    pos = feedback['pos']
    print(f"x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")


def main():
    try:
        tracker = MobileIOARTracker()
        tracker.continuous_tracking(callback=print_position, duration=30, blocking=False)

        for i in range(5):
            print("Main thread doing something else...")
            time.sleep(2)

        # Stop the tracker before exiting
        tracker.stop_tracking()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
