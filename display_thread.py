import cv2, threading, time, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class DisplayThread:
    def __init__(self, name="Display", image_size=(1280, 960)):
        self.window_name = name
        self.camera_images = {}
        self.plot_images = {}
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.image_size = image_size

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        cv2.destroyAllWindows()

    def update_frame(self, key, frame):
        with self.lock:
            self.camera_images[key] = frame

    def update_frames(self, images):
        with self.lock:
            self.camera_images = images

    def update_plot(self, key, figure):
        c = FigureCanvas(figure)
        c.draw()
        w, h = c.get_width_height()
        img = np.frombuffer(c.tostring_rgb(), dtype="uint8").reshape(h, w, 3)
        with self.lock:
            self.plot_images[key] = img

    def plot_action(self, ep_data, key="action_plot"):
        actions = np.array([step["action"] for step in ep_data])
        timesteps = np.arange(len(actions))
        plt.close()
        fig = plt.figure(figsize=(16, 4))
        for i, label in enumerate(["b", "s", "e", "t"]):
            plt.plot(timesteps, actions[:, i], label=f"Servo {label}")
        plt.xlabel("Time Step")
        plt.ylabel("Position")
        plt.legend()
        plt.tight_layout()
        self.update_plot(key, fig)

    def plot_train_validation_history(self, train_history, validation_history):
        # Ensure the data is in NumPy format and on the CPU
        train_values = [summary['loss'].cpu().numpy() if hasattr(summary['loss'], 'cpu') else summary['loss'] for
                        summary in train_history]
        val_values = [summary['loss'].cpu().numpy() if hasattr(summary['loss'], 'cpu') else summary['loss'] for
                      summary in validation_history]

        # Dynamically create epochs based on train_values
        epochs = np.arange(len(train_values))

        plt.close()
        fig = plt.figure(figsize=(16, 9))
        # Plot the training and validation loss
        plt.plot(epochs, train_values, label="Train Loss")
        plt.plot(epochs[:len(val_values)], val_values, label="Validation Loss")  # Match val_values length
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs. Validation Loss")
        plt.tight_layout()
        self.update_plot('train/validation', fig)

    def _combine_images(self, cameras, plots):
        camera_array = None
        if cameras:
            resized = [cv2.resize(c, self.image_size) for c in cameras]
            camera_array = resized[0] if len(resized) == 1 else np.hstack(resized)

        plot_array = None
        if plots:
            target_width = camera_array.shape[1] if camera_array is not None else self.image_size[0]
            resized_plots = []
            for p in plots:
                resized_plots.append(cv2.resize(p, (target_width, self.image_size[1])))
            plot_array = resized_plots[0] if len(resized_plots) == 1 else np.vstack(resized_plots)

        if camera_array is not None and plot_array is not None:
            return np.vstack([camera_array, plot_array])
        return camera_array if camera_array is not None else plot_array

    def _run(self):
        while self.running:
            with self.lock:
                cameras = list(self.camera_images.values())
                plots = list(self.plot_images.values())

            combined = self._combine_images(cameras, plots)
            if combined is not None:
                cv2.imshow(self.window_name, combined)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
            time.sleep(0.01)


def main():
    disp = DisplayThread(image_size=(1280, 960))
    disp.start()

    cams = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in range(2)]
    try:
        for i in range(10):
            images = {}
            for idx, cam in enumerate(cams):
                r, frame = cam.read()
                if r:
                    images[f"cam{idx}"] = frame
            disp.update_frames(images)
            time.sleep(0.05)
        tdata = [{"loss": 1 - 0.02 * i} for i in range(50)]
        vdata = [{"loss": 1.2 - 0.015 * i} for i in range(50)]
        disp.plot_train_validation_history(tdata, vdata, len(tdata))
        time.sleep(5)
    finally:
        disp.stop()
        for c in cams: c.release()


if __name__ == "__main__":
    main()
