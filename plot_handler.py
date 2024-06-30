import matplotlib.pyplot as plt
import numpy as np


class PlotHandler:

    def __init__(self, camera_names):
        plt.ioff()
        self.fig, self.axs = plt.subplots((len(camera_names) + 1) // 2, 2, figsize=(10,8))
        self.camera_plots = {}
        self.camera_names = camera_names

        for idx, camera in enumerate(camera_names):
            ax = self.axs[idx // 2, idx % 2] if self.axs.ndim == 2 else self.axs[idx]
            self.camera_plots[camera] = ax.imshow(
                np.zeros((480, 640, 3), dtype=np.uint8),
            )
            ax.set_title(camera)
            ax.axis('off')  # Turn off axis for faster rendering

        self.fig.tight_layout()
        self.fig.show()
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()

    def render_images(self, images):
        for camera in self.camera_names:
            self.camera_plots[camera].set_data(images[camera])
        plt.pause(0.01)
