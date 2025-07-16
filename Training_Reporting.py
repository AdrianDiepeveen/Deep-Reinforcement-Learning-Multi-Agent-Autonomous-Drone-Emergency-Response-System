"""
Training Reporting
"""
from __future__ import annotations
import os
import sys
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""
Constants
"""
# Line colours for plots
GREEN_BG = "#00140d"
GREEN_AX_BG = "#002b1a"
NEON = "#00ff7f"
LIME = "#7fff00"
SPRING = "#3cff00"
ORANGE = "#ffa500"
FG = "#ffffff"

"""
Training Dashboard Class
"""
class TrainingDashboard:
    # TrainingDashboard initialiser
    def __init__(self, master: tk.Frame) -> None:
        self.master = master

        # Configure rows for pygame and figures
        self.master.rowconfigure(0, weight=0)
        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Create embedded pygame frame
        self.pg_frame = tk.Frame(master, width=640, height=480, bg=GREEN_BG, highlightthickness=0)
        self.pg_frame.grid(row=0, column=0, pady=4)
        self.pg_frame.grid_columnconfigure(0, weight=1)

        # Create matplotlib figure and axes
        self.fig, axs = plt.subplots(1, 3, figsize=(10, 3.5), dpi=100)
        self.axes = list(axs)
        self._style_figure()

        # Initialise plot lines for each metric
        lines = self.axes[0].plot([], [], lw=2, color=NEON)
        self.line_score = lines[0]
        lines = self.axes[1].plot([], [], lw=2, color=LIME)
        self.line_mean = lines[0]
        lines = self.axes[2].plot([], [], lw=2, color=SPRING)
        self.line_dist = lines[0]

        # Embed matplotlib canvas into Tkinter layout
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # Initialise buffers for metrics
        self.scores: list[int] = []
        self.means: list[float] = []
        self.dists: list[int] = []

    # Method to embed pygame into Tkinter frame
    def prepare_pygame(self) -> None:
        self.pg_frame.update_idletasks()
        os.environ["SDL_WINDOWID"] = str(self.pg_frame.winfo_id())

        if sys.platform.startswith("win"):
            os.environ["SDL_VIDEODRIVER"] = "windib"

    # Reset figures
    def reset(self) -> None:
        self.scores.clear()
        self.means.clear()
        self.dists.clear()
        self._redraw()

    # Add metrics and refresh figures
    def add_point(self, score: int, mean: float, dist: int) -> None:
        self.scores.append(score)
        self.means.append(mean)
        self.dists.append(dist)
        self._redraw()

    # Style figures and axes
    def _style_figure(self):
        self.fig.patch.set_facecolor(GREEN_BG)
        titles = ("Fires Extinguished", "Average Fires Per Epoch", "Distance Travelled (Blocks)")

        for ax, title in zip(self.axes, titles):
            ax.set_facecolor(GREEN_AX_BG)
            ax.set_title(title, color=FG, pad=8, fontsize=10, weight="bold")
            ax.set_xlabel("Epoch", color=FG, fontsize=9)
            ax.set_ylabel("Count" if "Distance" not in title else "Blocks", color=FG, fontsize=9)
            ax.tick_params(colors=FG)
            for sp in ax.spines.values():
                sp.set_color(FG)
                
        self.fig.tight_layout(pad=3)

    # Update data and redraw canvas
    def _redraw(self):
        x = range(len(self.scores))
        self.line_score.set_data(x, self.scores)
        self.line_mean.set_data(x, self.means)
        self.line_dist.set_data(x, self.dists)
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()