"""
Simulation Reporting
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
# Line colours for figures
GREEN_BG = "#00140d"
GREEN_AX_BG = "#002b1a"
NEON = "#00ff7f"
LIME = "#7fff00"
SPRING = "#3cff00"
ORANGE = "#ffa500"
FG = "#ffffff"

"""
Simulation Dashboard Class
"""
class SimulationDashboard:
    # SimulationDashboard initialiser
    def __init__(self, master: tk.Frame) -> None:
        self.master = master
        
        # Configure rows for pygame and figures
        self.master.rowconfigure(0, weight=0)
        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Create embedded pygame frame
        self.pg_frame = tk.Frame(master, width=640, height=480, bg=GREEN_BG, highlightthickness=0)
        self.pg_frame.grid(row=0, column=0, columnspan=3, pady=4)
        self.pg_frame.grid_columnconfigure(0, weight=1)

        # Create matplotlib figure and axes
        self.fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), dpi=100, gridspec_kw={'hspace': 0.6})
        self.axes = list(axs.flatten())
        self._style_figure()

        # Initialise plot lines for each metric
        lines = self.axes[0].plot([], [], lw=2, color=NEON)
        self.l_fire = lines[0]
        lines = self.axes[1].plot([], [], lw=2, color=LIME)
        self.l_dist = lines[0]
        lines = self.axes[2].plot([], [], lw=2, color=SPRING)
        self.l_dcol = lines[0]
        lines = self.axes[3].plot([], [], lw=2, color=ORANGE)
        self.l_lcol = lines[0]
        lines = self.axes[4].plot([], [], lw=2, color="#ff66ff")
        self.l_empty = lines[0]
        lines = self.axes[5].plot([], [], lw=2, color=NEON)
        self.l_avg = lines[0]

        # Embed matplotlib canvas into Tkinter layout
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

        # Initialise buffers for metrics
        self.fires: list[int] = []
        self.dists: list[int] = []
        self.dcols: list[int] = []
        self.lcols: list[int] = []
        self.empty: list[int] = []
        self.avg: list[float] = []

    # Method to embed pygame into Tkinter frame
    def prepare_pygame(self) -> None:
        self.pg_frame.update_idletasks()
        os.environ["SDL_WINDOWID"] = str(self.pg_frame.winfo_id())

        if sys.platform.startswith("win"):
            os.environ["SDL_VIDEODRIVER"] = "windib"

    # Style figures and axes
    def _style_figure(self):
        self.fig.patch.set_facecolor(GREEN_BG)
        titles = ("Total Fires", "Total Distance (Blocks)", "Drone Collisions", "Lightning Collisions", "Battery Depleted Events", "Avg Fires Per Collision")
        
        for ax, title in zip(self.axes, titles):
            ax.set_facecolor(GREEN_AX_BG)
            ax.set_title(title, color=FG, pad=6, fontsize=9, weight="bold")
            ax.tick_params(colors=FG)
            for sp in ax.spines.values():
                sp.set_color(FG)
        self.fig.tight_layout(pad=2)

    # Reset figures
    def reset(self):
        self.fires.clear()
        self.dists.clear()
        self.dcols.clear()
        self.lcols.clear()
        self.empty.clear()
        self.avg.clear()
        self._redraw()

    # Add metrics and refresh figures
    def add_point(self, fires: int, dist: int, drone_collisions: int, lightning_collisions: int, empty_batt: int) -> None:
        self.fires.append(fires)
        self.dists.append(dist)
        self.dcols.append(drone_collisions)
        self.lcols.append(lightning_collisions)
        self.empty.append(empty_batt)
        avg = fires / drone_collisions if drone_collisions > 0 else 0.0
        self.avg.append(avg)
        self._redraw()

    # Update data and redraw canvas
    def _redraw(self):
        x = range(len(self.fires))

        for line, data in ((self.l_fire , self.fires), (self.l_dist , self.dists), (self.l_dcol , self.dcols), (self.l_lcol , self.lcols), (self.l_empty, self.empty), (self.l_avg  , self.avg)):
            line.set_data(x, data)

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()