"""
Environment Configuration
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MainApp

"""
Environment Configuration Page Class
"""
class EnvConfigPage(tk.Frame):
    # EnvConfigPage initialiser
    def __init__(self, parent: MainApp, cfg: dict) -> None:
        super().__init__(parent)
        self.parent, self.cfg = parent, cfg

        wrapper = ttk.Frame(self)                       

        # Create header with back button and page title
        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        # Unicode for back arrow
        ttk.Button(hdr, text="\u2190 Back", command=lambda: parent.show_page("ModelConfigPage")).pack(side="left", padx=6, pady=6)
        ttk.Label(hdr, text="Environment Configuration", font=("Arial", 14)).pack(side="left", padx=4, pady=6)

        wrapper = ttk.Frame(self)                   
        wrapper.place(relx=.5, rely=.28, anchor="center")

        ttk.Label(wrapper, text="Environment Configuration", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)

        # Helper method to create labeled entry fields
        def _entry(r: int, label: str, default: str) -> tk.StringVar:
            ttk.Label(wrapper, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=3)
            v = tk.StringVar(value=default)
            ttk.Entry(wrapper, textvariable=v).grid(row=r, column=1, padx=6, pady=3)
            return v

        # Create entries for grid dimensions and counts
        self.width_var = _entry(1, "Grid width (pixels):", "640")
        self.height_var = _entry(2, "Grid height (pixels):", "480")
        self.fire_var = _entry(3, "Number of fires:", "1")
        self.lite_var = _entry(4, "Number of lightning:", "0")
        self.drn_var = _entry(5, "Number of drones:", "1")

        # Slider for minimum separation between fires
        ttk.Label(wrapper, text="Minimum separation between fires (blocks):").grid(row=6, column=0, sticky="w", padx=6, pady=3)
        self.sep_var = tk.Scale(wrapper, from_=1, to=15, orient="horizontal", length=200, resolution=1)
        self.sep_var.set(self.cfg["min_sep"])
        self.sep_var.grid(row=6, column=1)

        # Slider for perception vision radius
        ttk.Label(wrapper, text="Perception vision radius (1 – 10):").grid(row=7, column=0, sticky="w", padx=6, pady=3)
        self.vision_var = tk.Scale(wrapper, from_=1, to=10, orient="horizontal", length=200, resolution=1)         
        self.vision_var.set(self.cfg["vision"])                       
        self.vision_var.grid(row=7, column=1)                        

        ttk.Button(wrapper, text="Start Training", command=self._start).grid(row=8, column=0, columnspan=2, pady=15)

    # Update aforementioned inputs
    def _start(self):
        w = int(self.width_var.get())
        h = int(self.height_var.get())
        nf = int(self.fire_var.get())
        nl = int(self.lite_var.get())
        nd = int(self.drn_var.get())
        sep = int(self.sep_var.get())

        self.cfg.update({"width": w, "height": h, "num_fire": nf, "num_light": nl, "num_drones": nd, "min_sep": sep, "vision": int(self.vision_var.get())})
        self.parent.show_page("TrainingPage")