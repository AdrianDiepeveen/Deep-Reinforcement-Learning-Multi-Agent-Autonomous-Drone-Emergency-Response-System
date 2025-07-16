"""
Simulation Configuration
"""
from __future__ import annotations
import glob
import os
import sys
import tkinter as tk
from   tkinter import ttk, messagebox
from Multi_Agent   import Agent
from Simulation_Reporting import SimulationDashboard

"""
Helper Functions
"""
# Helper function to list saved AI models
def list_saved_models() -> list[str]:
    os.makedirs("model", exist_ok=True)
    pats = ["model/*.npz", "model/*.pkl"]
    return sorted([p for pat in pats for p in glob.glob(pat)], key=os.path.getmtime)

"""
Simulation Configuration Page Class
"""
class SimulationConfigPage(tk.Frame):
    # SimulationConfigPage initialiser
    def __init__(self, parent, cfg, root_app):
        super().__init__(parent)
        self.cfg, self.root_app = cfg, root_app

        wrapper = ttk.Frame(self)                     

        # Create header with back button and page title
        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        # Unicode for back arrow
        ttk.Button(hdr, text="\u2190 Back", command=lambda: parent.show_page("StartupPage")).pack(side="left", padx=6, pady=6)
        ttk.Label(hdr, text="Simulation Configuration", font=("Arial", 14)).pack(side="left", padx=4, pady=6)

        wrapper = ttk.Frame(self)                        
        wrapper.place(relx=.5, rely=.28, anchor="center")

        ttk.Label(wrapper, text="Simulation Configuration", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=12)

        # Select AI algorithm for simulation
        ttk.Label(wrapper, text="AI Algorithm:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.algo_var = tk.StringVar(value="Deep Q-learning")

        # Select between Q-learning, Deep Q-learning and Particle Swarm Optimisation
        ttk.Combobox(wrapper, textvariable=self.algo_var, values=["Q-learning", "Deep Q-learning"], state="readonly", width=32).grid(row=1, column=1, padx=6, pady=4)

        # Helper method to create entry fields
        def _entry(r: int, label: str, default: str) -> tk.StringVar:
            ttk.Label(wrapper, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
            v = tk.StringVar(value=default)
            ttk.Entry(wrapper, textvariable=v).grid(row=r, column=1, padx=6, pady=4)
            return v

        self.drn_var = _entry(2, "Number of drones:", "1")
        self.fire_var = _entry(3, "Number of fires :", "1")
        self.lgt_var = _entry(4, "Number of lightning storms:", "0")

        # Slider for minimum separation between fires
        ttk.Label(wrapper, text="Minimum separation between fires (blocks):").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        self.sep_var = tk.Scale(wrapper, from_=1, to=15, orient="horizontal", length=180, resolution=1)
        self.sep_var.set(cfg.get("min_sep", 6))
        self.sep_var.grid(row=5, column=1)

        # Slider for perception vision radius
        ttk.Label(wrapper, text="Perception vision radius (1 – 10):").grid(row=6, column=0, sticky="w", padx=6, pady=4)
        self.vis_var = tk.Scale(wrapper, from_=1, to=10, orient="horizontal", length=180, resolution=1)
        self.vis_var.set(cfg.get("sim_vision", 10))
        self.vis_var.grid(row=6, column=1)
        

        # Load trained AI models
        saved = [os.path.basename(p) for p in list_saved_models()]
        self.model_var = tk.StringVar(value="None")
        
        if saved:
            ttk.Label(wrapper, text="Load trained AI model:").grid(row=7, column=0, sticky="w", padx=6, pady=4)
            self.model_combo = ttk.Combobox(wrapper, textvariable=self.model_var, values=["None"] + saved, state="readonly", width=32)
            self.model_combo.grid(row=7, column=1, padx=6, pady=4)

        ttk.Button(wrapper, text="Start Simulation", command=self._start).grid(row=8, column=0, columnspan=2, pady=18)

    # Update aforementioned inputs
    def _start(self):
    
        nd = int(self.drn_var.get())
        nf = int(self.fire_var.get())
        nl = int(self.lgt_var.get())
        sep = int(self.sep_var.get())
        
        self.cfg.update({
            "sim_algorithm": self.algo_var.get(),
            "sim_num_drones": nd,
            "sim_num_fires": nf,
            "sim_num_light": nl,
            "sim_selected_model": self.model_var.get(),
            "min_sep": sep,
            "sim_vision": int(self.vis_var.get()),
        })
        self.root_app.show_page("SimulationPage")

    # Refresh list of AI models when page shown
    def on_show(self):
        if hasattr(self, "model_combo"):
            saved = [os.path.basename(p) for p in list_saved_models()]
            self.model_combo['values'] = ["None"] + saved