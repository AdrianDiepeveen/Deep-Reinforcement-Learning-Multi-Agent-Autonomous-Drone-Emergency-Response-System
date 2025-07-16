"""
Training Configuration
"""
from __future__ import annotations
import glob
import os
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainApp

"""
Helper fuctions
"""
# Helper function to list saved AI models
def list_models() -> list[str]:
    os.makedirs("model", exist_ok=True)
    patterns = ["model/*.npz", "model/*.pkl"]
    paths = [p for pat in patterns for p in glob.glob(pat)]
    return sorted(paths, key=os.path.getmtime)

"""
Model Configuration Page Class
"""
class ModelConfigPage(tk.Frame):
    # ModelConfigPage initialiser
    def __init__(self, parent: MainApp, cfg: dict) -> None:
        super().__init__(parent)
        self.parent, self.cfg = parent, cfg

        # Create header with back button and page title
        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        # Unicode for back arrow
        ttk.Button(hdr, text="\u2190 Back", command=lambda: parent.show_page("StartupPage")).pack(side="left", padx=6, pady=6)
        ttk.Label(hdr, text="Model Training Configuration", font=("Arial", 14)).pack(side="left", padx=4, pady=6)

        wrapper = ttk.Frame(self)                        
        wrapper.place(relx=.5, rely=.28, anchor="center")

        ttk.Label(wrapper, text="Model Training Configuration", font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Select AI algorithm
        ttk.Label(wrapper, text="Select AI Algorithm:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.algo_var = tk.StringVar(value="Deep Q-learning")
        ttk.Combobox(wrapper, textvariable=self.algo_var, values=["Q-learning", "Deep Q-learning"], state="readonly", width=32).grid(row=1, column=1, padx=6, pady=3)

        # Number of epochs
        ttk.Label(wrapper, text="Number of epochs:").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(wrapper, textvariable=self.epochs_var).grid(row=2, column=1, padx=6, pady=3)

        # Epsilon
        ttk.Label(wrapper, text="Epsilon [0-1]:").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        self.epsilon_var = tk.StringVar(value="0.1")
        ttk.Entry(wrapper, textvariable=self.epsilon_var).grid(row=3, column=1, padx=6, pady=3)
        
        # Discount factor (gamma)
        ttk.Label(wrapper, text="Discount factor [0-1]:").grid(row=4, column=0, sticky="w", padx=6, pady=3)
        self.gamma_var = tk.StringVar(value="0.9")
        ttk.Entry(wrapper, textvariable=self.gamma_var).grid(row=4, column=1, padx=6, pady=3)

        # Load existing AI model if saved
        saved = [os.path.basename(p) for p in list_models()]
        self.model_var = tk.StringVar(value="None")
        if saved:
            ttk.Label(wrapper, text="Load Existing AI Model:").grid(row=5, column=0, sticky="e", padx=6, pady=3)
            self.model_combo = ttk.Combobox(wrapper, textvariable=self.model_var, values=["None"] + saved, state="readonly", width=32)
            self.model_combo.grid(row=5, column=1, padx=6, pady=3)

        ttk.Button(wrapper, text="Next", command=self._confirm).grid(row=6, column=0, columnspan=2, pady=15)

    # Confirm aforementioned selections and update shared configuration
    def _confirm(self):
       
        epochs = int(self.epochs_var.get())
        eps = float(self.epsilon_var.get())
        gamma = float(self.gamma_var.get())
     
        self.cfg.update({"algorithm": self.algo_var.get(), "epochs": epochs, "epsilon": eps, "gamma": gamma, "selected_model": self.model_var.get()})
        self.parent.show_page("EnvConfigPage")

    # Refresh list of AI models when page shown
    def on_show(self):
        if hasattr(self, "model_combo"):
            saved = [os.path.basename(p) for p in list_models()]
            self.model_combo['values'] = ["None"] + saved