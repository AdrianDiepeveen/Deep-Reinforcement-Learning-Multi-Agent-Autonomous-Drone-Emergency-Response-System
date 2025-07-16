"""
Main Application
"""
from __future__ import annotations
import glob
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")            

from Training_Config import ModelConfigPage
from Environment_Config import EnvConfigPage
from Training import TrainingPage
from Simulation_Config import SimulationConfigPage
from Simulation import SimulationPage
from Comparative_Reporting import ComparativeReportingPage

"""
Helper Functions
"""
# Function to list all saved AI model files
def list_models() -> list[str]:
    os.makedirs("model", exist_ok=True)
    patterns = ["model/*.npz", "model/*.pkl"]
    paths = [p for pat in patterns for p in glob.glob(pat)]
    return sorted(paths, key=os.path.getmtime)

"""
Main Application
"""
class MainApp(tk.Tk):
    # MainApp initialiser
    def __init__(self) -> None:
        super().__init__()

        self.title("Autonomous Drones Emergency Response System")

        # Scale user interface for presentation purposes
        self.state("zoomed")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Shared configuration dictionary across all pages
        self.cfg: dict[str, object] = {
            # Training
            "mode": None,
            "algorithm": "Deep Q-learning",
            "epochs": 100,
            "epsilon": .1,
            "selected_model": "None",
            "width": 640,
            "height": 480,
            "num_fire": 1,
            "num_light": 0,
            "num_drones": 1,
            "min_sep": 6,
            "vision": 10,  

            # Simulation 
            "sim_algorithm": "Deep Q-learning",
            "sim_num_drones": 1,
            "sim_num_fires": 1,
            "sim_num_light": 0,
            "sim_selected_model": "None",
            "sim_vision": 10     
        }

        # Instantiate pages and store in dictionary
        self.pages: dict[str, tk.Frame] = {
            "StartupPage": StartupPage(self, self.cfg),
            "ModelConfigPage": ModelConfigPage(self, self.cfg),
            "EnvConfigPage": EnvConfigPage(self, self.cfg),
            "TrainingPage": TrainingPage(self, self.cfg),
            "SimulationConfigPage": SimulationConfigPage(self, self.cfg, self),
            "SimulationPage": SimulationPage(self, self.cfg, self),
            "ComparativeReporting": ComparativeReportingPage(self, self),
        }
        self.show_page("StartupPage")

    # Method to display requested page
    def show_page(self, name: str) -> None:
        frame = self.pages[name]
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

"""
Startup Home Page Class
"""
class StartupPage(tk.Frame):
    # StartupPage initialiser
    def __init__(self, parent: MainApp, cfg: dict) -> None:
        super().__init__(parent)

        # Wallpaper
        banner_path = os.path.join("assets", "drone_banner.jpg")
        img = Image.open(banner_path)
        self._banner_img = ImageTk.PhotoImage(img)
        ttk.Label(self, image=self._banner_img).place(relx=0.5, rely=0.05, anchor="n")
        self.parent, self.cfg = parent, cfg

        wrapper = ttk.Frame(self)                     
        wrapper.place(relx=.5, rely=.28, anchor="center") 
        wrapper = ttk.Frame(self)
        wrapper.place(relx=0.5, rely=0.28, anchor="center")
        wrapper.columnconfigure((0,1,2), weight=1)

        ttk.Label(wrapper, text="Welcome To The Autonomnous Drones Emergency Response System\nSelect a Mode:", font=("Arial", 16), justify="center").grid(row=0, column=0, columnspan=3, pady=20)
        
        # Selection of modes
        ttk.Button(wrapper, text="Training Mode", command=self._to_training).grid(row=1, column=0, padx=20, pady=10)
        ttk.Button(wrapper, text="Simulation Mode", command=self._to_sim).grid(row=1, column=1, padx=20, pady=10)
        ttk.Button(wrapper, text="Comparative Reporting", command=lambda: parent.show_page("ComparativeReporting")).grid(row=1, column=2, padx=20, pady=10)

    # Change to training configuration page
    def _to_training(self):
        self.cfg["mode"] = "training"
        self.parent.show_page("ModelConfigPage")

    # Change to simulation configuration page
    def _to_sim(self):
        self.cfg["mode"] = "simulation"
        self.parent.show_page("SimulationConfigPage")

"""
Main Entry Point Of Application
"""
if __name__ == "__main__":
    MainApp().mainloop()