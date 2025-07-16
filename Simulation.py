"""
Simulation
"""
from __future__ import annotations
import os
import sys
import tkinter as tk
from tkinter import ttk
import pygame
import json

from Environment import DroneEnvironmentAI
from Multi_Agent import Agent
from Simulation_Reporting import SimulationDashboard

"""
Constants
"""
# Define path to ledger file containing comparative reporting data in json format
LEDGER = "Comparative_Reporting_Data.json"

"""
Helper Functions
"""
# Function to safely load model state from file
def safe_load_state(model, path: str) -> None:
    model.load(path)

# Function to append simulation metrics to ledger file containing comparative reporting data
def _append_to_ledger(rec: dict):
    if os.path.isfile(LEDGER):
        with open(LEDGER, "r", encoding="utf8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(rec)
    with open(LEDGER, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)
   

"""
Simulation Page Class
"""
class SimulationPage(tk.Frame):
    # SimulationPage initialiser
    def __init__(self, parent, cfg, root_app):
        super().__init__(parent)

        # Create header with back button and page title
        hdr = ttk.Frame(self)
        hdr.place(x=10, y=10)
        # Unicode for back arrow
        ttk.Button(hdr, text="\u2190 Back", command=lambda: root_app.show_page("StartupPage")).pack(side="left", padx=6, pady=6)
        ttk.Label(hdr, text="Simulation", font=("Arial", 14)).pack(side="left", padx=4, pady=6)

        self.cfg, self.root_app = cfg, root_app

        # Create buttons for reset, save and quit
        self.btn_reset = ttk.Button(self, text="Reset", command=self._reset)
        self.btn_save = ttk.Button(self, text="Save Report", command=self._save_report)
        self.btn_quit = ttk.Button(self, text="Quit", command=lambda: sys.exit())

        self.btn_reset.grid(row=2, column=0, padx=8, pady=6, sticky="e")
        self.btn_save.grid(row=2, column=1, padx=8, pady=6)
        self.btn_quit.grid(row=2, column=2, padx=8, pady=6, sticky="w")

        self.dashboard: SimulationDashboard | None = None
        self.agent: Agent | None = None
        self.game: DroneEnvironmentAI | None = None

    # Prepare dashboard and restart simulation
    def on_show(self):
        if self.dashboard is None:
            self.rowconfigure(1, weight=1)
            for c in (0, 1, 2):
                self.columnconfigure(c, weight=1)
            self.dashboard = SimulationDashboard(self)

        pygame.display.quit()
        self.dashboard.prepare_pygame()

        # Restart simulation
        self._reset()

    # Initialise simulation environment and agent
    def _reset(self):
        self.agent = Agent(num_drones=self.cfg["sim_num_drones"], algo=self.cfg["sim_algorithm"])
        self.agent.epsilon = 0.0

        ckpt = self.cfg["sim_selected_model"]
        if ckpt != "None":
            path = os.path.join("model", ckpt)
            safe_load_state(self.agent.model, path)

        # Initialise simulation environment
        self.game = DroneEnvironmentAI(num_fire = self.cfg["sim_num_fires"],
                                       num_light = self.cfg["sim_num_light"],
                                       num_drones = self.cfg["sim_num_drones"],
                                       is_simulation = True,
                                       min_sep_blocks = self.cfg["min_sep"],
                                       vision_scale = self.cfg["sim_vision"])

        self.dashboard.reset()

        # Speed of simulation frame rate
        self.after(100, self._loop)

    # Run one simulation step and update dashboard
    def _loop(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        actions = []
        for d in range(self.cfg["sim_num_drones"]):
            st = self.agent.get_state_for_drone(self.game, d)
            self.agent._last_game = self.game
            self.agent._last_idx  = d

            # AGENT-ORIENTED SYSTEM REQUIREMENTS
            # 2.) INTERACTION WITH ENVIRONMENT THROUGH SPECIFIC ACTUATORS
            # ACTUATORS
            actions.append(self.agent.get_action(st))

        # AGENT-ORIENTED SYSTEM REQUIREMENTS
        # 2.) INTERACTION WITH ENVIRONMENT THROUGH SPECIFIC ACTUATORS
        # ACTUATORS
        self.game.execute_step_multi(actions)

        self.dashboard.add_point(self.game.score, self.game.dist_total, self.game.collision_total, self.game.light_collision_total, self.game.empty_batt_total)

        self.after(100, self._loop)

    # Save current simulation metrics to ledger file containing comparative reporting data
    def _save_report(self):
        stamp = self.cfg["sim_selected_model"]
        if stamp == "None":
            import time
            stamp = f"sim_{time.strftime('%Y%m%d_%H%M%S')}.sim"

        # Prefix to indentify simulation metrics
        session_name = f"Simulation_{stamp}"
        rec = {"session": session_name, "type": "simulation", "metrics": {"fires": self.dashboard.fires, "avg"  : self.dashboard.avg, "dists": self.dashboard.dists, "dcols": self.dashboard.dcols, "lcols": self.dashboard.lcols, "empty": self.dashboard.empty}}
        _append_to_ledger(rec)
        from tkinter import messagebox
        messagebox.showinfo("Saved", f"Metrics appended for session:\n{stamp}")