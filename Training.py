"""
Training Class
"""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pygame
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import MainApp

from Multi_Agent import Agent
from Environment import DroneEnvironmentAI
from Training_Reporting import TrainingDashboard
import json

"""
Constants
"""
# Define path to ledger file containing comparative reporting data in json format
LEDGER = "Comparative_Reporting_Data.json"

"""
Helper Functions
"""
# Function to append training metrics to ledger file containing comparative reporting data
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
Training Page Class
"""
class TrainingPage(tk.Frame):
    # TrainingPage initialiser
    def __init__(self, parent: MainApp, cfg: dict) -> None:
        super().__init__(parent)

        # Create header with back button and page title
        hdr = ttk.Frame(self)
        hdr.place(x=10, y=10)
        # Unicode for back arrow
        ttk.Button(hdr, text="\u2190 Back", command=lambda: parent.show_page("StartupPage")).pack(side="left", padx=6, pady=6)
        ttk.Label(hdr, text="Training",font=("Arial", 14)).pack(side="left", padx=4, pady=6)

        self.parent, self.cfg = parent, cfg
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        ttk.Button(self, text="Save Trained Model", command=lambda: setattr(self, "stop_req", True)).grid(row=2, column=0, pady=5)

        self.dashboard: TrainingDashboard | None = None
        self.agent: Agent | None = None
        self.game: DroneEnvironmentAI| None = None

        self.stop_req = False
        self.total_score = self.record = self.game_cnt = 0
        
    # Prepare dashboard and start training
    def on_show(self):
        self._initialise()
        self.after(1, self._train_loop)

    # Initialise dashboard, agent and environment
    def _initialise(self):
        # Initialise dashboard and embed pygame
        if self.dashboard is None:
            self.dashboard = TrainingDashboard(self)
        pygame.display.quit()
        self.dashboard.prepare_pygame()

        # Initialise agent and apply hyperparameters
        self.agent = Agent(self.cfg["num_drones"], algo=self.cfg["algorithm"])

        # Common hyperparameters between AI models
        # Epsilon
        self.agent.epsilon = self.cfg["epsilon"]

        # Discount factor (gamma)
        self.agent.gamma = self.cfg.get("gamma", self.agent.gamma)

        # If Deep Q-learning model
        if hasattr(self.agent, "trainer"):
            self.agent.trainer.gamma = self.cfg["gamma"]

        # If Q-learning model
        if hasattr(self.agent, "qtable"):
            self.agent.qtable.gamma = self.cfg["gamma"]

        # If Particle Swarm Optimisation model
        if hasattr(self.agent, "psomodel"):
            self.agent.psomodel.gamma = self.cfg["gamma"]

        if self.cfg["selected_model"] != "None":
            # load from model/ directory
            import os
            path = os.path.join("model", self.cfg["selected_model"])
            self.agent.model.load(path)

        # Initialise simulation environment
        self.game = DroneEnvironmentAI(w = self.cfg["width"],
                                       h = self.cfg["height"],
                                       num_fire = self.cfg["num_fire"],
                                       num_light = self.cfg["num_light"],
                                       num_drones = self.cfg["num_drones"],
                                       min_sep_blocks = self.cfg["min_sep"],
                                       vision_scale = self.cfg["vision"])

        self.stop_req  = False
        self.total_score = self.record = self.game_cnt = 0
        self.dashboard.reset()

    # Run training until completion or stopped by user
    def _train_loop(self):
        if self.game_cnt >= self.cfg["epochs"]:
            print("\nTraining complete.")
            self._training_done_popup()
            return
        
        if self.stop_req:
            self._save_and_back()
            return

        done = False
        while not done:
            # Collect current states and actions for each agent
            states, moves = [], []
            for d in range(self.cfg["num_drones"]):
                st = self.agent.get_state_for_drone(self.game, d)
                self.agent._last_game = self.game
                self.agent._last_idx  = d
                mv = self.agent.get_action(st)
                states.append(st)
                moves.append(mv)

            # Execute environment step and receive rewards
            rewards, done, score = self.game.execute_step_multi(moves)

            # Get next states for training
            nexts = [self.agent.get_state_for_drone(self.game, d) for d in range(self.cfg["num_drones"])]

            # Train agent
            for d in range(self.cfg["num_drones"]):
                self.agent.train_short_memory(states[d], moves[d], rewards[d], nexts[d], done)
                self.agent.remember(states[d], moves[d], rewards[d], nexts[d], done)

            if done:
                dist_blocks = self.game.frame_iteration * self.cfg["num_drones"]
                self.game.reset()
                self.game_cnt += 1
                self.agent.n_games += 1
                self.agent.train_long_memory()

                self.record = max(self.record, score)
                self.total_score += score
                mean = self.total_score / self.game_cnt
                self.dashboard.add_point(score, mean, dist_blocks)

                print(f"Epoch {self.game_cnt}/{self.cfg['epochs']} | " f"Fires: {score} | Best: {self.record}")

        self.after(40, self._train_loop)

    # Save trained AI model
    def _save_model(self):
        # Prompt user for customised AI model name to save
        name = simpledialog.askstring("Save AI Model", "Enter AI Model Name:", parent=self.master)

        if not name:
            return
        os.makedirs("model", exist_ok=True)
        stamp = name
        ext = (".npz" if self.cfg["algorithm"] == "Deep Q-learning"
               else ".pkl" if self.cfg["algorithm"] == "Q-learning"
               else ".pso")
        
        fname = f"{stamp}{ext}"
        self.agent.model.save(fname)

        session_name = f"Training_{fname}"
        rec = {"session": session_name, "type": "training", "metrics": {"scores": self.dashboard.scores, "means" : self.dashboard.means, "dists" : self.dashboard.dists}}
        _append_to_ledger(rec)

        messagebox.showinfo("Model saved", f"Saved to model/{fname}")

    # Save AI model and return back to startup home page
    def _save_and_back(self):
        self._save_model()
        self.parent.show_page("StartupPage")

    # Present user with popup upon completion of training
    def _training_done_popup(self):
        win = tk.Toplevel(self)
        win.transient(self.master)
        win.title("Training finished")
        win.grab_set()

        ttk.Label(win, text="Training Complete.", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, padx=15, pady=12)
        ttk.Button(win, text="Return to Home", command=lambda: (win.destroy(), self.parent.show_page("StartupPage"))).grid(row=1, column=0, padx=10, pady=10)
        ttk.Button(win, text="Save Trained Model", command=lambda: (win.destroy(), self._save_and_back())).grid(row=1, column=1, padx=10, pady=10)
        
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        x = (win.winfo_screenwidth() // 2) - (w // 2)
        y = (win.winfo_screenheight() // 2) - (h // 2)
        win.geometry(f"{w}x{h}+{x}+{y}")