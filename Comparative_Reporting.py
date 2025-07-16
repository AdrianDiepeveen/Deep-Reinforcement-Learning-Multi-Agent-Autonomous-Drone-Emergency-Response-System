"""
Comparative Reporting
"""
from __future__ import annotations
import json
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure   
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Reuse colours from Simulation Reporting
from Simulation_Reporting import GREEN_BG, GREEN_AX_BG, NEON, LIME, SPRING, FG

"""
Constants
"""
# Define path to ledger file containing comparative reporting data in json format
LEDGER = "Comparative_Reporting_Data.json"

# Define line colours for each figure
_COLOURS = (NEON, LIME, SPRING)

"""
Helper Functions
"""
# Function to load saved reporting metrics from ledger file containing comparative reporting data
def _load_ledger() -> list[dict]:
    if not os.path.isfile(LEDGER) or os.path.getsize(LEDGER) == 0:
        return []

    with open(LEDGER, "r", encoding="utf8") as f:
        return json.load(f)
        
# Cache sessions by name
_SESS = {d["session"]: d for d in _load_ledger()}

"""
Side Panel Class
"""
class _SidePanel(ttk.Frame):
    # SidePanel initialiser
    def __init__(self, master: ttk.Frame) -> None:
        super().__init__(master, padding=6)
        self.columnconfigure(0, weight=1)

        ttk.Label(self, text="Select Session:", font=("Arial", 11)).grid(row=0, column=0, sticky="w")

        # Create dropdown of available sessions
        self.combo = ttk.Combobox(self, state="readonly", values=list(_SESS.keys()))
        self.combo.bind("<<ComboboxSelected>>", self._on_pick)
        self.combo.grid(row=1, column=0, sticky="ew", pady=4)


        # Create matplotlib figure
        self.fig  = Figure(figsize=(6, 2.5), dpi=100)
        self.axs  = [self.fig.add_subplot(1, 3, i + 1) for i in range(3)]

        self._style_axes()

        # Embed figure in this panel
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self.rowconfigure(2, weight=1)

    # Reload ledger file and update session dropdown
    def refresh_sessions(self):
        data = _load_ledger()
        global _SESS
        _SESS.clear()
        _SESS.update({d["session"]: d for d in data})
        vals = list(_SESS.keys())
        self.combo['values'] = vals
        self.combo.set('')

    # Apply background and styling to axes
    def _style_axes(self):
        self.fig.patch.set_facecolor(GREEN_BG)
        for ax in self.axs:
            ax.set_facecolor(GREEN_AX_BG)
            ax.tick_params(colors=FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(FG)

    # Handle session selection
    def _on_pick(self, _ev):
        name = self.combo.get()
        rec  = _SESS.get(name)
        if not rec:
            return

        # Clear and restyle axis before plotting new data
        for ax in self.axs:
            ax.clear()
            ax.set_facecolor(GREEN_AX_BG)
            ax.tick_params(colors=FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(FG)

        # Determine which metrics to plot based on session type
        if rec["type"] == "training":
            d = rec["metrics"]

            self._plot(0, d["scores"], "Fires Per Epoch")
            self._plot(1, d["means"] , "Avg Fires")
            self._plot(2, d["dists"] , "Distance (Blocks)")

            self.fig.suptitle(f"Training – {name}", fontsize=9, color=FG)

        else:  
            d = rec["metrics"]

            self._plot(0, d["fires"], "Total Fires")
            self._plot(1, d["avg"],   "Avg Fires Per Collision")
            self._plot(2, d["dists"], "Total Distance (Blocks)")

            self.fig.suptitle(f"Simulation – {name}", fontsize=9, color=FG)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # Plot on axis
    def _plot(self, idx: int, arr: list, label: str):
        ax = self.axs[idx]
        ax.plot(range(len(arr)), arr, lw=2, color=_COLOURS[idx])
        ax.set_title(label, fontsize=8, color=FG)


"""
Comparative Reporting Class
"""
class ComparativeReportingPage(ttk.Frame):
    # ComparativeReportingPage initialiser
    def __init__(self, parent, root_app):
        super().__init__(parent)

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # Create header with back button and page title
        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        # Unicode for back arrow
        ttk.Button(hdr, text="\u2190 Home", command=lambda: root_app.show_page("StartupPage")).pack(side="left", padx=6, pady=6)
        ttk.Label(hdr, text="Comparative Reporting", font=("Arial", 15)).pack(side="left", padx=4, pady=6)

        body = ttk.Frame(self)
        body.pack(expand=True, fill="both")

        # Body containing two side panels separated by a line
        body.rowconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        body.columnconfigure(0, weight=1)

        self.top = _SidePanel(body)
        self.bottom = _SidePanel(body)

        self.top.grid(row=0, column=0, sticky="nsew")
        self.bottom.grid(row=2, column=0, sticky="nsew")

        ttk.Separator(body, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=4)
        
    # Refresh both panels
    def on_show(self):
        self.top.refresh_sessions()
        self.bottom.refresh_sessions()