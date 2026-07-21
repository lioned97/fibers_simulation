"""Desk for the phase-3 lens search: run it, and browse what it produced.

Run with ``py phase3_gui.py``.  Two tabs:

Run       shows how far a previous run got and what it finished last, then
          offers Continue (resume from the checkpoint) or Start over (discard
          it), with the live log.  A search takes hours, so the point is that
          an interrupted one never has to be repeated blindly.

Designs   lists every family pairing the search has exported, with its
          numbers, its ray trace and its STL paths, so the design that gets
          printed is chosen by a person looking at the optics rather than by
          whichever scalar happened to come out smallest.

Only the standard library, numpy and matplotlib are used, all of which the
project already depends on.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from method_export import METHODS_DIRNAME, list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
CHECKPOINT = os.path.join(FIGURES, "phase3_checkpoint.json")
METHODS = os.path.join(FIGURES, METHODS_DIRNAME)
FAMILIES = ("quadratic", "asphere", "biconic", "freeform")
TOTAL_PAIRS = len(FAMILIES)**2

INK = "#131920"
MUTED = "#5c6773"
ACCENT = "#2f5fc4"


def read_checkpoint():
    """Checkpoint contents, or None when there is nothing to resume."""
    if not os.path.isfile(CHECKPOINT):
        return None
    try:
        with open(CHECKPOINT, encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return {"unreadable": True}


def interpreter():
    """A console interpreter even when the GUI runs under pythonw."""
    path = sys.executable
    if os.path.basename(path).lower().startswith("pythonw"):
        candidate = os.path.join(os.path.dirname(path), "python.exe")
        if os.path.isfile(candidate):
            return candidate
    return path


class Phase3App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.pack(fill="both", expand=True)
        self.process = None
        self.log_queue = queue.Queue()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        self.run_tab = ttk.Frame(notebook, padding=10)
        self.designs_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.run_tab, text="  Run  ")
        notebook.add(self.designs_tab, text="  Designs  ")

        self._build_run_tab()
        self._build_designs_tab()
        self.refresh_status()
        self.refresh_designs()
        self.after(150, self._drain_log)

    # ---------------------------------------------------------------- run tab
    def _build_run_tab(self):
        status = ttk.LabelFrame(self.run_tab, text="Previous run", padding=10)
        status.pack(fill="x")

        self.status_label = ttk.Label(status, text="", font=("Segoe UI", 11, "bold"))
        self.status_label.pack(anchor="w")
        self.detail_label = ttk.Label(status, text="", foreground=MUTED,
                                      justify="left")
        self.detail_label.pack(anchor="w", pady=(2, 6))
        self.progress = ttk.Progressbar(status, maximum=TOTAL_PAIRS)
        self.progress.pack(fill="x")
        self.remaining_label = ttk.Label(status, text="", foreground=MUTED,
                                         wraplength=880, justify="left")
        self.remaining_label.pack(anchor="w", pady=(6, 0))

        buttons = ttk.Frame(self.run_tab)
        buttons.pack(fill="x", pady=10)
        self.continue_button = ttk.Button(buttons, text="Continue from checkpoint",
                                          command=self.continue_run)
        self.continue_button.pack(side="left")
        self.fresh_button = ttk.Button(buttons, text="Start over",
                                       command=self.start_over)
        self.fresh_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(buttons, text="Stop", state="disabled",
                                      command=self.stop_run)
        self.stop_button.pack(side="left")
        ttk.Button(buttons, text="Refresh",
                   command=self.refresh_status).pack(side="right")

        log_frame = ttk.LabelFrame(self.run_tab, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=16, wrap="none", font=("Consolas", 9),
                           background="#12161c", foreground="#d7dee7",
                           insertbackground="#d7dee7")
        bar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

    def refresh_status(self):
        data = read_checkpoint()
        if data is None:
            self.status_label.config(text="No previous run found")
            self.detail_label.config(
                text="Continue and Start over do the same thing right now: "
                     "a full search of all 16 family pairings.")
            self.progress.config(value=0)
            self.remaining_label.config(text="")
            self.continue_button.config(text="Run search")
            return
        if data.get("unreadable"):
            self.status_label.config(text="Checkpoint unreadable")
            self.detail_label.config(
                text="The file exists but could not be parsed. Start over "
                     "writes a new one.")
            self.progress.config(value=0)
            self.remaining_label.config(text="")
            return

        done_pairs = [tuple(pair) for pair in data.get("done", [])]
        done_labels = {f"{a} + {b}" for a, b in done_pairs}
        remaining = [f"{a} + {b}" for a in FAMILIES for b in FAMILIES
                     if f"{a} + {b}" not in done_labels]
        self.status_label.config(
            text=f"Stopped after {len(done_pairs)} of {TOTAL_PAIRS} pairings")
        last = data.get("last_completed") or "none yet"
        self.detail_label.config(
            text=f"Last finished: {last}\nCheckpoint saved: "
                 f"{data.get('updated_at', 'unknown')}")
        self.progress.config(value=len(done_pairs))
        self.remaining_label.config(
            text=("Still to do: " + ", ".join(remaining)) if remaining
            else "All pairings finished - re-running only redoes the final "
                 "selection.")
        self.continue_button.config(text="Continue from checkpoint")

    def continue_run(self):
        self._launch(fresh=False)

    def start_over(self):
        if os.path.isfile(CHECKPOINT):
            if not messagebox.askyesno(
                    "Start over",
                    "Discard the saved progress and search all 16 pairings "
                    "from the beginning?\n\nAlready-exported designs in the "
                    "methods folder are kept and will be overwritten as each "
                    "pairing finishes again."):
                return
            try:
                os.remove(CHECKPOINT)
            except OSError as exc:
                messagebox.showerror("Start over", f"Could not delete:\n{exc}")
                return
        self._launch(fresh=True)

    def _launch(self, fresh):
        if self.process is not None:
            messagebox.showinfo("Already running", "A search is already running.")
            return
        self.log.delete("1.0", "end")
        self._append(f"$ {interpreter()} phase3_optimize.py\n")
        try:
            self.process = subprocess.Popen(
                [interpreter(), "-u", "phase3_optimize.py"], cwd=HERE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except OSError as exc:
            self.process = None
            messagebox.showerror("Could not start", str(exc))
            return
        self.continue_button.config(state="disabled")
        self.fresh_button.config(state="disabled")
        self.stop_button.config(state="normal")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.process.stdout:
            self.log_queue.put(line)
        self.log_queue.put(None)

    def _drain_log(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line is None:
                    self._finished()
                else:
                    self._append(line)
                    # a pairing just landed: keep the browser current
                    if line.startswith("  saved "):
                        self.refresh_designs()
                        self.refresh_status()
        except queue.Empty:
            pass
        self.after(150, self._drain_log)

    def _finished(self):
        code = self.process.poll() if self.process else None
        self.process = None
        self.continue_button.config(state="normal")
        self.fresh_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self._append(f"\n--- finished (exit code {code}) ---\n")
        self.refresh_status()
        self.refresh_designs()

    def stop_run(self):
        if self.process is None:
            return
        if not messagebox.askyesno(
                "Stop", "Stop the search?\n\nFinished pairings stay in the "
                        "checkpoint, so Continue will pick up from there."):
            return
        self.process.terminate()
        self._append("\n--- stop requested ---\n")

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    # ------------------------------------------------------------ designs tab
    def _build_designs_tab(self):
        header = ttk.Frame(self.designs_tab)
        header.pack(fill="x")
        self.designs_label = ttk.Label(header, text="", font=("Segoe UI", 10))
        self.designs_label.pack(side="left")
        ttk.Button(header, text="Refresh",
                   command=self.refresh_designs).pack(side="right")
        ttk.Button(header, text="Open folder",
                   command=self.open_folder).pack(side="right", padx=6)

        body = ttk.PanedWindow(self.designs_tab, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(body)
        columns = ("sensitivity", "resolution", "photons", "gap")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings",
                                 height=14)
        self.tree.heading("#0", text="central + side")
        self.tree.column("#0", width=190, anchor="w")
        for name, title, width in (("sensitivity", "nT/sqrt(Hz)", 95),
                                   ("resolution", "res (um)", 80),
                                   ("photons", "photons/s", 100),
                                   ("gap", "gap (um)", 75)):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor="e")
        tree_bar = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_bar.set)
        tree_bar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        body.add(left, weight=1)

        right = ttk.Frame(body)
        self.detail = tk.Text(right, height=9, wrap="word", font=("Consolas", 9),
                              background="#f6f7f9", foreground=INK,
                              relief="flat", padx=8, pady=6)
        self.detail.pack(fill="x")
        self.figure = Figure(figsize=(7.4, 3.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        body.add(right, weight=3)

        self.methods = []

    def refresh_designs(self):
        self.methods = list_methods(METHODS)
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(self.methods):
            parameters = row.get("parameters", {})
            self.tree.insert(
                "", "end", iid=str(index), text=row.get("label", "?"),
                values=(f"{row.get('sensitivity_nt', float('nan')):.3f}",
                        f"{row.get('resolution_um', float('nan')):.3f}",
                        f"{row.get('photons_s', float('nan')):.3e}",
                        f"{parameters.get('air_gap_um', float('nan')):.0f}"))
        if self.methods:
            self.designs_label.config(
                text=f"{len(self.methods)} design(s), best first - "
                     "lower nT/sqrt(Hz) is better")
            self.tree.selection_set("0")
        else:
            self.designs_label.config(
                text="No designs yet. They appear here as the search finishes "
                     "each pairing.")
            self.detail.delete("1.0", "end")
            self.figure.clear()
            self.canvas.draw_idle()

    def on_select(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        row = self.methods[int(selection[0])]
        parameters = row.get("parameters", {})

        def line(key, value, unit=""):
            return f"{key:<28}{value}{unit}\n"

        text = line("family pairing", row.get("label", "?"))
        text += line("sensitivity", f"{row.get('sensitivity_nt', 0):.4g}",
                     " nT/sqrt(Hz)  (raw model)")
        text += line("signal RMS width",
                     f"{row.get('resolution_um', 0):.4g}", " um")
        text += line("photons into fiber", f"{row.get('photons_s', 0):.4g}", " /s")
        text += line("air gap / clearance",
                     f"{parameters.get('air_gap_um', 0):.1f} / "
                     f"{parameters.get('central_optical_clearance_um', 0):.1f}", " um")
        text += line("cap radius central/side",
                     f"{parameters.get('central_aperture_um', 0):.1f} / "
                     f"{parameters.get('side_aperture_um', 0):.1f}", " um")
        text += line("post height central/side",
                     f"{parameters.get('central_height_um', 0):.0f} / "
                     f"{parameters.get('side_height_um', 0):.0f}", " um")
        text += line("max slope central/side",
                     f"{row.get('central_max_slope', 0):.2f} / "
                     f"{row.get('side_max_slope', 0):.2f}")
        text += line("TIR loss central/side",
                     f"{100*row.get('central_tir_fraction', 0):.1f} / "
                     f"{100*row.get('side_tir_fraction', 0):.1f}", " %")
        text += line("max I / I_sat", f"{row.get('max_saturation', 0):.3g}")
        text += line("STL", row.get("full_stl", "-"))
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)

        self.figure.clear()
        picture = row.get("raytrace_png")
        axis = self.figure.add_subplot(111)
        axis.set_axis_off()
        path = os.path.join(row["directory"], picture) if picture else None
        if path and os.path.isfile(path):
            axis.imshow(mpimg.imread(path))
        else:
            axis.text(0.5, 0.5, "no ray-trace picture for this design",
                      ha="center", va="center", color=MUTED)
        self.figure.tight_layout(pad=0.2)
        self.canvas.draw_idle()

    def open_folder(self):
        selection = self.tree.selection()
        target = (self.methods[int(selection[0])]["directory"]
                  if selection and self.methods else METHODS)
        if not os.path.isdir(target):
            messagebox.showinfo("Open folder", f"Not created yet:\n{target}")
            return
        if sys.platform.startswith("win"):
            os.startfile(target)                                # noqa: S606
        else:
            subprocess.Popen(["xdg-open" if sys.platform.startswith("linux")
                              else "open", target])


def main():
    root = tk.Tk()
    root.title("NV-diamond probe - phase 3 lens search")
    root.geometry("1180x760")
    try:
        ttk.Style().theme_use("vista" if sys.platform.startswith("win") else "clam")
    except tk.TclError:
        pass
    Phase3App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
