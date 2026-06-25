# ---------------------------------
# Library imports
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import shapefile
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog, messagebox
from tqdm import tqdm
import logging
import sys, warnings, atexit
import os
import argparse
import glob
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from sklearn.cluster import MiniBatchKMeans
from logging.handlers import RotatingFileHandler
from InputJsonApi import run_pipeline_for_facilityid
from InputJsonApi import (
    extractMeterData_json,
    extractLineData_json,
    extractServiceLines_json,
    extractMVLineData_json,
    auto_determine_snap_tolerance,
    get_transformer_capacity_from_json,
    build_line_length_map_from_json,
    get_transformer_xy_from_json,
    build_phase_indexer_from_json
)
import pyproj
import json
from geojson import Feature, Point, FeatureCollection

# ---------------------------------
# 2) Global variables
meterData = None
lvData = None
mvData = None
transformerData = None
eserviceData = None
meterLocations = None
initialVoltages = None
totalLoads = None
phase_loads = None
peano = None
phases = None
lvLinesX = None
lvLinesY = None
lvLines = None
mvLinesX = None
mvLinesY = None
mvLines = None
filteredEserviceLines = None
initialTransformerLocation = None
latest_split_result = None     
SNAP_TOLERANCE = 0.005
IS_GUI = False   # default: รันแบบ headless
reopt_btn = None # กัน NameError ตอน headless
_LOG_CONFIGURED = False




_TEE_FILE = None  # ใกล้ ๆ global อื่น

def setup_logging(log_path, to_console=False, level=logging.INFO, tee_stdout_to_file=False):
    """
    สร้าง root logger ที่เขียนลงไฟล์ + (ออปชัน) แสดงบน console และ tee print() ลงไฟล์ด้วย
    """
    logger = logging.getLogger()

    # --- เคลียร์ handler เก่าก่อน ---
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # --- ปิด tee เดิม + รีเซ็ต stdout/stderr ถ้ามี ---
    global _TEE_FILE
    if _TEE_FILE:
        try:
            _TEE_FILE.close()
        except Exception:
            pass
        _TEE_FILE = None
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    if to_console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # --- tee stdout/stderr ลงไฟล์ log ---
    if tee_stdout_to_file:
        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass
            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        _TEE_FILE = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _TEE_FILE)
        sys.stderr = _Tee(sys.__stderr__, _TEE_FILE)

        def _close_tee_file():
            global _TEE_FILE
            try:
                if _TEE_FILE and not _TEE_FILE.closed:
                    _TEE_FILE.close()
            except Exception:
                pass

        atexit.register(_close_tee_file)

    atexit.register(logging.shutdown)
    return logger



# ---------------------------------
# 3) Classes: TextHandler, TkProgress, EdgeNavigatorDialog
class TextHandler(logging.Handler):
    """A custom logging handler that sends log messages to a Tkinter Text widget."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        # Insert text from the GUI thread via 'after'
        self.text_widget.after(0, self._insert_message, msg)

    def _insert_message(self, msg):
        self.text_widget.insert(tk.END, msg + "\n")
        self.text_widget.see(tk.END)

class TkProgress:
    """
    Progress-bar แบบ determinate + แสดงชื่อขั้นตอนและเปอร์เซ็นต์
    """
    def __init__(self, bar_widget: ttk.Progressbar, label_widget: tk.Label):
        self.bar   = bar_widget
        self.label = label_widget
        self.total = 1
        self.cur   = 0
        self.stage = ""

    # ---------- public API ----------
    def start(self, total: int, stage: str = ""):
        self.total = max(1, total)
        self.cur   = 0
        self.stage = stage
        self.bar['maximum'] = self.total
        self.bar['value']   = 0
        self._show()

    def step(self, n: int = 1):
        self.cur = min(self.cur + n, self.total)
        self._show()

    def finish(self, stage_done: str = "Done"):
        self.cur   = self.total
        self.stage = stage_done
        self._show()

    # ---------- internal ----------
    def _show(self):
        pct = int(100 * self.cur / self.total)
        self.label.config(text=f"{self.stage}  {pct:3d}%")
        self.bar['value'] = self.cur
        self.bar.update_idletasks()
        self.label.update_idletasks()



class EdgeNavigatorDialog(tk.Toplevel):
    """
    แสดงตาราง candidate-edges (edge_diffs_df) และปุ่ม Prev/Next พร้อม embedded preview
    เพื่อให้ผู้ใช้เลื่อนไปมาดู edge แต่ละเส้นได้ง่ายขึ้นโดยไม่ต้องเปิดหน้าต่างใหม่
    """
    def __init__(self, master, G, coord_mapping,
                 edge_df,
                 lvLinesX, lvLinesY,
                 mvLinesX, mvLinesY,
                 meterLocs,
                 start_idx=0, **kw):
        super().__init__(master, **kw)
        self.G         = G
        self.coord     = coord_mapping
        self.edge_df   = edge_df.reset_index()  # ดึง splitting_index มาเป็นคอลัมน์
        self.lvLinesX  = lvLinesX
        self.lvLinesY  = lvLinesY
        self.mvLinesX  = mvLinesX
        self.mvLinesY  = mvLinesY
        self.meterLocs = meterLocs
        self.result    = None
        self.curr_idx  = start_idx
        self.protocol("WM_DELETE_WINDOW", self.cancel)  

        self.title("เลือกจุดตัดจ่ายใหม่")
        self.geometry("640x600")

        # --- 1) Treeview + binding ---
        columns = ("splitting_index", "Edge_Diff", "Load_G1", "Load_G2")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=10)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for col, txt in [
            ("splitting_index", "Index"),
            ("Edge_Diff",    "ΔLoad (kW)"),
            ("Load_G1",      "Load G1 (kW)"),
            ("Load_G2",      "Load G2 (kW)"),
        ]:
             self.tree.heading(col, text=txt)
            
        for _, row in self.edge_df.iterrows():
            self.tree.insert("", tk.END, values=(
                row["splitting_index"],
                f"{row['Edge_Diff']:.1f}",
                f"{row['Load_G1']:.1f}",
                f"{row['Load_G2']:.1f}"
            ))
        # เลือก start_idx
        kids = self.tree.get_children()
        if kids:
            self.tree.selection_set(kids[self.curr_idx])
            self.tree.see(kids[self.curr_idx])

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # --- 2) Status label ---
        self.status_lbl = tk.Label(self, anchor="w")
        self.status_lbl.pack(fill=tk.X, padx=5)
        self._update_status()

        # --- 3) Navigation buttons ---
        nav_frm = tk.Frame(self)
        nav_frm.pack(fill=tk.X, pady=5)
        btn_prev   = tk.Button(nav_frm, text="◀ Prev", command=self.prev_edge, width=10)
        btn_next   = tk.Button(nav_frm, text="Next ▶", command=self.next_edge, width=10)
        btn_full   = tk.Button(nav_frm, text="Preview Full Map",command=self.show_full_map,  width=15)
        btn_ok     = tk.Button(nav_frm, text="Use this edge", command=self.use_current_edge)
        btn_cancel = tk.Button(nav_frm, text="End process", command=self.cancel)
        for b in (btn_prev, btn_next, btn_full, btn_ok, btn_cancel):
            b.pack(side=tk.LEFT, padx=3)


        # --- 4) Embedded matplotlib preview ---
        self.fig, self.ax = plt.subplots(figsize=(5,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._update_preview()

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(self.tree.item(sel[0])["values"][0])
        if idx != self.curr_idx:
            self.curr_idx = idx
            self._update_status()
            self._update_preview()

    def _update_status(self):
        diff = self.edge_df.iloc[self.curr_idx]["Edge_Diff"]
        self.status_lbl.config(text=f"Candidate #{self.curr_idx}  |  ΔLoad = {diff:.1f} kW")

    def _update_preview(self):
        """วาด preview แบบเร็ว ไม่สร้าง legend ซ้ำ และไม่เรียก plt ระดับโมดูล"""
        self.ax.clear()

        # ==== ใช้ LineCollection เร็วกว่า plot เป็นเส้น-ต่อ-เส้น ====
        from matplotlib.collections import LineCollection
        if self.lvLinesX:
            lv_coll = LineCollection(
                [list(zip(x, y)) for x, y in zip(self.lvLinesX, self.lvLinesY)],
                colors='lime', linewidths=.8, linestyles='--')
            self.ax.add_collection(lv_coll)
        if self.mvLinesX:
            mv_coll = LineCollection(
                [list(zip(x, y)) for x, y in zip(self.mvLinesX, self.mvLinesY)],
                colors='maroon', linewidths=.8, linestyles='-.')
            self.ax.add_collection(mv_coll)

        # meter จุดดำ
        self.ax.plot(self.meterLocs[:, 0], self.meterLocs[:, 1], 'k.', ms=3)

        # ไฮไลต์ edge ปัจจุบัน
        u, v = self.edge_df.iloc[self.curr_idx]['Edge']
        x1, y1 = self.coord[u];  x2, y2 = self.coord[v]
        self.ax.plot([x1, x2], [y1, y2], 'r-', lw=2.5)

        self.ax.set_aspect('equal')
        self.ax.axis()          # ลด overhead legend/กรอบ
        self.canvas.draw_idle()

    def prev_edge(self):
        if self.curr_idx > 0:
            self.curr_idx -= 1
            kid = self.tree.get_children()[self.curr_idx]
            self.tree.selection_set(kid)
            self.tree.see(kid)
            self._update_status()
            self._update_preview()

    def next_edge(self):
        if self.curr_idx < len(self.edge_df) - 1:
            self.curr_idx += 1
            kid = self.tree.get_children()[self.curr_idx]
            self.tree.selection_set(kid)
            self.tree.see(kid)
            self._update_status()
            self._update_preview()

    def use_current_edge(self):
        self.result = self.curr_idx
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()
    
    def __del__(self):
        try:
            plt.close(self.fig)  # ปิด embedded figure
        except:
            pass
    
    def destroy(self):
    
        try:
            # ใช้เพียงคำสั่งเดียวเพื่อปิดทุก figure รวมถึง self.fig
            plt.close('all')
        except Exception as e:
            logging.warning(f"Error closing matplotlib figures: {e}")
        
        # เรียก Method destroy ของคลาสแม่
        super().destroy()
         

    def show_full_map(self):
        """เปิดหน้าต่าง matplotlib ขนาดใหญ่พร้อมชั้นข้อมูลทั้งหมด
        และไฮไลต์ candidate‑edge ปัจจุบันเป็นเส้นสีแดง
        """
        # สร้างหน้าต่าง Toplevel ใหม่สำหรับแสดง full map
        map_window = tk.Toplevel(self)
        map_window.title(f"Full Map - Candidate #{self.curr_idx}")
        map_window.geometry("1000x800")  # ขนาดเริ่มต้น
        
        # สร้าง main frame ที่จะบรรจุ canvas และ toolbar
        main_frame = tk.Frame(map_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # สร้าง figure และ canvas สำหรับ full map
        full_fig = plt.Figure(figsize=(10, 8), dpi=100)
        full_ax = full_fig.add_subplot(111)
        
        # สร้าง FigureCanvasTkAgg สำหรับแสดงใน Toplevel
        canvas = FigureCanvasTkAgg(full_fig, master=main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # เพิ่ม NavigationToolbar2Tk เพื่อให้สามารถ zoom, pan, save ได้
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, main_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # วาดเส้น LV & MV
        for x, y in zip(self.lvLinesX, self.lvLinesY):
            full_ax.plot(x, y, color='lime', linewidth=1, linestyle='--', label='LV Line' if 'LV Line' not in full_ax.get_legend_handles_labels()[1] else "")
        for x, y in zip(self.mvLinesX, self.mvLinesY):
            full_ax.plot(x, y, color='maroon', linewidth=1, linestyle='-.', label='MV Line' if 'MV Line' not in full_ax.get_legend_handles_labels()[1] else "")

        # วาดตำแหน่งมิเตอร์ทั้งหมด
        full_ax.plot(self.meterLocs[:, 0], self.meterLocs[:, 1],
                'k.', markersize=3, label='Meter')

        # ไฮไลต์ edge ที่เลือก
        edge = self.edge_df.iloc[self.curr_idx]['Edge']
        n1, n2 = edge
        x1, y1 = self.coord[n1]
        x2, y2 = self.coord[n2]
        full_ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Candidate Edge')

        full_ax.set_aspect('equal')
        full_ax.set_title(f"Candidate #{self.curr_idx}  |  ΔLoad = {self.edge_df.iloc[self.curr_idx]['Edge_Diff']:.1f} kW")
        full_ax.legend()
        full_ax.grid(True)
        
        # ปรับปรุง layout และแสดงผล
        full_fig.tight_layout()
        canvas.draw()
        
        # สร้างตัวแปรสำหรับเก็บสถานะการ pan
        pan_enabled = False
        pan_start = None
        
        # Event เมื่อคลิกเมาส์
        def on_button_press(event):
            nonlocal pan_enabled, pan_start
            
            # ต้องมีการระบุพิกัด x, y ที่ถูกต้อง
            if event.xdata is not None and event.ydata is not None:
                if event.button == 1:  # left click
                    if event.dblclick:
                        # ดับเบิลคลิกเพื่อรีเซ็ตมุมมอง
                        full_ax.set_xlim(auto=True)
                        full_ax.set_ylim(auto=True)
                        canvas.draw()
                    else:
                        # เริ่ม pan
                        pan_enabled = True
                        pan_start = (event.xdata, event.ydata)
                elif event.button == 3:  # right click - ยกเลิก pan
                    pan_enabled = False
                    pan_start = None
        
        # Event เมื่อปล่อยเมาส์
        def on_button_release(event):
            nonlocal pan_enabled, pan_start
            if event.button == 1:  # left click
                pan_enabled = False
                pan_start = None
        
        # Event เมื่อเลื่อนเมาส์
        def on_mouse_move(event):
            nonlocal pan_enabled, pan_start
            
            if pan_enabled and pan_start is not None:
                # ต้องมีการระบุพิกัด x, y ที่ถูกต้อง
                if event.xdata is not None and event.ydata is not None:
                    # คำนวณระยะทางที่เลื่อน
                    dx = event.xdata - pan_start[0]
                    dy = event.ydata - pan_start[1]
                    
                    # ดึงขอบเขตปัจจุบัน
                    xmin, xmax = full_ax.get_xlim()
                    ymin, ymax = full_ax.get_ylim()
                    
                    # กำหนดขอบเขตใหม่
                    full_ax.set_xlim(xmin - dx, xmax - dx)
                    full_ax.set_ylim(ymin - dy, ymax - dy)
                    
                    # วาดใหม่
                    canvas.draw()
                    
                    # อัปเดตจุดเริ่มต้น
                    pan_start = (event.xdata, event.ydata)
        
        # Event เมื่อใช้ scroll wheel สำหรับการ zoom
        def on_scroll(event):
            # ทิศทางของ scroll (ขึ้นหรือลง)
            if event.button == 'up':
                scale_factor = 0.9  # zoom in
            else:
                scale_factor = 1.1  # zoom out
                
            # ต้องมีการระบุพิกัด x, y ที่ถูกต้อง
            if event.xdata is not None and event.ydata is not None:
                # ดึงขอบเขตปัจจุบัน
                xmin, xmax = full_ax.get_xlim()
                ymin, ymax = full_ax.get_ylim()
                
                # คำนวณขอบเขตใหม่
                xmin = event.xdata - (event.xdata - xmin) * scale_factor
                xmax = event.xdata + (xmax - event.xdata) * scale_factor
                ymin = event.ydata - (event.ydata - ymin) * scale_factor
                ymax = event.ydata + (ymax - event.ydata) * scale_factor
                
                # กำหนดขอบเขตใหม่
                full_ax.set_xlim(xmin, xmax)
                full_ax.set_ylim(ymin, ymax)
                
                # วาดใหม่
                canvas.draw()
        
        # ลงทะเบียน event handlers
        canvas.mpl_connect('button_press_event', on_button_press)
        canvas.mpl_connect('button_release_event', on_button_release)
        canvas.mpl_connect('motion_notify_event', on_mouse_move)
        canvas.mpl_connect('scroll_event', on_scroll)
        
        # เพิ่มปุ่มปิดและปุ่มรีเซ็ตมุมมอง
        button_frame = tk.Frame(map_window)
        button_frame.pack(pady=5)
        
        # ปุ่มรีเซ็ตมุมมอง
        def reset_view():
            full_ax.set_xlim(auto=True)
            full_ax.set_ylim(auto=True)
            full_ax.set_aspect('equal')
            canvas.draw()
        
        reset_btn = tk.Button(button_frame, text="Reset View", command=reset_view)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # ปุ่มปิด
        close_btn = tk.Button(button_frame, text="Close", command=map_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        # ปรับให้หน้าต่างขยายได้
        map_window.protocol("WM_DELETE_WINDOW", map_window.destroy)
        
        # พยายามทำให้หน้าต่างเต็มจอ
        map_window.state('zoomed')  # Windows
        try:
            # MacOS / Linux
            map_window.attributes('-zoomed', True)
        except:
            pass
        
        # ทำให้หน้าต่างเป็น modal dialog
        map_window.transient(self)
        map_window.grab_set()
        self.wait_window(map_window)

# ---------------------------------
# 4) Filters
class SummaryFilter(logging.Filter):
    """Only allow console messages that match certain keywords (optional)."""
    def filter(self, record):
        msg = record.getMessage()
        keywords = [
            "All shapefiles loaded successfully.",
            "Calculate LoadCenter each Group...",
            "Optimizing Transformer Location each Group...",
            "Finding splitting point by load balance difference on edges...",
            "Result",
            "Loss Report",
            "Exporting",
            "Group 1 => Load", 
            "Group 2 => Load", 
            "Plotting",
            "CSV saved",
            "Enter new splitting point candidate index to re-run",
            "Initial processing",
            "Program finished successfully.",
            "Re-executing post-process steps with new splitting candidate index:",
            " => Load="
            
        ]
        return any(kw in msg for kw in keywords)
# ---------------------------------
# 4.5) Utilities
def ensure_folder_exists(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logging.error(f"Cannot create folder '{path}': {e}")    
# ---------------------------------
# 5) loadShapefiles
def loadShapefiles(parent):
    """Load the shapefiles when called by the UI."""
    global meterData, lvData, mvData, transformerData, eserviceData

    # Load meter shapefile
    meterFileName = askopenfilename(
        title="Select the Meter Shapefile",
        filetypes=[("Shapefile", "*.shp")],
        parent=parent
    )
    if not meterFileName:
        logging.error("Meter shapefile not selected.")
        return
    meterData = shapefile.Reader(meterFileName, encoding='cp874')
    logging.info(f"Meter shapefile selected: {meterFileName}")

    # Load LV Conductor shapefile
    lvFileName = askopenfilename(
        title="Select the LV Conductor Shapefile",
        filetypes=[("Shapefile", "*.shp")],
        parent=parent
    )
    if not lvFileName:
        logging.error("LV Conductor shapefile not selected.")
        return
    lvData = shapefile.Reader(lvFileName, encoding='utf-8')
    logging.info(f"LV Conductor shapefile selected: {lvFileName}")

    # Load MV Conductor shapefile
    mvFileName = askopenfilename(
        title="Select the MV Conductor Shapefile",
        filetypes=[("Shapefile", "*.shp")],
        parent=parent
    )
    if not mvFileName:
        logging.error("MV Conductor shapefile not selected.")
        return
    mvData = shapefile.Reader(mvFileName, encoding='utf-8')
    logging.info(f"MV Conductor shapefile selected: {mvFileName}")

    # Load Eservice Line shapefile
    eserviceFileName = askopenfilename(
        title="Select the Eservice Line Shapefile",
        filetypes=[("Shapefile", "*.shp")],
        parent=parent
    )
    if not eserviceFileName:
        logging.error("Eservice Line shapefile not selected.")
        return
    eserviceData = shapefile.Reader(eserviceFileName, encoding='utf-8')
    logging.info(f"Eservice Line shapefile selected: {eserviceFileName}")

    # Load Transformer shapefile
    transformerFileName = askopenfilename(
        title="Select the Transformer Shapefile",
        filetypes=[("Shapefile", "*.shp")],
        parent=parent
    )
    if not transformerFileName:
        logging.error("Transformer shapefile not selected.")
        return
    transformerData = shapefile.Reader(transformerFileName, encoding='cp874')
    logging.info(f"Transformer shapefile selected: {transformerFileName}")
    logger = logging.getLogger()
    
    # ลบ FileHandler ที่มีอยู่แล้วออก
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    base_name = os.path.splitext(os.path.basename(transformerFileName))[0]
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    log_filename = os.path.join(f"{folder_path}", f"Optimization_{base_name}_log.txt")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logging.info(f"Logger file handler updated: {log_filename}")

    logging.info("All shapefiles loaded successfully.")

# ---------------------------------
# 6) snapping & tolerance utility
def snap_coordinates_to_tolerance(coords_list, tolerance=0.1):
    """
    Snap coordinates ที่อยู่ใกล้กันภายในระยะ tolerance ให้ใช้พิกัดเดียวกัน
    
    Args:
        coords_list: list of (x, y) coordinates
        tolerance: ระยะทางที่ถือว่าเป็นจุดเดียวกัน (หน่วยเดียวกับพิกัด)
    
    Returns:
        dict: mapping จาก original coordinate ไป snapped coordinate
    """
    if not coords_list:
        return {}
    
    # แปลงเป็น numpy array
    coords_array = np.array(coords_list)
    
    # สร้าง KDTree สำหรับการหาจุดใกล้เคียง
    tree = cKDTree(coords_array)
    
    # หาคู่ของจุดที่อยู่ใกล้กันภายใน tolerance
    pairs = tree.query_pairs(tolerance)
    
    # สร้าง mapping จาก index ไป representative coordinate
    coord_mapping = {}
    processed = set()
    
    for i, coord in enumerate(coords_list):
        if i in processed:
            continue
            
        # หาจุดทั้งหมดที่อยู่ใกล้กับจุดนี้
        nearby_indices = [i]
        for pair in pairs:
            if i in pair:
                other_idx = pair[1] if pair[0] == i else pair[0]
                if other_idx not in processed:
                    nearby_indices.append(other_idx)
        
        # ใช้จุดแรกเป็น representative
        representative = coords_list[i]
        
        # map ทุกจุดในกลุ่มไปยัง representative
        for idx in nearby_indices:
            coord_mapping[coords_list[idx]] = representative
            processed.add(idx)
    
    # สำหรับจุดที่ไม่มีจุดใกล้เคียง
    for i, coord in enumerate(coords_list):
        if coord not in coord_mapping:
            coord_mapping[coord] = coord
    
    return coord_mapping

def find_optimal_tolerance(coords_list, min_tolerance=0.0000001, max_tolerance=0.001, 
                          target_reduction_ratio=0.95, max_iterations=20):
    """
    หาค่า tolerance ที่เหมาะสมสำหรับ coordinate snapping โดยใช้ binary search
    
    Args:
        coords_list: list of (x, y) coordinates
        min_tolerance: ค่า tolerance ต่ำสุดที่ต้องการทดสอบ (default: 0.0000001)
        max_tolerance: ค่า tolerance สูงสุดที่ต้องการทดสอบ (default: 0.001)
        target_reduction_ratio: อัตราส่วนของจำนวนพิกัดที่ต้องการหลังจาก snap (default: 0.95)
        max_iterations: จำนวนรอบสูงสุดในการค้นหา (default: 20)
    
    Returns:
        float: ค่า tolerance ที่เหมาะสม
    """
    
    if not coords_list:
        logging.warning("Empty coordinates list provided to find_optimal_tolerance")
        return min_tolerance
    
    # แปลงเป็น set เพื่อกำจัดพิกัดที่ซ้ำกัน
    unique_coords = list(set(coords_list))
    original_count = len(unique_coords)
    
    logging.info(f"Finding optimal tolerance for {original_count} unique coordinates")
    logging.info(f"Target reduction ratio: {target_reduction_ratio}")
    
    # ถ้ามีพิกัดน้อย ใช้ค่า tolerance ต่ำ
    if original_count < 100:
        logging.info(f"Small dataset ({original_count} points), using minimum tolerance")
        return min_tolerance
    
    # คำนวณจำนวนพิกัดเป้าหมายหลังจาก snap
    target_count = int(original_count * target_reduction_ratio)
    
    # Binary search หาค่า tolerance ที่เหมาะสม
    low = min_tolerance
    high = max_tolerance
    best_tolerance = min_tolerance
    best_count = original_count
    
    iteration = 0
    while iteration < max_iterations and (high - low) > min_tolerance:
        mid_tolerance = (low + high) / 2
        
        # ทดสอบ snap ด้วย tolerance ปัจจุบัน
        snap_map = snap_coordinates_to_tolerance(unique_coords, mid_tolerance)
        snapped_coords = list(set(snap_map.values()))
        snapped_count = len(snapped_coords)
        
        reduction_ratio = snapped_count / original_count
        
        logging.debug(f"Iteration {iteration}: tolerance={mid_tolerance:.8f}, "
                     f"snapped_count={snapped_count}, ratio={reduction_ratio:.3f}")
        
        # ตรวจสอบว่าใกล้เคียงกับเป้าหมายหรือไม่
        if abs(reduction_ratio - target_reduction_ratio) < 0.01:
            best_tolerance = mid_tolerance
            best_count = snapped_count
            break
        
        # ปรับช่วงการค้นหา
        if reduction_ratio > target_reduction_ratio:
            # ยังมีจุดมากเกินไป, เพิ่ม tolerance
            low = mid_tolerance
        else:
            # จุดน้อยเกินไป, ลด tolerance
            high = mid_tolerance
            
        # เก็บค่าที่ดีที่สุด
        if abs(reduction_ratio - target_reduction_ratio) < abs(best_count/original_count - target_reduction_ratio):
            best_tolerance = mid_tolerance
            best_count = snapped_count
            
        iteration += 1
    
    # ตรวจสอบผลลัพธ์
    final_snap_map = snap_coordinates_to_tolerance(unique_coords, best_tolerance)
    final_snapped_coords = list(set(final_snap_map.values()))
    final_count = len(final_snapped_coords)
    final_ratio = final_count / original_count
    
    logging.info(f"Optimal tolerance found: {best_tolerance:.8f}")
    logging.info(f"Original coordinates: {original_count}, Snapped coordinates: {final_count}")
    logging.info(f"Reduction ratio: {final_ratio:.3f}")
    
    # คำนวณระยะทางเฉลี่ยระหว่างจุดที่ถูก snap
    total_distance = 0
    snap_count = 0
    for orig_coord in unique_coords:
        snapped_coord = final_snap_map[orig_coord]
        if orig_coord != snapped_coord:
            distance = np.hypot(orig_coord[0] - snapped_coord[0], 
                               orig_coord[1] - snapped_coord[1])
            total_distance += distance
            snap_count += 1
    
    if snap_count > 0:
        avg_snap_distance = total_distance / snap_count
        logging.info(f"Average snap distance: {avg_snap_distance:.6f} units")
    
    return best_tolerance


def analyze_coordinate_distribution(coords_list):
    """
    วิเคราะห์การกระจายตัวของพิกัดเพื่อช่วยกำหนด tolerance
    
    Args:
        coords_list: list of (x, y) coordinates
    
    Returns:
        dict: ข้อมูลสถิติของการกระจายตัวพิกัด
    """
    if not coords_list:
        return {}
    
    coords_array = np.array(list(set(coords_list)))
    
    # คำนวณระยะทางขั้นต่ำระหว่างจุด
    if len(coords_array) > 1:
        tree = cKDTree(coords_array)
        distances, _ = tree.query(coords_array, k=2)  # k=2 เพื่อหาจุดที่ใกล้ที่สุด (ไม่รวมตัวเอง)
        min_distances = distances[:, 1]  # ระยะทางไปยังจุดที่ใกล้ที่สุด
        
        stats = {
            'count': len(coords_array),
            'min_distance': np.min(min_distances),
            'max_distance': np.max(min_distances),
            'mean_distance': np.mean(min_distances),
            'median_distance': np.median(min_distances),
            'std_distance': np.std(min_distances),
            'percentile_10': np.percentile(min_distances, 10),
            'percentile_25': np.percentile(min_distances, 25),
            'percentile_50': np.percentile(min_distances, 50),
            'percentile_75': np.percentile(min_distances, 75),
            'percentile_90': np.percentile(min_distances, 90)
        }
        
        # แนะนำ tolerance ตามการกระจายตัว
        suggested_tolerance = stats['percentile_10'] / 2  # ใช้ครึ่งหนึ่งของ percentile ที่ 10
        stats['suggested_tolerance'] = suggested_tolerance
        
        logging.info("Coordinate distribution analysis:")
        logging.info(f"  Total points: {stats['count']}")
        logging.info(f"  Min nearest distance: {stats['min_distance']:.8f}")
        logging.info(f"  Mean nearest distance: {stats['mean_distance']:.8f}")
        logging.info(f"  Suggested tolerance: {stats['suggested_tolerance']:.8f}")
        
        return stats
    else:
        return {'count': len(coords_array)}


def auto_determine_snap_tolerance(meter_locations, lv_lines, mv_lines, 
                                 reduction_ratio=0.98, use_analysis=True):
    """
    กำหนดค่า snap tolerance อัตโนมัติตามข้อมูลที่มี
    
    Args:
        meter_locations: array ของตำแหน่งมิเตอร์
        lv_lines: list ของ LV lines
        mv_lines: list ของ MV lines
        reduction_ratio: อัตราส่วนการลดจำนวนพิกัด (default: 0.98)
        use_analysis: ใช้การวิเคราะห์การกระจายตัวหรือไม่ (default: True)
    
    Returns:
        float: ค่า snap tolerance ที่เหมาะสม
    """
    # รวบรวมพิกัดทั้งหมด
    all_coords = []
    
    # จากมิเตอร์
    if meter_locations is not None and len(meter_locations) > 0:
        all_coords.extend([(loc[0], loc[1]) for loc in meter_locations])
    
    # จาก LV lines
    for line in lv_lines:
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        all_coords.extend(coords)
    
    # จาก MV lines
    for line in mv_lines:
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        all_coords.extend(coords)
    
    if not all_coords:
        logging.warning("No coordinates found for tolerance calculation")
        return 0.000002  # ค่า default
    
    # วิเคราะห์การกระจายตัว
    if use_analysis:
        stats = analyze_coordinate_distribution(all_coords)
        if 'suggested_tolerance' in stats:
            # ใช้ค่าที่แนะนำจากการวิเคราะห์เป็นจุดเริ่มต้น
            min_tol = stats['suggested_tolerance'] / 10
            max_tol = stats['suggested_tolerance'] * 10
        else:
            min_tol = 0.0000001
            max_tol = 0.001
    else:
        min_tol = 0.0000001
        max_tol = 0.001
    
    # หาค่า tolerance ที่เหมาะสม
    optimal_tolerance = find_optimal_tolerance(
        all_coords, 
        min_tolerance=min_tol,
        max_tolerance=max_tol,
        target_reduction_ratio=reduction_ratio
    )
    
    return optimal_tolerance

# ---------------------------------
# 7) line-fault detection & fixing
def identify_failed_snap_lines(lines, snap_tolerance, snap_map=None):
    """
    ระบุสายที่อาจมีปัญหาในการ snap (เช่น สายที่สั้นกว่า tolerance หรือมีจุดต่อที่ไม่ตรงกัน)
    
    Args:
        lines: list of line dictionaries with 'X' and 'Y' coordinates
        snap_tolerance: ค่า tolerance ที่ใช้ในการ snap
        snap_map: coordinate mapping จาก snap_coordinates_to_tolerance (optional)
    
    Returns:
        dict: ข้อมูลสายที่มีปัญหา
    """
    failed_lines = []
    short_lines = []
    isolated_lines = []
    problematic_connections = []
    
    # ถ้าไม่มี snap_map ให้สร้างใหม่
    if snap_map is None:
        all_coords = []
        for line in lines:
            coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                      if not (np.isnan(x) or np.isnan(y))]
            all_coords.extend(coords)
        snap_map = snap_coordinates_to_tolerance(list(set(all_coords)), snap_tolerance)
    
    # วิเคราะห์แต่ละสาย
    for idx, line in enumerate(lines):
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        
        if len(coords) < 2:
            failed_lines.append({
                'index': idx,
                'type': 'invalid',
                'reason': 'Less than 2 valid coordinates',
                'line': line
            })
            continue
        
        # ตรวจสอบความยาวของสาย
        total_length = 0
        for i in range(len(coords) - 1):
            dist = np.hypot(coords[i][0] - coords[i+1][0], 
                           coords[i][1] - coords[i+1][1])
            total_length += dist
        
        # สายสั้นเกินไป
        if total_length < snap_tolerance * 2:
            short_lines.append({
                'index': idx,
                'length': total_length,
                'tolerance': snap_tolerance,
                'start': coords[0],
                'end': coords[-1],
                'line': line
            })
        
        # ตรวจสอบการ snap ของจุดเริ่มต้นและจุดสิ้นสุด
        start_snapped = snap_map.get(coords[0], coords[0])
        end_snapped = snap_map.get(coords[-1], coords[-1])
        
        # ตรวจสอบว่าจุดเริ่มต้นและสิ้นสุดถูก snap ไปที่จุดเดียวกันหรือไม่
        if start_snapped == end_snapped and len(coords) > 2:
            problematic_connections.append({
                'index': idx,
                'reason': 'Start and end snap to same point',
                'original_start': coords[0],
                'original_end': coords[-1],
                'snapped_point': start_snapped,
                'line': line
            })
    
    # สร้างกราฟเพื่อตรวจสอบการเชื่อมต่อ
    G = nx.Graph()
    node_id_map = {}
    node_counter = 0
    
    for line in lines:
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        if len(coords) < 2:
            continue
            
        snapped_coords = [snap_map.get(coord, coord) for coord in coords]
        
        for i in range(len(snapped_coords)):
            if snapped_coords[i] not in node_id_map:
                node_id_map[snapped_coords[i]] = node_counter
                node_counter += 1
        
        for i in range(len(snapped_coords) - 1):
            if snapped_coords[i] != snapped_coords[i+1]:
                n1 = node_id_map[snapped_coords[i]]
                n2 = node_id_map[snapped_coords[i+1]]
                G.add_edge(n1, n2)
    
    # หาสายที่โดดเดี่ยว (isolated components)
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        if len(components) > 1:
            # หา component ที่ใหญ่ที่สุด
            main_component = max(components, key=len)
            
            # ระบุสายที่อยู่ใน component เล็ก
            for idx, line in enumerate(lines):
                coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                          if not (np.isnan(x) or np.isnan(y))]
                if len(coords) < 2:
                    continue
                    
                snapped_coords = [snap_map.get(coord, coord) for coord in coords]
                line_nodes = set()
                
                for coord in snapped_coords:
                    if coord in node_id_map:
                        line_nodes.add(node_id_map[coord])
                
                # ถ้าไม่มี node ใดอยู่ใน main component
                hits = [c for c in components if line_nodes.intersection(c)]
                comp_size = len(hits[0]) if hits else 0
                isolated_lines.append({
                    'index': idx,
                    'component_size': comp_size,
                    'line': line
                })
    
    result = {
        'total_lines': len(lines),
        'failed_lines': failed_lines,
        'short_lines': short_lines,
        'problematic_connections': problematic_connections,
        'isolated_lines': isolated_lines,
        'total_issues': len(failed_lines) + len(short_lines) + 
                       len(problematic_connections) + len(isolated_lines)
    }
    
    # Log summary
    logging.info(f"Line snap analysis complete:")
    logging.info(f"  Total lines: {result['total_lines']}")
    logging.info(f"  Invalid lines: {len(failed_lines)}")
    logging.info(f"  Short lines: {len(short_lines)}")
    logging.info(f"  Problematic connections: {len(problematic_connections)}")
    logging.info(f"  Isolated lines: {len(isolated_lines)}")
    
    return result


def fix_failed_snap_lines(failed_analysis, lines, snap_tolerance):
    """
    พยายามแก้ไขสายที่มีปัญหาในการ snap
    
    Args:
        failed_analysis: ผลลัพธ์จาก identify_failed_snap_lines()
        lines: list ของสายทั้งหมด
        snap_tolerance: ค่า tolerance
    
    Returns:
        list: สายที่แก้ไขแล้ว
    """
    fixed_lines = lines.copy()
    fix_log = []
    
    # แก้ไขสายสั้น - รวมกับสายใกล้เคียง
    for short_line_info in failed_analysis['short_lines']:
        idx = short_line_info['index']
        start = short_line_info['start']
        end = short_line_info['end']
        
        # หาสายที่มีจุดต่อใกล้เคียง
        merge_candidate = None
        min_dist = snap_tolerance * 3  # ขยายระยะการค้นหา
        
        for i, other_line in enumerate(lines):
            if i == idx:
                continue
                
            other_coords = [(x, y) for x, y in zip(other_line['X'], other_line['Y']) 
                           if not (np.isnan(x) or np.isnan(y))]
            if len(other_coords) < 2:
                continue
                
            # ตรวจสอบระยะทางระหว่างจุดต่อ
            for coord in [other_coords[0], other_coords[-1]]:
                dist_to_start = np.hypot(coord[0] - start[0], coord[1] - start[1])
                dist_to_end = np.hypot(coord[0] - end[0], coord[1] - end[1])
                
                if min(dist_to_start, dist_to_end) < min_dist:
                    min_dist = min(dist_to_start, dist_to_end)
                    merge_candidate = i
        
        if merge_candidate is not None:
            fix_log.append({
                'action': 'merge',
                'line_index': idx,
                'merged_with': merge_candidate,
                'reason': 'Short line merged with nearby line'
            })
            # ในการใช้งานจริง อาจต้องรวมสายจริงๆ
            # แต่ตอนนี้เพียงแค่บันทึกว่าควรรวม
    
    # แก้ไขสายที่มีจุดเริ่มต้นและสิ้นสุด snap ไปที่เดียวกัน
    for prob_conn in failed_analysis['problematic_connections']:
        idx = prob_conn['index']
        line = lines[idx]
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        
        if len(coords) > 2:
            # ใช้จุดกลางเป็นจุดแบ่ง
            mid_idx = len(coords) // 2
            
            # สร้างสายใหม่ 2 เส้น
            line1_x = [coord[0] for coord in coords[:mid_idx+1]]
            line1_y = [coord[1] for coord in coords[:mid_idx+1]]
            line2_x = [coord[0] for coord in coords[mid_idx:]]
            line2_y = [coord[1] for coord in coords[mid_idx:]]
            
            fix_log.append({
                'action': 'split',
                'line_index': idx,
                'split_point': coords[mid_idx],
                'reason': 'Loop detected - split into two lines'
            })
    
    return fixed_lines, fix_log


def export_failed_lines_shapefile(failed_analysis, lines, output_path):
    """
    Export สายที่มีปัญหาเป็น shapefile เพื่อตรวจสอบ
    
    Args:
        failed_analysis: ผลลัพธ์จาก identify_failed_snap_lines()
        lines: list ของสายทั้งหมด
        output_path: path สำหรับ output shapefile
    """
    import shapefile
    
    w = shapefile.Writer(output_path, shapeType=shapefile.POLYLINE)
    w.field('FID', 'N')
    w.field('Type', 'C', size=20)
    w.field('Reason', 'C', size=100)
    w.field('Length', 'F', decimal=6)
    w.field('Index', 'N')
    
    fid = 0
    
    # Export สายที่มีปัญหาแต่ละประเภท
    for failed in failed_analysis['failed_lines']:
        idx = failed['index']
        line = failed['line']
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        if len(coords) >= 2:
            w.line([coords])
            w.record(fid, 'Failed', failed['reason'], 0, idx)
            fid += 1
    
    for short in failed_analysis['short_lines']:
        idx = short['index']
        line = short['line']
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        if len(coords) >= 2:
            w.line([coords])
            w.record(fid, 'Short', f'Length < 2*tolerance', 
                    short['length'], idx)
            fid += 1
    
    for prob in failed_analysis['problematic_connections']:
        idx = prob['index']
        line = prob['line']
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        if len(coords) >= 2:
            w.line([coords])
            w.record(fid, 'Problematic', prob['reason'], 0, idx)
            fid += 1
    
    for isolated in failed_analysis['isolated_lines']:
        idx = isolated['index']
        line = isolated['line']
        coords = [(x, y) for x, y in zip(line['X'], line['Y']) 
                  if not (np.isnan(x) or np.isnan(y))]
        if len(coords) >= 2:
            w.line([coords])
            w.record(fid, 'Isolated', 
                    f'Component size: {isolated["component_size"]}', 0, idx)
            fid += 1
    
    w.close()
    logging.info(f"Failed lines exported to: {output_path}")

# ---------------------------------
# 8) network validation
def validate_network_after_snap(G, coord_mapping, meterNodes, transformerNode):
    if G.number_of_nodes() == 0:
        return {
            'is_connected': False,
            'num_components': 0,
            'components': [],
            'unreachable_meters': list(meterNodes),
            'meters_with_long_path': [],
            'duplicate_edges': [],
            'self_loops': [],
            'isolated_nodes': [],
            'summary': {
                'total_nodes': 0,
                'total_edges': 0,
                'total_meters': len(meterNodes),
                'reachable_meters': 0,
                'network_complete': False
            }
        }
    validation_result = {
        'is_connected': nx.is_connected(G),
        'num_components': nx.number_connected_components(G),
        'components': list(nx.connected_components(G)),
        'unreachable_meters': [],
        'meters_with_long_path': [],
        'duplicate_edges': [],
        'self_loops': list(nx.selfloop_edges(G)),
        'isolated_nodes': list(nx.isolates(G))
    }
    for meter in meterNodes:
        try:
            path_length = nx.shortest_path_length(G, transformerNode, meter, weight='weight')
            if path_length > 1000:
                validation_result['meters_with_long_path'].append({'meter': meter,'distance': path_length})
        except nx.NetworkXNoPath:
            validation_result['unreachable_meters'].append(meter)

    seen = set()
    for u, v in G.edges():
        e = tuple(sorted((u, v)))
        if e in seen:
            validation_result['duplicate_edges'].append(e)
        seen.add(e)

    validation_result['summary'] = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'total_meters': len(meterNodes),
        'reachable_meters': len(meterNodes) - len(validation_result['unreachable_meters']),
        'network_complete': validation_result['is_connected'] and len(validation_result['unreachable_meters']) == 0
    }
    logging.info("Network validation after snap complete.")
    return validation_result


# ---------------------------------
# 9) extract data from shapefile
def extractLineData(lineData, snap_tolerance=0.1):
    """แก้ไขฟังก์ชันเดิมให้รองรับ coordinate snapping"""
    logging.info(f"Extracting line data with coordinate snapping (tolerance={snap_tolerance}m)")
    shapes = lineData.shapes()
    linesX = []
    linesY = []
    lines = []
    all_coords = []
    
    # รวบรวมพิกัดทั้งหมด
    for shape in shapes:
        coords = [(point[0], point[1]) for point in shape.points if not (np.isnan(point[0]) or np.isnan(point[1]))]
        all_coords.extend(coords)
    
    # สร้าง coordinate mapping สำหรับ snapping
    coord_snap_map = snap_coordinates_to_tolerance(list(set(all_coords)), snap_tolerance)
    
    # ประมวลผลแต่ละเส้น
    for shape in shapes:
        original_coords = [(point[0], point[1]) for point in shape.points if not (np.isnan(point[0]) or np.isnan(point[1]))]
        
        if len(original_coords) < 2:
            continue
            
        # Snap coordinates
        snapped_coords = [coord_snap_map.get(coord, coord) for coord in original_coords]
        
        # แยก x, y
        x_coords = [coord[0] for coord in snapped_coords]
        y_coords = [coord[1] for coord in snapped_coords]
        
        linesX.append(x_coords)
        linesY.append(y_coords)
        lines.append({'X': x_coords, 'Y': y_coords})
    
    logging.info(f"Processed {len(lines)} lines with coordinate snapping")
    return linesX, linesY, lines

def extractLineDataWithAttributes(lineData, required_field, required_value, snap_tolerance=0.1):
    """แก้ไขฟังก์ชันเดิมให้รองรับ coordinate snapping"""
    logging.info(f"Extracting line data with attribute filter: {required_field} == {required_value} and snapping tolerance {snap_tolerance}m")
    
    shapes = lineData.shapes()
    records = lineData.records()
    fields = [field[0] for field in lineData.fields[1:]]
    
    # รวบรวมข้อมูลเส้นที่ตรงเงื่อนไข
    filtered_lines = []
    all_coords = []
    
    for shape, record in zip(shapes, records):
        attributes = dict(zip(fields, record))
        if attributes.get(required_field) == required_value:
            coords = [(p[0], p[1]) for p in shape.points if not (np.isnan(p[0]) or np.isnan(p[1]))]
            if len(coords) >= 2:
                filtered_lines.append((coords, attributes))
                all_coords.extend(coords)
    
    # Snap coordinates
    coord_snap_map = snap_coordinates_to_tolerance(list(set(all_coords)), snap_tolerance)
    
    # ประมวลผลเส้นที่กรองแล้ว
    linesX = []
    linesY = []
    lines = []
    
    for coords, attributes in filtered_lines:
        snapped_coords = [coord_snap_map.get(coord, coord) for coord in coords]
        
        x = [coord[0] for coord in snapped_coords]
        y = [coord[1] for coord in snapped_coords]
        
        linesX.append(x)
        linesY.append(y)
        lines.append({
            'X': x, 
            'Y': y, 
            'ATTRIBUTES': attributes
        })
    
    logging.info(f"Found {len(lines)} lines matching {required_field}={required_value} with coordinate snapping.")
    return linesX, linesY, lines

def extractMeterData(meterData):
    logging.info("Extracting meter data from shapefile records...")
    meterShapes = meterData.shapes()
    meterRecords = meterData.records()
    fields = [field[0] for field in meterData.fields[1:]]
    df = pd.DataFrame(meterRecords, columns=fields)
    numMeters = len(meterShapes)
    meterLocations = np.array([meterShapes[i].points[0] for i in range(numMeters)])
    required_fields = {'OPSA_MET_2', 'OPSA_MET_3', 'OPSA_MET_4', 'PEANO'}
    if not required_fields.issubset(df.columns):
        missing = required_fields - set(df.columns)
        logging.error(f"Missing required fields in Meter shapefile: {missing}")
        raise KeyError(f"Missing required fields in Meter shapefile: {missing}")
    initialVoltages = df['OPSA_MET_3'].values.astype(float)
    totalLoads = df['OPSA_MET_4'].values.astype(float)
    phases = df['OPSA_MET_2'].values
    peano = df['PEANO'].values
    
    
    # log เตือนค่าแรงดันต่ำ
    lowVoltageIndices = initialVoltages < 100
    if np.any(lowVoltageIndices):
        num_low_voltage = np.sum(lowVoltageIndices)
        logging.warning(f"Found {num_low_voltage} meters with voltage < 100V. These meters will be included in calculation.")
    
    phase_loads = {'A': np.zeros(len(totalLoads)),
                   'B': np.zeros(len(totalLoads)),
                   'C': np.zeros(len(totalLoads))}
    for i, phase_str in enumerate(phases):
        connected_phases = list(phase_str.upper())
        if not connected_phases:
            logging.debug(f"Skipping meter index={i} (empty phases).")
            continue
        load_per_phase = totalLoads[i] / len(connected_phases)
        for ph in connected_phases:
            if ph in phase_loads:
                phase_loads[ph][i] = load_per_phase
            else:
                logging.warning(f"Unknown phase '{ph}' for meter index {i}.")
    logging.info(f"Meter data extracted successfully. Total meters: {len(meterLocations)}")
    return meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases

# ---------------------------------
# 10) build & analyze LV network
def buildLVNetworkWithLoads(
    lvLines, mvLines, meterLocations, transformerLocation, phase_loads, conductorResistance,
    conductorReactance=None, *, svcLines=None,
    use_shape_length=False, lvData=None, length_field="SHAPE.LEN",
    snap_tolerance=0.1, line_length_map=None, coord_snap_map=None
):
    """
    Build LV graph with optional coordinate snapping + per-segment lengths from JSON.

    - ถ้ามี coord_snap_map ให้มา จะใช้ทันที (ไม่สร้างใหม่)
    - ถ้า use_shape_length=True:
        - ใช้ line_length_map ที่ส่งมา; ถ้าไม่มีและมี lvData → สร้างจาก JSON
        - ต้องมีอย่างใดอย่างหนึ่ง (line_length_map หรือ lvData) ไม่งั้น raise
    - น้ำหนัก edge และ R/X จะคำนวณจากความยาว (เมตร): ใช้ length_map ก่อน, ไม่มีก็ fallback ยูคลิด
    """
    if svcLines is None:
        svcLines = []

    if transformerLocation is None or len(transformerLocation) != 2:
        raise ValueError("transformerLocation ต้องเป็น (x, y)")

    logging.info(f"Building LV network with coordinate snapping (tolerance={snap_tolerance} m)…")

    G = nx.Graph()
    node_id = 0
    node_mapping = {}   # (x,y) -> node_id
    coord_mapping = {}  # node_id -> (x,y)
    lv_nodes = []

    # ---------- สร้าง/รับ coord_snap_map ----------
    if coord_snap_map is None:
        all_line_coords = []
        for line in lvLines:
            all_line_coords.extend([(x, y) for x, y in zip(line['X'], line['Y']) if not (np.isnan(x) or np.isnan(y))])
        for line in mvLines:
            all_line_coords.extend([(x, y) for x, y in zip(line['X'], line['Y']) if not (np.isnan(x) or np.isnan(y))])
        coord_snap_map = snap_coordinates_to_tolerance(list(set(all_line_coords)), snap_tolerance)

    # ---------- เตรียม line_length_map เมื่อขอใช้ความยาวจาก JSON ----------
    
    if use_shape_length:
        # ถ้ายังไม่มี map และไม่ได้ส่ง lvData เข้ามา ลองหยิบจากโกลบอล out_json
        src_json = lvData if lvData is not None else globals().get('out_json', None)

        if line_length_map is None and src_json is not None:
            try:
                line_length_map = build_line_length_map_from_json(
                    json_input=src_json,
                    coord_snap_map=coord_snap_map,
                    length_field=length_field if length_field else "SHAPE.LEN",
                    fallback_fields=("Shape_Leng", "SHAPE_Leng"),
                    unit_factor=1.0
                )
                logging.info(f"[LEN] built from JSON: {len(line_length_map)//2} segments")
            except Exception as e:
                logging.warning(f"[LEN] failed to build from JSON -> fallback to Euclidean: {e}")
                line_length_map = None

        # ถ้ายังไม่มีจริง ๆ ก็ถอยไปใช้ยูคลิด (ไม่ raise)
        if line_length_map is None:
            logging.warning("use_shape_length=True แต่ไม่มี line_length_map/lvData -> ใช้ Euclidean length แทน")
            use_shape_length = False
    # ---------- helper ----------
    def _build_meter_to_service_map(svcLines, snap_map):
        m2svc = {}
        for line in svcLines:
            pts = [(x, y) for x, y in zip(line['X'], line['Y']) if not np.isnan(x)]
            if len(pts) < 2: 
                continue
            p0 = snap_map.get(tuple(pts[0]), tuple(pts[0]))     # meter end
            p1 = snap_map.get(tuple(pts[-1]), tuple(pts[-1]))   # LV end
            m2svc[p0] = p1
            m2svc[p1] = p0
        return m2svc
    def add_line_to_network(line, is_lv=True, is_service=False):
        nonlocal node_id, lv_nodes
        coords = [(x, y) for x, y in zip(line['X'], line['Y'])
                if not (np.isnan(x) or np.isnan(y))]
        if len(coords) < 2:
            return

        # snap จุดทุกจุดตาม coord_snap_map
        snapped = [coord_snap_map.get(tuple(c), tuple(c)) for c in coords]
        prev_node = None
        for pt in snapped:
            if pt not in node_mapping:
                node_mapping[pt] = node_id
                coord_mapping[node_id] = pt
                node_id += 1
            cur_node = node_mapping[pt]

            if is_lv:
                lv_nodes.append(cur_node)

            if prev_node is not None and prev_node != cur_node:
                a = coord_mapping[prev_node]
                b = coord_mapping[cur_node]
                used_len = None
                if use_shape_length and line_length_map:
                    used_len = (line_length_map.get((a, b))
                                or line_length_map.get((b, a)))
                if used_len is None:
                    used_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))

                R = (used_len / 1000.0) * conductorResistance
                X = (used_len / 1000.0) * (
                    conductorReactance if conductorReactance is not None
                    else 0.1 * conductorResistance
                )

                if not G.has_edge(prev_node, cur_node):
                    G.add_edge(
                        prev_node, cur_node,
                        weight=used_len,
                        resistance=R,
                        reactance=X,
                        is_service=is_service   
                    )
            prev_node = cur_node


    # ---------- ใส่เส้น ----------
    for line in lvLines:
        add_line_to_network(line, is_lv=True)
    for line in mvLines:
        add_line_to_network(line, is_lv=False)
    # ---------- ใส่เส้น Eservice (จาก JSON) เป็นเส้นจริงในกราฟ ----------
    svcLines = svcLines or []  # กัน None
    for line in svcLines:
        add_line_to_network(line, is_lv=False, is_service=True)
    # ---------- เตรียม KDTree ของ LV nodes (order ให้ deterministic) ----------
    lv_nodes = sorted(set(lv_nodes))
    if lv_nodes:
        lv_pts = np.array([coord_mapping[n] for n in lv_nodes])
        kdt_lv = cKDTree(lv_pts)
    else:
        kdt_lv = None

    # ---------- เตรียม mapping "ปลายสายฝั่งมิเตอร์" ของ eservice ----------
    service_meter_points = []
    service_meter_nodes = []

    for line in svcLines:
        coords = [(x, y) for x, y in zip(line['X'], line['Y'])
                if not (np.isnan(x) or np.isnan(y))]
        if not coords:
            continue

        # จุดสุดท้ายของเส้นคือฝั่งมิเตอร์
        p_meter = coord_snap_map.get(tuple(coords[-1]), tuple(coords[-1]))
        node = node_mapping.get(p_meter)
        if node is not None:
            service_meter_points.append(p_meter)
            service_meter_nodes.append(node)

    kdt_svc_meter = None
    if service_meter_points:
        service_meter_points_arr = np.array(service_meter_points)
        kdt_svc_meter = cKDTree(service_meter_points_arr)
        
    # ---------- เชื่อมมิเตอร์ ----------
    meterNodes = []
    if 'tk_progress' in globals():
        tk_progress.start(len(meterLocations), stage="Attach meters")

    for idx, m_xy in enumerate(meterLocations):
        # สร้าง node ของมิเตอร์
        meterNode = node_id
        node_mapping[tuple(m_xy)] = meterNode
        coord_mapping[meterNode] = tuple(m_xy)
        node_id += 1
        G.add_node(meterNode)

        # ใส่โหลดต่อเฟสให้ node มิเตอร์
        for ph in 'ABC':
            G.nodes[meterNode][f'load_{ph}'] = float(phase_loads[ph][idx])

        snap_m = coord_snap_map.get(tuple(m_xy), tuple(m_xy))

        target_node = None
        dist = 0.0
        is_service_edge = False

        # 1) ถ้ามี eservice → ล็อกให้ต่อเข้าปลายสายฝั่งมิเตอร์ก่อน (ล็อกตาม JSON)
        if kdt_svc_meter is not None:
            d_svc, i_svc = kdt_svc_meter.query(snap_m)
            # ใช้ threshold ตาม snap_tolerance (ปรับได้)
            if d_svc <= snap_tolerance:
                target_node = service_meter_nodes[int(i_svc)]
                dist = float(d_svc)
                is_service_edge = True

        # 2) ถ้าไม่เจอ service ที่ match → fallback ไป snap เข้าสาย LV ใกล้สุด
        if target_node is None:
            if not lv_nodes:
                logging.error("No LV nodes to snap meter.")
                continue

            # KDTree สำหรับ LV nodes
            lv_coords = np.array([coord_mapping[n] for n in lv_nodes])
            kdt_lv = cKDTree(lv_coords)

            d_lv, i_lv = kdt_lv.query(snap_m)
            target_node = lv_nodes[int(i_lv)]
            dist = float(d_lv)
            is_service_edge = False

        # 3) สร้าง edge มิเตอร์ -> node เป้าหมาย
        Rm = (dist / 1000.0) * conductorResistance
        Xm = (dist / 1000.0) * (
            conductorReactance if conductorReactance is not None
            else 0.1 * conductorResistance
        )

        G.add_edge(
            meterNode, target_node,
            weight=dist,
            resistance=Rm,
            reactance=Xm,
            is_service=is_service_edge
        )

        meterNodes.append(meterNode)

        if 'tk_progress' in globals():
            tk_progress.step()

    if 'tk_progress' in globals():
        tk_progress.finish()



    # ---------- เพิ่ม Transformer node ----------
    tx_raw = tuple(transformerLocation)
    tx_snap = coord_snap_map.get(tx_raw, tx_raw)
    if tx_snap in node_mapping:
        transformerNode = node_mapping[tx_snap]
    else:
        transformerNode = node_id
        node_mapping[tx_snap] = transformerNode
        coord_mapping[transformerNode] = tx_snap
        node_id += 1
        G.add_node(transformerNode)
        for ph in 'ABC':
            G.nodes[transformerNode][f'load_{ph}'] = 0.0

        # snap เข้า LV node ใกล้สุด
        if lv_nodes:
            lv_pts = np.array([coord_mapping[n] for n in lv_nodes])  # สร้างใหม่ให้สอดคล้องกับ lv_nodes ที่ sort แล้ว
            tx_arr = np.array(tx_snap)
            dists = np.sqrt(np.sum((lv_pts - tx_arr) ** 2, axis=1))
            min_idx = int(np.argmin(dists))
            closest = lv_nodes[min_idx]
            min_dist = float(dists[min_idx])
            Rt = (min_dist / 1000.0) * conductorResistance
            Xt = (min_dist / 1000.0) * (conductorReactance if conductorReactance is not None else 0.1 * conductorResistance)
            G.add_edge(transformerNode, closest, weight=min_dist, resistance=Rt, reactance=Xt)
        else:
            logging.warning("No LV lines to connect the transformer.")

    # ---------- เติมโหลด 0 ให้ node ที่ยังไม่มี ----------
    for n in G.nodes:
        for ph in 'ABC':
            G.nodes[n].setdefault(f'load_{ph}', 0.0)

    # ---------- ตรวจสุขภาพ ----------
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        logging.warning(f"Network is not fully connected after snapping (components={len(comps)})")
    else:
        logging.info("Network is fully connected after snapping.")

    short_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d.get('weight', 0.0) < snap_tolerance]
    if short_edges:
        logging.warning(f"Found {len(short_edges)} edges shorter than snap tolerance")

    logging.info(f"LV network built: nodes={G.number_of_nodes()} edges={G.number_of_edges()} (snap_tol={snap_tolerance} m)")
    return (G, transformerNode, meterNodes, node_mapping, coord_mapping)

def buildLVNetworkUsingLineAttribute(lvData, length_field="Shape_Leng"):
    logging.info(f"Building LV network using line attribute '{length_field}' as distance...")
    G = nx.Graph()
    node_id = 0
    node_mapping = {}
    coord_mapping = {}
    shapes = lvData.shapes()
    records = lvData.records()
    fields = [f[0] for f in lvData.fields[1:]]
    for shape_idx, (shape, rec) in enumerate(zip(shapes, records)):
        attrs = dict(zip(fields, rec))
        if length_field not in attrs:
            logging.warning(f"Record {shape_idx} does not have field '{length_field}'. Skipping.")
            continue
        try:
            line_length = float(attrs[length_field])
        except Exception as e:
            logging.error(f"Error converting {length_field} in record {shape_idx}: {e}")
            continue
        if len(shape.points) < 2:
            logging.warning(f"Shape {shape_idx} has less than 2 points. Skipping.")
            continue
        start_coord = tuple(shape.points[0])
        end_coord   = tuple(shape.points[-1])
        if start_coord not in node_mapping:
            node_mapping[start_coord] = node_id
            coord_mapping[node_id] = start_coord
            node_id += 1
        if end_coord not in node_mapping:
            node_mapping[end_coord] = node_id
            coord_mapping[node_id] = end_coord
            node_id += 1
        start_node = node_mapping[start_coord]
        end_node = node_mapping[end_coord]
        G.add_edge(start_node, end_node, weight=line_length)
    logging.info(f"Created {G.number_of_edges()} edges in LV network using line attribute.")
    return G, node_mapping, coord_mapping

def snapPointToLVNetwork(G, node_mapping, coord_mapping, pointXY):
    import math
    min_dist = float('inf')
    nearest_node = None
    x_m, y_m = pointXY
    for node_id, (x_n, y_n) in coord_mapping.items():
        dist = math.hypot(x_m - x_n, y_m - y_n)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node_id
    new_point_id = max(coord_mapping.keys(), default=0) + 1
    G.add_node(new_point_id)
    coord_mapping[new_point_id] = pointXY
    G.add_edge(new_point_id, nearest_node, weight=min_dist)
    return new_point_id

# ---------------------------------
# 11) Network load center
def calculateNetworkLoadCenter(meterLocations, phase_loads, lvLines, mvLines, conductorResistance,
                              conductorReactance=None, lvData=None, svcLines=None, snap_tolerance=SNAP_TOLERANCE):
    """
    แก้ไขฟังก์ชันเดิมให้ใช้ coordinate snapping และเพิ่มประสิทธิภาพสำหรับมิเตอร์จำนวนมาก
    """
    logging.info("Calculating network load center with coordinate snapping and optimizations...")
    if len(meterLocations) == 0:
        logging.warning("No meter locations; returning [0, 0].")
        return np.array([0, 0], dtype=float)
    
    # สร้างกราฟเครือข่าย พร้อม coordinate snapping
    G, tNode, mNodes, node_mapping, coord_mapping = buildLVNetworkWithLoads(
        lvLines, mvLines, meterLocations, meterLocations[0], phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=lvData is not None,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=snap_tolerance
    )
    
    # คำนวณโหลดรวมของแต่ละมิเตอร์
    total_loads = phase_loads['A'] + phase_loads['B'] + phase_loads['C']
    
    # สร้าง array สำหรับเก็บผลรวมระยะทาง
    node_list = list(G.nodes())
    node_id_to_index = {nid: idx for idx, nid in enumerate(node_list)}
    sum_distances = np.zeros(len(node_list))
    
    # กรองมิเตอร์ที่ไม่มีโหลด
    valid_meters = [(i, meter_node) for i, meter_node in enumerate(mNodes) if total_loads[i] > 0]
    
    if len(valid_meters) == 0:
        logging.warning("No meters with non-zero loads")
        return np.array([0, 0], dtype=float)
    
    logging.info(f"Processing {len(valid_meters)} meters with non-zero loads out of {len(mNodes)} total meters")
    
    # ตรวจสอบขนาดของปัญหาและเลือกวิธีการคำนวณ
    num_valid_meters = len(valid_meters)
    
    if num_valid_meters <= 100:
        # วิธีเดิมสำหรับมิเตอร์น้อย
        if 'tk_progress' in globals():
            tk_progress.start(num_valid_meters, stage="Distance calc")
        
        for i, (meter_idx, meter_node) in enumerate(valid_meters):
            load = total_loads[meter_idx]
            distances = nx.single_source_dijkstra_path_length(G, meter_node, weight='weight')
            
            for node, dist in distances.items():
                if node in node_id_to_index:
                    sum_distances[node_id_to_index[node]] += load * dist
            
            if 'tk_progress' in globals():
                tk_progress.step()
        
        if 'tk_progress' in globals():
            tk_progress.finish()
    
    else:
        # วิธีที่เร็วขึ้นสำหรับมิเตอร์จำนวนมาก
        logging.info(f"Using optimized calculation for {num_valid_meters} meters")
        
        # 1. จัดกลุ่มมิเตอร์ที่อยู่ใกล้กัน
        meter_coords = np.array([coord_mapping[node] for _, node in valid_meters])
        
        # ใช้ k-means clustering แบบง่าย
        num_clusters = min(50, num_valid_meters // 10)  # ประมาณ 10 มิเตอร์ต่อ cluster
        
        if num_clusters > 1:
            # Simple k-means clustering
            
            try:
                kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=100, random_state=42)
                cluster_labels = kmeans.fit_predict(meter_coords)
                cluster_centers = kmeans.cluster_centers_
            except:
                # Fallback ถ้าไม่มี sklearn
                logging.info("sklearn not available, using grid-based clustering")
                cluster_labels, cluster_centers = _simple_grid_clustering(meter_coords, num_clusters)
        else:
            cluster_labels = np.zeros(len(valid_meters), dtype=int)
            cluster_centers = [np.mean(meter_coords, axis=0)]
        
        # 2. คำนวณ representative meters สำหรับแต่ละ cluster
        if 'tk_progress' in globals():
            tk_progress.start(num_clusters, stage="Optimized distance calc")
        
        for cluster_id in range(num_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_meters = [vm for i, vm in enumerate(valid_meters) if cluster_mask[i]]
            
            if not cluster_meters:
                continue
            
            # หา representative meter (มิเตอร์ที่มีโหลดมากที่สุดใน cluster)
            max_load_idx = 0
            max_load = 0
            for i, (meter_idx, _) in enumerate(cluster_meters):
                if total_loads[meter_idx] > max_load:
                    max_load = total_loads[meter_idx]
                    max_load_idx = i
            
            rep_meter_idx, rep_meter_node = cluster_meters[max_load_idx]
            
            # คำนวณ total load ของ cluster
            cluster_total_load = sum(total_loads[meter_idx] for meter_idx, _ in cluster_meters)
            
            # คำนวณระยะทางจาก representative meter
            try:
                distances = nx.single_source_dijkstra_path_length(G, rep_meter_node, weight='weight')
                
                # Apply ด้วย cluster load
                for node, dist in distances.items():
                    if node in node_id_to_index:
                        sum_distances[node_id_to_index[node]] += cluster_total_load * dist
            except:
                logging.warning(f"Failed to calculate distances for cluster {cluster_id}")
            
            if 'tk_progress' in globals():
                tk_progress.step()
        
        if 'tk_progress' in globals():
            tk_progress.finish()
        
        # 3. Fine-tune ด้วยมิเตอร์ที่มีโหลดสูงมาก (top 10%)
        high_load_threshold = np.percentile(total_loads[total_loads > 0], 90)
        high_load_meters = [(i, node) for i, node in valid_meters 
                           if total_loads[i] > high_load_threshold]
        
        if len(high_load_meters) < 50:  # ถ้ามีไม่มากเกินไป
            logging.info(f"Fine-tuning with {len(high_load_meters)} high-load meters")
            for meter_idx, meter_node in high_load_meters:
                load = total_loads[meter_idx]
                try:
                    distances = nx.single_source_dijkstra_path_length(G, meter_node, weight='weight')
                    for node, dist in distances.items():
                        if node in node_id_to_index:
                            # ลบค่าเดิมที่คำนวณจาก cluster และใส่ค่าที่แม่นยำ
                            sum_distances[node_id_to_index[node]] += load * dist * 0.1  # weight adjustment
                except:
                    pass
    
    # หาโหนดที่มีผลรวมระยะทางต่ำสุด
    best_node_idx = np.argmin(sum_distances)
    best_node_id = node_list[best_node_idx]
    bestCoord = np.array(coord_mapping[best_node_id])
    
    logging.info(f"Best network load center: {bestCoord}")
    return bestCoord


def _simple_grid_clustering(coords, n_clusters):
    """
    Simple grid-based clustering เมื่อไม่มี sklearn
    """
    # หาขอบเขตของพิกัด
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    # คำนวณขนาด grid
    grid_size = int(np.sqrt(n_clusters))
    x_step = (max_x - min_x) / grid_size
    y_step = (max_y - min_y) / grid_size
    
    # กำหนด cluster labels
    labels = np.zeros(len(coords), dtype=int)
    centers = []
    
    for i in range(len(coords)):
        x, y = coords[i]
        grid_x = int((x - min_x) / x_step) if x_step > 0 else 0
        grid_y = int((y - min_y) / y_step) if y_step > 0 else 0
        
        # จำกัดไม่ให้เกิน grid
        grid_x = min(grid_x, grid_size - 1)
        grid_y = min(grid_y, grid_size - 1)
        
        labels[i] = grid_y * grid_size + grid_x
    
    # คำนวณ centers
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.any():
            centers.append(coords[mask].mean(axis=0))
        else:
            # ถ้าไม่มีจุดใน cluster นี้ ใช้จุดกลางของ grid
            grid_x = cluster_id % grid_size
            grid_y = cluster_id // grid_size
            center_x = min_x + (grid_x + 0.5) * x_step
            center_y = min_y + (grid_y + 0.5) * y_step
            centers.append([center_x, center_y])
    
    return labels, np.array(centers)

# ---------------------------------
# 12) Power-flow & loss
def calculatePowerLoss(G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage):
    logging.info("Calculating power flow and losses using DFS tree method...")
    T = nx.dfs_tree(G, source=transformerNode)
    voltages = {node: {'A': initialVoltage, 'B': initialVoltage, 'C': initialVoltage} for node in G.nodes()}
    branch_currents = {}
    node_currents = {}
    for node in G.nodes():
        loadA = G.nodes[node]['load_A']
        loadB = G.nodes[node]['load_B']
        loadC = G.nodes[node]['load_C']
        node_currents[node] = {
            'A': (loadA*1000)/(initialVoltage*powerFactor) if loadA>0 else 0.0,
            'B': (loadB*1000)/(initialVoltage*powerFactor) if loadB>0 else 0.0,
            'C': (loadC*1000)/(initialVoltage*powerFactor) if loadC>0 else 0.0
        }
    reverse_order = list(nx.dfs_postorder_nodes(T, source=transformerNode))
    accumulated_currents = {n: {'A':0.0, 'B':0.0, 'C':0.0} for n in G.nodes()}
    for node in reverse_order:
        currA = node_currents[node]['A']
        currB = node_currents[node]['B']
        currC = node_currents[node]['C']
        for child in T.successors(node):
            currA += accumulated_currents[child]['A']
            currB += accumulated_currents[child]['B']
            currC += accumulated_currents[child]['C']
        accumulated_currents[node]['A'] = currA
        accumulated_currents[node]['B'] = currB
        accumulated_currents[node]['C'] = currC
        if node != transformerNode:
            parent = list(T.predecessors(node))[0]
            branch_currents[(parent, node)] = {'A': currA, 'B': currB, 'C': currC}
    order = list(nx.dfs_preorder_nodes(T, source=transformerNode))
    for node in order:
        if node == transformerNode:
            continue
        parent = list(T.predecessors(node))[0]
        r = G[parent][node]['resistance']
        for ph in ['A','B','C']:
            voltages[node][ph] = voltages[parent][ph] - (branch_currents[(parent, node)][ph] * r)
    power_losses = {}
    for edge, currents in branch_currents.items():
        r = G[edge[0]][edge[1]]['resistance']
        lossA = (currents['A']**2)*r
        lossB = (currents['B']**2)*r
        lossC = (currents['C']**2)*r
        power_losses[edge] = lossA + lossB + lossC
    totalPowerLoss = sum(power_losses.values())
    finalVoltages = {node: voltages[node] for node in meterNodes}
    logging.info(f"Total power loss = {totalPowerLoss:.2f} W.")
    return finalVoltages, power_losses, totalPowerLoss
########################################
# 4. NEW: ITERATIVE UNBALANCED POWER FLOW CALCULATION
########################################

def calculateUnbalancedPowerFlow(G, transformerNode, meterNodes, powerFactor, base_voltage,
                                 max_iter=10, tol=1e-3):
    """
    Perform an iterative unbalanced load-flow calculation using a backward-forward sweep.
    
    Assumptions:
      - Each node's load (stored in G.nodes[node]['load_A'], etc.) is in kW.
      - Loads are modeled as constant power loads.
      - Reactive power is inferred from powerFactor:
          S = P*1000*(powerFactor + j*tan(phi)) with phi = arccos(powerFactor).
      - Each edge has 'resistance' and 'reactance' attributes (ohms).
    
    Returns:
      - node_voltages: dict[node] = {'A': complex voltage, 'B': ..., 'C': ...}
      - branch_currents: dict[(u,v)] = {'A': complex current, ...}
      - total_power_loss: Total real power loss in watts (sum over all edges).
    """
    # Calculate load angle from power factor
    phi = np.arccos(powerFactor)
    tan_phi = np.tan(phi)
    
    # Build a radial DFS tree from transformerNode
    T = nx.dfs_tree(G, source=transformerNode)
    
    # Initialize node voltages (complex) for each phase; you may initialize phase angles if desired.
    node_voltages = {}
    for n in G.nodes():
        node_voltages[n] = {
            'A': complex(base_voltage, 0.0),
            'B': complex(base_voltage, 0.0),
            'C': complex(base_voltage, 0.0)
        }
    # Fix the transformer node voltage
    node_voltages[transformerNode] = {
        'A': complex(base_voltage, 0.0),
        'B': complex(base_voltage, 0.0),
        'C': complex(base_voltage, 0.0)
    }
    
    for iteration in range(max_iter):
        old_voltages = {n: {ph: node_voltages[n][ph] for ph in node_voltages[n]} for n in G.nodes()}
        
        # BACKWARD SWEEP: Compute branch currents by accumulating load currents
        branch_currents = {}
        # For each node, compute its load current (constant power model: I = conj(S)/conj(V))
        # First, compute the load current at each node (for each phase)
        load_currents = {}
        for n in G.nodes():
            # Convert the kW load to complex power S (in VA)
            # S = P + jQ, with Q = P * tan(phi)
            P_A = G.nodes[n].get('load_A', 0.0) * 1000.0
            P_B = G.nodes[n].get('load_B', 0.0) * 1000.0
            P_C = G.nodes[n].get('load_C', 0.0) * 1000.0
            S_A = P_A + 1j * (P_A * tan_phi)
            S_B = P_B + 1j * (P_B * tan_phi)
            S_C = P_C + 1j * (P_C * tan_phi)
            # Compute load current: I = S* / V*  (avoid division by zero)
            V_A = node_voltages[n]['A'] if abs(node_voltages[n]['A'])>1e-6 else 1.0
            V_B = node_voltages[n]['B'] if abs(node_voltages[n]['B'])>1e-6 else 1.0
            V_C = node_voltages[n]['C'] if abs(node_voltages[n]['C'])>1e-6 else 1.0
            I_A = np.conjugate(S_A) / np.conjugate(V_A)
            I_B = np.conjugate(S_B) / np.conjugate(V_B)
            I_C = np.conjugate(S_C) / np.conjugate(V_C)
            load_currents[n] = {'A': I_A, 'B': I_B, 'C': I_C}
        
        # Initialize accumulated currents at each node to zero
        accumulated_current = {n: {'A': 0+0j, 'B': 0+0j, 'C': 0+0j} for n in G.nodes()}
        # Process nodes in reverse (post-order)
        for n in list(nx.dfs_postorder_nodes(T, source=transformerNode)):
            # Net current at node = its load current + sum(child branch currents)
            I_net_A = load_currents[n]['A'] + accumulated_current[n]['A']
            I_net_B = load_currents[n]['B'] + accumulated_current[n]['B']
            I_net_C = load_currents[n]['C'] + accumulated_current[n]['C']
            # If not the source, add this current to the parent's accumulated current
            if n != transformerNode:
                parent = list(T.predecessors(n))[0]
                branch_currents[(parent, n)] = {'A': I_net_A, 'B': I_net_B, 'C': I_net_C}
                accumulated_current[parent]['A'] += I_net_A
                accumulated_current[parent]['B'] += I_net_B
                accumulated_current[parent]['C'] += I_net_C
        
        # FORWARD SWEEP: Update node voltages from the transformer downward.
        for n in nx.dfs_preorder_nodes(T, source=transformerNode):
            if n == transformerNode:
                continue
            parent = list(T.predecessors(n))[0]
            # Get edge impedance (Z = R + jX) for each phase. Use default ratio if 'reactance' is missing.
            edge_data = G[parent][n]
            R = edge_data.get('resistance', 0.0)
            X = edge_data.get('reactance', 0.1 * R)
            Z = complex(R, X)
            # For each phase, voltage drop = I_edge * Z.
            I_A = branch_currents.get((parent, n), {}).get('A', 0+0j)
            I_B = branch_currents.get((parent, n), {}).get('B', 0+0j)
            I_C = branch_currents.get((parent, n), {}).get('C', 0+0j)
            node_voltages[n]['A'] = node_voltages[parent]['A'] - I_A * Z
            node_voltages[n]['B'] = node_voltages[parent]['B'] - I_B * Z
            node_voltages[n]['C'] = node_voltages[parent]['C'] - I_C * Z
        
        # Check for convergence: max change in voltage magnitude across all nodes/phases.
        max_diff = 0.0
        for n in G.nodes():
            for ph in ['A','B','C']:
                diff = abs(abs(node_voltages[n][ph]) - abs(old_voltages[n][ph]))
                if diff > max_diff:
                    max_diff = diff
        if max_diff < tol:
            logging.info(f"Unbalanced loadflow converged in {iteration+1} iterations (max voltage change {max_diff:.4f} V).")
            break
    else:
        logging.warning("Unbalanced loadflow did not converge within the maximum iterations.")
    
    # Calculate total real power loss over all edges: sum(|I|^2 * R) over phases.
    total_power_loss = 0.0
    for (u, v), currents in branch_currents.items():
        R = G[u][v].get('resistance', 0.0)
        loss = (abs(currents['A'])**2 + abs(currents['B'])**2 + abs(currents['C'])**2) * R
        total_power_loss += loss
    
    return node_voltages, branch_currents, total_power_loss

# ---------------------------------
# 13) Objectives & Constraints

def transformerOnLVWithCond3Constraint(transformerLocation, lvLines):
    epsilon = 1e-3
    closestDistance = float('inf')
    P = np.array(transformerLocation)
    for line in lvLines:
        xcoords = line['X']
        ycoords = line['Y']
        for i in range(len(xcoords) - 1):
            A = np.array([xcoords[i], ycoords[i]])
            B = np.array([xcoords[i+1], ycoords[i+1]])
            if np.all(A == B):
                dist = np.linalg.norm(P - A)
            else:
                t = np.dot(P - A, B - A)/np.dot(B - A, B - A)
                t = np.clip(t, 0, 1)
                proj = A + t*(B-A)
                dist = np.linalg.norm(P - proj)
            if dist < closestDistance:
                closestDistance = dist
    return epsilon - closestDistance

# ---------------------------------
# 14) Additional utility
def snap_to_conductor(load_center, conductor_lines):
    logging.info("Snapping point to the nearest conductor...")
    min_distance = np.inf
    snapped_point = load_center.copy()
    P = np.array(load_center)
    for line in conductor_lines:
        coords = list(zip(line['X'], line['Y']))
        coords = [c for c in coords if not np.isnan(c[0])]
        for i in range(len(coords) - 1):
            A = np.array(coords[i])
            B = np.array(coords[i+1])
            if np.all(A == B):
                distance = np.linalg.norm(P - A)
                projection = A
            else:
                t = np.dot(P - A, B - A)/np.dot(B - A, B - A)
                t = np.clip(t,0,1)
                projection = A + t*(B-A)
                distance = np.linalg.norm(P - projection)
            if distance < min_distance:
                min_distance = distance
                snapped_point = projection
    logging.debug(f"Snapped point: {snapped_point} (distance={min_distance:.2f})")
    return snapped_point

def get_junction_node_coords(G, coord_mapping, min_degree=3):
    logging.info(f"Collecting junction node coords (degree >= {min_degree})...")
    junction_coords = []
    for node in G.nodes():
        if G.degree(node) >= min_degree:
            if node in coord_mapping:
                junction_coords.append(coord_mapping[node])
    logging.info(f"Found {len(junction_coords)} junction nodes.")
    return junction_coords

def notOnJunctionConstraint(x, junction_coords, epsilon=1.0):
    x = np.array(x)
    min_dist = float('inf')
    for jc in junction_coords:
        dist = np.hypot(x[0] - jc[0], x[1] - jc[1])
        if dist < min_dist:
            min_dist = dist
    return min_dist - epsilon

# ---------------------------------
# 15) Splitting & partitioning
def findSplittingPoint(G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, project_id, candidate_index=0):
    global _EDGE_DF_CACHE
    logging.info("Finding splitting point by load balance difference on edges...")

    T = nx.dfs_tree(G, source=transformerNode)

    node_loads = {n: (G.nodes[n].get('load_A', 0) +
                      G.nodes[n].get('load_B', 0) +
                      G.nodes[n].get('load_C', 0))
                  for n in G.nodes()}

    cumulative_loads = {}
    def computeCumulativeLoad(node):
        if node in cumulative_loads:
            return cumulative_loads[node]
        load = node_loads[node]
        for child in T.successors(node):
            load += computeCumulativeLoad(child)
        cumulative_loads[node] = load
        return load

    total_load = computeCumulativeLoad(transformerNode)

    meter_set = set(meterNodes)
    meter_counts = {}
    def cum_meter(n):
        if n in meter_counts:
            return meter_counts[n]
        cnt = 1 if n in meter_set else 0
        for c in T.successors(n):
            cnt += cum_meter(c)
        meter_counts[n] = cnt
        return cnt

    total_meters = cum_meter(transformerNode)

    edge_diffs = []
    for (n1, n2) in T.edges():
        if (n1 in meter_set) or (n2 in meter_set):
            continue
        child = cumulative_loads[n2]
        parent = total_load - child
        diff = abs(child - parent)
        edge_diffs.append(((n1, n2), diff))

    edge_diffs.sort(key=lambda x: x[1])

    min_meters = max(1, int(0.20 * total_meters))
    chosen_idx = None
    for idx, (edge, _) in enumerate(edge_diffs):
        meters_child = meter_counts[edge[1]]
        meters_parent = total_meters - meters_child
        if meters_child >= min_meters and meters_parent >= min_meters:
            chosen_idx = idx
            break
    if chosen_idx is None:
        chosen_idx = 0

    candidate_index = chosen_idx if candidate_index == 0 else candidate_index

    # สร้าง DataFrame (ครบทุกคอลัมน์)
    rows = []
    for (edge, diff) in edge_diffs:
        n1, n2 = edge
        x1, y1 = coord_mapping.get(n1, (np.nan, np.nan))
        x2, y2 = coord_mapping.get(n2, (np.nan, np.nan))
        child_load  = cumulative_loads[n2]
        parent_load = total_load - child_load
        rows.append({
            'Edge': edge,
            'Edge_Diff': diff,
            'N1_X': x1, 'N1_Y': y1,
            'N2_X': x2, 'N2_Y': y2,
            'Load_G1': parent_load,
            'Load_G2': child_load
        })
    edge_diffs_df = pd.DataFrame(rows)
    edge_diffs_df.reset_index(inplace=True)
    edge_diffs_df.rename(columns={'index': 'splitting_index'}, inplace=True)
    _EDGE_DF_CACHE = edge_diffs_df

    # --- SAVE CSV (ไม่เขียน index ทับคอลัมน์) ---
    base_dir = os.path.join("pea_no_projects", "output", str(project_id))
    folder_path = base_dir
    ensure_folder_exists(folder_path)
    csv_path = f"{folder_path}/edgediff.csv"
    edge_diffs_df.to_csv(csv_path, index=False)
    logging.info(f"Splitting edges info saved to CSV: {csv_path}. Found {len(edge_diffs)} edges.")

    proj_tm3 = pyproj.CRS.from_proj4("+proj=utm +zone=47 +datum=WGS84 +units=m +no_defs")
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(proj_tm3, proj_wgs84, always_xy=True)
    
    # 6) แปลงพิกัด TM3 เป็น lat/lon สำหรับ JSON เท่านั้น
    n1_lons, n1_lats = transformer.transform(edge_diffs_df["N1_X"].values, edge_diffs_df["N1_Y"].values)
    n2_lons, n2_lats = transformer.transform(edge_diffs_df["N2_X"].values, edge_diffs_df["N2_Y"].values)

    # สร้างสำเนา DataFrame สำหรับ JSON แล้วเพิ่ม lat/lon
    edge_diffs_json_df = edge_diffs_df.copy()
    edge_diffs_json_df["N1_Lon"] = n1_lons
    edge_diffs_json_df["N1_Lat"] = n1_lats
    edge_diffs_json_df["N2_Lon"] = n2_lons
    edge_diffs_json_df["N2_Lat"] = n2_lats

    # ลบคอลัมน์ X/Y ถ้าไม่ต้องการใน JSON
    edge_diffs_json_df = edge_diffs_json_df.drop(columns=["N1_X", "N1_Y", "N2_X", "N2_Y"])

     # บันทึก JSON 
    output_folder = base_dir
    edge_diffs_json_path = os.path.join(output_folder, "edge_diffs.json")
    with open(edge_diffs_json_path, "w", encoding="utf-8") as f:
        json.dump(edge_diffs_json_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    logging.info(f"GeoJSON saved: {edge_diffs_json_path}")

    if edge_diffs_df.empty:
        _EDGE_DF_CACHE = None
        logging.warning("No valid edges for splitting.")
        return None, None, None, []

    if candidate_index < 0 or candidate_index >= len(edge_diffs):
        logging.error("Candidate index out of range, using index 0 instead.")
        candidate_index = 0

    best_edge = edge_diffs[candidate_index][0]
    best_edge_diff = edge_diffs[candidate_index][1]
    u, v = best_edge
    u_coord = np.array(coord_mapping[u])
    v_coord = np.array(coord_mapping[v])
    splitting_point_coord = (u_coord + v_coord) / 2

    logging.info(f"Splitting edge chosen (candidate={candidate_index}): {best_edge}, diff={best_edge_diff:.2f}")

    # --- EXPORT SHAPEFILE: fields และ record ให้ตรงกัน ---
    shp_path = f"{folder_path}/edgediff.shp"
    w = shapefile.Writer(shp_path, shapeType=shapefile.POLYLINE)
    w.field("FID","N")
    w.field("Index", "N")
    w.field("Edge_Diff", "F", decimal=2)

    for i, row in edge_diffs_df.iterrows():
        w.line([[
            [row['N1_X'], row['N1_Y']],
            [row['N2_X'], row['N2_Y']]
        ]])
        # FID, Index, Edge_Diff (ครบ 3 fields)
        w.record(int(i), int(row['splitting_index']), float(row['Edge_Diff']))  # ✅

    w.close()
    logging.info(f"Shapefile of edges exported: {shp_path}")

    return best_edge, splitting_point_coord, best_edge_diff, edge_diffs


def partitionNetworkAtPoint(G, transformerNode, meterNodes, splitting_edge=None):
    logging.info("Partitioning network by removing splitting edge...")
    G_copy = G.copy()
    if splitting_edge is not None:
        if G_copy.has_edge(*splitting_edge):
            G_copy.remove_edge(*splitting_edge)
            logging.info(f"Removed splitting edge: {splitting_edge}")
        else:
            logging.warning("Splitting edge not found in G_copy.")
    
    comps = list(nx.connected_components(G_copy))
    transformerComp = None
    for comp in comps:
        if transformerNode in comp:
            transformerComp = comp
            break
    group1_meters = []
    group2_meters = []
    for m in meterNodes:
        if m in transformerComp:
            group1_meters.append(m)
        else:
            group2_meters.append(m)
    logging.info(f"Partition done. Group1 has {len(group1_meters)} meters, Group2 has {len(group2_meters)} meters.")
    return group1_meters, group2_meters

def performForwardBackwardSweepAndDivideLoads(G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, splitting_edge):
    """
    แบ่งเครือข่ายโดยใช้ splitting_edge และคำนวณแรงดันและกระแสไฟฟ้าด้วย unbalanced power flow
    """
    logging.info("Performing unbalanced power flow and dividing loads by splitting edge...")
    
    # 1. แบ่งเครือข่ายที่จุดตัด
    group1_meters, group2_meters = partitionNetworkAtPoint(G, transformerNode, meterNodes, splitting_edge)
    
    # 2. ทำการคำนวณแรงดันและกระแสไฟฟ้าด้วย unbalanced power flow
    node_voltages, branch_currents, total_power_loss = calculateUnbalancedPowerFlow(
        G, transformerNode, meterNodes, powerFactor, initialVoltage, max_iter=10, tol=1e-3
    )
    
    # 3. แปลงค่าเชิงซ้อนเป็นค่าขนาด (magnitude) สำหรับการส่งคืนและการใช้กับ shapefile
    real_node_voltages = {}
    for node, phases in node_voltages.items():
        real_node_voltages[node] = {
            'A': abs(phases['A']),
            'B': abs(phases['B']),
            'C': abs(phases['C'])
        }
    
    real_branch_currents = {}
    for edge, phases in branch_currents.items():
        real_branch_currents[edge] = {
            'A': abs(phases['A']),
            'B': abs(phases['B']),
            'C': abs(phases['C'])
        }
    
    # รายงานการแบ่งกลุ่ม
    g1_load = sum(G.nodes[n].get('load_A', 0) + G.nodes[n].get('load_B', 0) + G.nodes[n].get('load_C', 0) for n in group1_meters)
    g2_load = sum(G.nodes[n].get('load_A', 0) + G.nodes[n].get('load_B', 0) + G.nodes[n].get('load_C', 0) for n in group2_meters)
    
    logging.info(f"Network partitioned at edge {splitting_edge}")
    logging.info(f"Group 1: {len(group1_meters)} meters, {g1_load:.2f} kW total load")
    logging.info(f"Group 2: {len(group2_meters)} meters, {g2_load:.2f} kW total load")
    
    return real_node_voltages, real_branch_currents, group1_meters, group2_meters

def get_bounding_box(points):
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
    return [(min_x, max_x), (min_y, max_y)]

# ---------------------------------
# 16) Optimization functions
def optimizeTransformerLocationOnLVCond3(meterLocations, phase_loads, initialTransformerLocation,
                                         lvLines, initialVoltage, conductorResistance, powerFactor,
                                         conductorReactance=None, lvData=None, svcLines=None,
                                         junction_coords=None, epsilon_junction=1.0, custom_bounds=None):
    logging.info("Optimizing Transformer location on LV cond=3 lines...")
    
    def obj_func(x):
        return objectiveFunction(x, meterLocations, phase_loads, initialVoltage,
                                 conductorResistance, lvLines, powerFactor, load_center_only=True,
                                 conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines)
    
    def lv_line_constraint(x):
        return transformerOnLVWithCond3Constraint(x, lvLines)
    
    def voltage_constraint(x):
        return voltageConstraint(x, meterLocations, phase_loads, initialVoltage,
                                 conductorResistance, lvLines, powerFactor,
                                 conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines)
    
    if len(lvLines) == 0:
        logging.warning("No LV cond=3 lines found. Skipping LV cond=3 optimization.")
        return initialTransformerLocation
    
    if custom_bounds is None:
        allLineX = np.concatenate([l['X'] for l in lvLines])
        allLineY = np.concatenate([l['Y'] for l in lvLines])
        bounds = [(min(allLineX), max(allLineX)), (min(allLineY), max(allLineY))]
    else:
        bounds = custom_bounds
    
    loadCenter = calculateNetworkLoadCenter(meterLocations, phase_loads, lvLines, [], conductorResistance,
                                            conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines)
    
    constraints_list = [
        {'type': 'ineq', 'fun': lv_line_constraint},
        {'type': 'ineq', 'fun': voltage_constraint}
    ]
    
    if junction_coords is not None and len(junction_coords) > 0:
        def not_on_junction(x):
            return notOnJunctionConstraint(x, junction_coords, epsilon_junction)
        constraints_list.append({'type': 'ineq', 'fun': not_on_junction})
    
    result = minimize(
        fun=obj_func,
        x0=loadCenter,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list,
        options={'maxiter': 50, 'disp': False}
    )
    
    if not result.success:
        logging.error(f"Optimization on LV lines failed: {result.message}")
        return loadCenter
    logging.info(f"LV cond=3 optimization successful => {result.x}, fun={result.fun:.4f}")
    return result.x

# ส่วนที่ 10: แก้ไขฟังก์ชันอื่นๆ ที่เกี่ยวข้อง
# แทนที่ฟังก์ชันเหล่านี้ในโค้ดเดิม:

def objectiveFunction(transformerLocation, meterLocations, phase_loads, initialVoltage,
                     conductorResistance, lvLines, powerFactor, load_center_only=False, 
                     conductorReactance=None, lvData=None, svcLines=None):
    """Objective function แบบใช้ powerflow + network load center + local load รอบตำแหน่ง TR"""
    logging.debug(f"Evaluating objective function at location {transformerLocation}...")

    # --- เตรียมข้อมูลโหลดของกลุ่ม ---
    total_loads_arr = phase_loads['A'] + phase_loads['B'] + phase_loads['C']
    total_group_load = float(total_loads_arr.sum())
    if total_group_load <= 0:
        logging.warning("Group has zero total load in objectiveFunction; returning large score.")
        return 1e12

    # --- สร้างกราฟสำหรับตำแหน่ง TR candidate นี้ ---
    G, tNode, mNodes, nm, cm = buildLVNetworkWithLoads(
        lvLines, [], meterLocations, transformerLocation, phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=lvData is not None,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )

    # mapping มิเตอร์ node -> load
    meter_node_to_load = {}
    for i, m_node in enumerate(mNodes):
        meter_node_to_load[m_node] = float(total_loads_arr[i])

    # --- คำนวณ powerflow ---
    node_voltages, power_flow, totalPowerLoss = calculateUnbalancedPowerFlow(
        G, tNode, mNodes, powerFactor, initialVoltage, max_iter=10, tol=1e-3
    )

    # รวม voltage drop ของมิเตอร์ทุกตัว
    totalVoltageDrop = 0.0
    for node in mNodes:
        for ph in ['A', 'B', 'C']:
            totalVoltageDrop += (initialVoltage - abs(node_voltages[node][ph]))

    # --- network load center ของกลุ่ม (จุดอ้างอิง, ไม่ได้บังคับให้ต้องตรง) ---
    netCenter = calculateNetworkLoadCenter(
        meterLocations,
        phase_loads,
        lvLines,
        [],
        conductorResistance,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )
    distanceFromNetworkCenter = float(np.linalg.norm(transformerLocation - netCenter))

    # --- คำนวณ "local load fraction" รอบตำแหน่ง TR ---
    # ใช้ระยะตามขนาดกลุ่มช่วยกำหนด local_radius
    min_x, min_y = meterLocations.min(axis=0)
    max_x, max_y = meterLocations.max(axis=0)
    diag = float(np.hypot(max_x - min_x, max_y - min_y))

    # รัศมีที่ถือว่าเป็น "บริเวณใกล้ TR" (ปรับได้)
    local_radius = max(80.0, 0.25 * diag)

    # ระยะทางจาก TR (tNode) ไปมิเตอร์ทุกตัวตามกราฟ
    try:
        dist_from_tr = nx.single_source_dijkstra_path_length(G, tNode, weight='weight')
    except Exception as e:
        logging.warning(f"Distance calc from TR failed in objectiveFunction: {e}")
        dist_from_tr = {}

    local_load = 0.0
    for m_node, load in meter_node_to_load.items():
        if load <= 0:
            continue
        d = dist_from_tr.get(m_node, None)
        if d is None:
            continue
        if d <= local_radius:
            local_load += load

    local_frac = local_load / total_group_load if total_group_load > 0 else 0.0
    # penalty = 0 เมื่อ local_frac=1 (โหลดอยู่ใกล้ TR ทั้งหมด)
    # penalty สูงสุด เมื่อ local_frac → 0 (TR อยู่ในบริเวณที่แทบไม่มีโหลด)
    local_load_penalty = 1.0 - local_frac

    # --- รวมเป็นคะแนน ---
    if load_center_only:
        voltage_drop_weight = 4.0
        power_loss_weight   = 0.5
        load_center_weight  = 30.0   # ลดจาก 60 ให้ไม่กดมากเกินไป
        local_load_weight   = 80.0   # เพิ่มน้ำหนักให้สนใจโหลดใกล้ TR
    else:
        voltage_drop_weight = 8.0
        power_loss_weight   = 1.0
        load_center_weight  = 20.0
        local_load_weight   = 60.0

    score = (
        voltage_drop_weight * totalVoltageDrop +
        power_loss_weight   * totalPowerLoss +
        load_center_weight  * distanceFromNetworkCenter +
        local_load_weight   * local_load_penalty
    )

    logging.info(
        "Objective at %s -> Vdrop=%.2f, Loss=%.2f, dCenter=%.2f, local_frac=%.3f, score=%.2f",
        transformerLocation, totalVoltageDrop, totalPowerLoss,
        distanceFromNetworkCenter, local_frac, score
    )
    return score


def voltageConstraint(transformerLocation, meterLocations, phase_loads, initialVoltage,
                     conductorResistance, lvLines, powerFactor, conductorReactance=None, 
                     lvData=None, svcLines=None):
    """แก้ไขฟังก์ชันเดิมให้ใช้ coordinate snapping"""
    logging.debug(f"Checking voltage constraint at location={transformerLocation}...")
    
    G, tNode, mNodes, nm, cm = buildLVNetworkWithLoads(
        lvLines, [], meterLocations, transformerLocation, phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=lvData is not None,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )
    # เปลี่ยนไปใช้ calculateUnbalancedPowerFlow แทน calculatePowerLoss
    node_voltages, _, _ = calculateUnbalancedPowerFlow(G, tNode, mNodes, powerFactor, initialVoltage)
    min_voltage_diff = float('inf')
    for node in mNodes:
        for ph in ['A','B','C']:
            diff = abs(node_voltages[node][ph]) - 200  # ใช้ abs เพื่อเอาค่า magnitude ของแรงดัน
            if diff < min_voltage_diff:
                min_voltage_diff = diff
    return min_voltage_diff

def optimizeGroup(meterLocations, phase_loads, initialTransformerLocation,
                  lvLines, mvLines, initialVoltage, conductorResistance,
                  powerFactor, epsilon_junction=1.0, conductorReactance=None, 
                  lvData=None, svcLines=None):
    """
    เลือกตำแหน่งหม้อแปลงบนโหนดของเครือข่าย (LV/MV + service) สำหรับกลุ่มหนึ่ง
    - ล็อกให้ใช้เฉพาะคอมโพเนนต์เดียวกับมิเตอร์ในกลุ่ม
    - กรองโหนดด้วย bounding box + radial + convex hull
    - เพิ่มเงื่อนไข: โหนดต้องมี 'สัดส่วนโหลดใกล้เคียง' เพียงพอ ไม่ใช่ปลายกิ่งโหลดเบา
    """
    logging.info("Optimizing group-level transformer location on existing LV nodes (skip meter nodes)…")

    if len(meterLocations) == 0:
        logging.info("No meters in this group => skip optimization.")
        return None

    # 0) เตรียมข้อมูลโหลดรวมของกลุ่ม
    total_loads_arr = phase_loads['A'] + phase_loads['B'] + phase_loads['C']
    total_group_load = float(total_loads_arr.sum())
    if total_group_load <= 0:
        logging.warning("Group has zero total load; returning initial transformer location.")
        return np.array(initialTransformerLocation, dtype=float)

    # 1) กรอบของกลุ่ม + buffer
    bounds = get_bounding_box(meterLocations)
    min_x, max_x = bounds[0]
    min_y, max_y = bounds[1]
    logging.info(f"Group bounds before buffer: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
    diag = np.hypot(max_x - min_x, max_y - min_y)
    buffer = max(5.0, min(25.0, 0.05 * diag))   # 5% ของเส้นทแยงมุม, หน้าต่าง [5, 25] เมตร
    min_x -= buffer; max_x += buffer; min_y -= buffer; max_y += buffer

    # 2) สร้างกราฟชั่วคราวเฉพาะกลุ่ม (แต่ใช้สายตาม lvLines/mvLines/svcLines เดิม)
    G_temp, tNode_temp, mNodes_temp, node_mapping_temp, coord_mapping_temp = buildLVNetworkWithLoads(
        lvLines, mvLines, meterLocations, initialTransformerLocation, phase_loads, conductorResistance,
        conductorReactance=conductorReactance, use_shape_length=lvData is not None, lvData=lvData, svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )

    # mapping มิเตอร์ -> โหลด (ใช้ index ตรง ๆ)
    meter_node_to_load = {mNodes_temp[i]: float(total_loads_arr[i]) for i in range(len(mNodes_temp))}

    # 3) หา "คอมโพเนนต์หลัก" ของกลุ่มจากมิเตอร์
    comp_id_of = {}
    comps = list(nx.connected_components(G_temp))
    for cid, comp in enumerate(comps):
        for n in comp:
            comp_id_of[n] = cid

    meter_comp_count = {}
    for n in mNodes_temp:
        cid = comp_id_of.get(n, None)
        if cid is not None:
            meter_comp_count[cid] = meter_comp_count.get(cid, 0) + 1

    if meter_comp_count:
        best_cid = max(meter_comp_count, key=meter_comp_count.get)
        comp_nodes = [n for n in comps[best_cid]]
        logging.info(f"Selected component #{best_cid} for this group (meters in comp={meter_comp_count[best_cid]}).")
    else:
        # กรณีสุดวิสัย: ไม่เจอคอมโพเนนต์ของมิเตอร์ (ไม่ควรเกิด) => ใช้ทั้งกราฟ
        comp_nodes = list(G_temp.nodes())
        logging.warning("Meters not found in any component (unexpected) — falling back to all nodes.")

    # 4) จำกัด candidate เฉพาะโหนดในคอมโพเนนต์ + ไม่ใช่ meter node + มีพิกัด
    candidates = [n for n in comp_nodes if (n in coord_mapping_temp) and (n not in mNodes_temp)]

    # 5) กรองด้วย bounding box (+buffer)
    cand_in_bounds = []
    for n in candidates:
        x, y = coord_mapping_temp[n]
        if (min_x <= x <= max_x) and (min_y <= y <= max_y):
            cand_in_bounds.append(n)

    # ถ้าหลังกรองว่าง ให้ย้อนกลับมาใช้โหนดทั้งหมดในคอมโพเนนต์
    if not cand_in_bounds:
        logging.warning("No candidate nodes inside group bounds; using all nodes in component as candidates.")
        cand_in_bounds = candidates[:]
        
    # 6) Radial clamp รอบ centroid ของ 'กลุ่ม'
    cx, cy = meterLocations.mean(axis=0)
    dists  = np.hypot(meterLocations[:,0] - cx, meterLocations[:,1] - cy)
    r_max  = dists.max()                              # รัศมีใหญ่สุดของกลุ่มจริง
    radial_margin = max(3.0, min(15.0, 0.03 * diag)) # เผื่อขอบเล็กน้อย
    radius = r_max + radial_margin
    _cand_backup = cand_in_bounds[:]
    cand_in_bounds = [n for n in cand_in_bounds
                      if np.hypot(coord_mapping_temp[n][0] - cx,
                                  coord_mapping_temp[n][1] - cy) <= radius]
    if not cand_in_bounds:
        logging.warning("Radial clamp removed all candidates; revert to bbox-only candidates.")
        cand_in_bounds = _cand_backup
    
    # 7) Convex-Hull clamp ของกลุ่ม (ขยายเล็กน้อย)
    try:
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path
        if len(meterLocations) >= 3:
            hull = ConvexHull(meterLocations)
            poly = meterLocations[hull.vertices]         # จุดบน hull
            pcx, pcy = poly.mean(axis=0)                 # centroid ของ hull
            scale = 1.08                                 # ขยายออก ~8%
            poly_scaled = np.column_stack([
                pcx + (poly[:,0] - pcx) * scale,
                pcy + (poly[:,1] - pcy) * scale
            ])
            path = Path(poly_scaled)

            _cand_backup2 = cand_in_bounds[:]
            cand_in_bounds = [n for n in cand_in_bounds
                              if path.contains_point(coord_mapping_temp[n])]
            if not cand_in_bounds:
                logging.warning("Hull clamp removed all candidates; revert to radial candidates.")
                cand_in_bounds = _cand_backup2
    except Exception as e:
        logging.warning(f"Hull clamp skipped: {e}")

    # 8) ถ้ายังว่าง ให้กลับไปใช้ candidates ใน component
    if not cand_in_bounds:
        logging.warning("No candidates after clamps; falling back to component nodes.")
        cand_in_bounds = candidates[:]

    # 9) กำหนดรัศมีสำหรับดู 'local load fraction' รอบ candidate
    # ใช้สัดส่วนของขนาดกลุ่ม + minimum
    local_radius = max(80.0, 0.25 * diag)   # ปรับได้ตาม scale network
    min_local_frac = 0.10                   # อย่างน้อยต้องมี >=10% ของโหลดกลุ่มอยู่ใกล้ ๆ

    best_coord = None
    best_score = float('inf')

    for node_id in cand_in_bounds:
        if node_id not in coord_mapping_temp:
            continue

        node_xy = np.array(coord_mapping_temp[node_id], dtype=float)

        # 9.1 คำนวณ score จาก objective เดิม
        raw_score = objectiveFunction(
            node_xy, meterLocations, phase_loads, initialVoltage,
            conductorResistance, lvLines, powerFactor, load_center_only=True,
            conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines
        )

        # 9.2 คำนวณ local load รอบ candidate (เดินบนกราฟ)
        local_load = 0.0
        for m_node, load in meter_node_to_load.items():
            if load <= 0:
                continue
            try:
                d = nx.shortest_path_length(G_temp, node_id, m_node, weight='weight')
            except nx.NetworkXNoPath:
                continue
            if d <= local_radius:
                local_load += load

        local_frac = local_load / total_group_load if total_group_load > 0 else 0.0

        # 9.3 ถ้าโหลดใกล้ ๆ น้อยกว่า threshold ให้ข้าม candidate นี้ (เป็นกิ่งปลายโหลดน้อย)
        if local_frac < min_local_frac:
            logging.debug(
                "Skip candidate node %s at %s : local_frac=%.3f < %.3f",
                node_id, node_xy, local_frac, min_local_frac
            )
            continue

        score = raw_score  # ตอนนี้ใช้ raw_score ตรง ๆ; ถ้าจะ fine tune อนาคตค่อยคูณ penalty factor

        if score < best_score:
            best_score = score
            best_coord = node_xy

    # 10) Fallback: ถ้าทุกรายถูกกรองทิ้ง (เช่น กลุ่ม very small) ใช้ logic เดิม
    if best_coord is None:
        logging.warning(
            "All candidates were filtered out by local-load rule; "
            "falling back to snapping init guess to nearest node in component."
        )
        if candidates:
            pts = np.array([coord_mapping_temp[n] for n in candidates], dtype=float)
            tree = cKDTree(pts)
            guess = np.array(initialTransformerLocation, dtype=float)
            _, idx = tree.query(guess)
            best_coord = pts[idx]
        else:
            logging.error("Component has no candidate nodes; returning initial transformer location.")
            return np.array(initialTransformerLocation, dtype=float)

    logging.info(
        "Discrete node-based optimization (component-locked, local-load-aware) => %s, score=%.2f",
        best_coord, best_score
    )
    return best_coord

def optimize_phase_balance(
    meterLocations: np.ndarray,
    totalLoads:     np.ndarray,
    phase_loads:    dict,
    peano:          np.ndarray,
    lvData,                         # รองรับ None ได้ (จะ fallback)
    original_phases: list[str],
    phase_indexer=None,             # << เพิ่ม แต่ optional; call เดิมไม่กระทบ
    *, 
    tol=1e-9, max_passes=50,
    target_unbalance_pct: float = 10.0
):
    """
    เลือกเฟสให้มิเตอร์เพื่อลด %Unbalance
    - ถ้ามี phase_indexer: ใช้สิทธิ์เฟสจาก JSON
    - ไม่มีก็ลองอ่านจาก lvData (shapefile) แบบโค้ดเดิม
    - ถ้ายังไม่มีข้อมูลสิทธิ์เฟสเลย → อนุญาต ABC ทุกจุด
    """

    # --- helper: %unbalance ---
    def unbalance_pct(PA, PB, PC):
        S = PA + PB + PC
        if S <= 0:
            return 0.0
        avg = S/3.0
        return 100.0 * max(abs(PA-avg), abs(PB-avg), abs(PC-avg)) / avg

    N = len(meterLocations)
    loads = totalLoads.astype(float).copy()

    # --- สร้าง allowed_by_i ---
    allowed_by_i = None

    if phase_indexer is not None:
        # ใช้ indexer จาก JSON ก่อน
        allowed_by_i = [phase_indexer(tuple(meterLocations[i])) for i in range(N)]
        allowed_by_i = [a if a else ['A','B','C'] for a in allowed_by_i]
    else:
        # Fallback: อ่านจาก shapefile แบบเดิม (ถ้ามี)
        try:
            if lvData is not None:
                code_map = {4: "A", 1: "C", 2: "B", 3: "BC", 5: "CA", 6: "AB", 7: "ABC"}

                fields    = [f[0] for f in lvData.fields[1:]]
                recs      = lvData.records()
                shapes    = lvData.shapes()
                if 'PHASEDESIG' in fields:
                    idx_phase = fields.index('PHASEDESIG')
                else:
                    # ไม่มีฟิลด์นี้ → ยอมแพ้ shapefile path
                    raise KeyError("PHASEDESIG not found")

                mids = []
                phase_designs = []
                for rec, shp in zip(recs, shapes):
                    raw = rec[idx_phase]
                    phs = code_map.get(int(raw), "")
                    allowed = [c for c in phs if c in ('A','B','C')]
                    if not allowed:
                        continue
                    mids.append(np.mean(shp.points, axis=0))
                    phase_designs.append(allowed)

                tree = cKDTree(np.vstack(mids)) if len(mids) > 0 else None

                allowed_by_i = []
                for i in range(N):
                    if tree is None:
                        allowed_by_i.append(['A','B','C'])
                        continue
                    _, seg_i = tree.query(meterLocations[i])
                    allowed = phase_designs[int(seg_i)]
                    allowed_by_i.append(allowed if allowed else ['A','B','C'])
        except Exception:
            allowed_by_i = None

    if allowed_by_i is None:
        # ไม่มีข้อมูลทั้ง JSON/shapefile → ABC ทุกจุด
        allowed_by_i = [['A','B','C'] for _ in range(N)]

    # --- เริ่มกระบวนการเลือกเฟส ---
    if original_phases is None:
        original_phases = [''] * N
    else:
        # ให้เป็น string uppercase เสมอ
        original_phases = [str(p).upper() if p is not None else '' for p in original_phases]

    new_phases = [''] * N
    P = {'A':0.0, 'B':0.0, 'C':0.0}

    # 0) anchor ABC
    for i, ph in enumerate(original_phases):
        if set(ph) == {'A','B','C'}:
            new_phases[i] = 'ABC'
            P['A'] += phase_loads['A'][i]
            P['B'] += phase_loads['B'][i]
            P['C'] += phase_loads['C'][i]

    def current_unb():
        return unbalance_pct(P['A'], P['B'], P['C'])

    # 1) greedy (ข้าม ABC) + หยุดเมื่อถึงเป้า
    for i in np.argsort(-loads):
        if new_phases[i] == 'ABC':
            continue
        allowed = allowed_by_i[i]
        if not allowed or loads[i] <= 0:
            continue

        base_choices = allowed[:]
        pref = original_phases[i].strip()
        if pref in ('A','B','C') and pref in base_choices:
            best_phi, best_val = pref, None
            for phi in base_choices:
                tP = P.copy(); tP[phi] += loads[i]
                val = unbalance_pct(tP['A'], tP['B'], tP['C'])
                if (best_val is None) or (val < best_val - 1e-12) or (abs(val - best_val) <= 1e-12 and phi == pref):
                    best_val, best_phi = val, phi
        else:
            best_phi, best_val = None, None
            for phi in base_choices:
                tP = P.copy(); tP[phi] += loads[i]
                val = unbalance_pct(tP['A'], tP['B'], tP['C'])
                if (best_val is None) or (val < best_val):
                    best_val, best_phi = val, phi

        new_phases[i] = best_phi
        P[best_phi] += loads[i]
        if current_unb() < target_unbalance_pct - tol:
            break

    # เติมค่าให้จุดที่ยังว่าง
    for i in range(N):
        if new_phases[i] == '' and loads[i] > 0 and allowed_by_i[i]:
            pref = original_phases[i]
            if pref in ('A','B','C') and pref in allowed_by_i[i]:
                new_phases[i] = pref
                P[pref] += loads[i]
            else:
                pick = allowed_by_i[i][0]
                new_phases[i] = pick
                P[pick] += loads[i]

    # 2) local search – ข้าม ''/ABC และหยุดเมื่อถึงเป้า
    def try_move(i, to_phi):
        cur = new_phases[i]
        if cur not in ('A','B','C') or cur == 'ABC' or cur == to_phi:
            return False, 0.0, cur
        if to_phi not in allowed_by_i[i]:
            return False, 0.0, cur
        old_unb = current_unb()
        P[cur]  -= loads[i]
        P[to_phi] += loads[i]
        new_unb = current_unb()
        if new_unb < old_unb - tol:
            new_phases[i] = to_phi
            return True, (new_unb - old_unb), cur
        # rollback
        P[to_phi] -= loads[i]
        P[cur]    += loads[i]
        return False, (new_unb - old_unb), cur

    passes = 0
    improved_any = True
    while (current_unb() >= target_unbalance_pct - tol) and improved_any and (passes < max_passes):
        improved_any = False
        passes += 1
        for i in np.argsort(-loads):
            if new_phases[i] in ('', 'ABC') or loads[i] <= 0:
                continue
            for phi in allowed_by_i[i]:
                ok, _, _ = try_move(i, phi)
                if ok:
                    improved_any = True
                    if current_unb() < target_unbalance_pct - tol:
                        break
            if current_unb() < target_unbalance_pct - tol:
                break

    best_unb = current_unb()

    # 3) minimize moves – ข้าม ''/ABC และต้องไม่เกินเป้า
    for i in np.argsort(loads):
        cur = new_phases[i]
        if cur in ('', 'ABC'):
            continue
        orig = original_phases[i]
        if (orig == cur) or (orig not in allowed_by_i[i]) or loads[i] <= 0:
            continue
        P[cur]  -= loads[i]
        P[orig] += loads[i]
        new_unb = current_unb()
        if new_unb <= best_unb + tol and new_unb <= target_unbalance_pct - tol:
            new_phases[i] = orig
            best_unb = min(best_unb, new_unb)
        else:
            P[orig] -= loads[i]
            P[cur]  += loads[i]

    # --- build new_phase_loads ---
    new_phase_loads = {'A': np.zeros(N), 'B': np.zeros(N), 'C': np.zeros(N)}
    for i, ph in enumerate(new_phases):
        if ph == 'ABC':
            for c in ('A','B','C'):
                new_phase_loads[c][i] = phase_loads[c][i]
        elif ph in ('A','B','C'):
            new_phase_loads[ph][i] = loads[i]
        else:
            # ไม่ระบุ (โหลดเป็น 0 หรือไม่สามารถกำหนดได้) → คงไว้ 0
            pass

    return new_phases, new_phase_loads


# ---------- PHASE UNBALANCE: before/after ----------

def _phase_totals(phase_loads: dict) -> tuple[float, float, float]:
    """Sum kW per phase from per-meter arrays in phase_loads{'A','B','C'}."""
    PA = float(np.nansum(phase_loads['A']))
    PB = float(np.nansum(phase_loads['B']))
    PC = float(np.nansum(phase_loads['C']))
    return PA, PB, PC

def _unbalance_pct_from_totals(PA: float, PB: float, PC: float) -> float:
    """%Unbalance = max(|Pφ - Pavg|)/Pavg * 100 ; if total==0 => 0"""
    S = PA + PB + PC
    if S <= 0:
        return 0.0
    avg = S/3.0
    return 100.0 * max(abs(PA-avg), abs(PB-avg), abs(PC-avg)) / avg

def compute_unbalance_percent(phase_loads: dict) -> tuple[float, dict]:
    """
    Return (%Unbalance, {'A':PA,'B':PB,'C':PC}) for given phase_loads.
    phase_loads: dict with arrays per phase in kW (like in your pipeline).
    """
    PA, PB, PC = _phase_totals(phase_loads)
    pct = _unbalance_pct_from_totals(PA, PB, PC)
    return pct, {'A': PA, 'B': PB, 'C': PC}

def summarize_unbalance_change(original_phase_loads: dict, new_phase_loads: dict):
    """
    Log and return a summary dict for %Unbalance before vs after reassignment.
    """
    pct_before, totals_before = compute_unbalance_percent(original_phase_loads)
    pct_after,  totals_after  = compute_unbalance_percent(new_phase_loads)

    logging.info(
        f"Phase totals BEFORE (kW): A={totals_before['A']:.2f}, "
        f"B={totals_before['B']:.2f}, C={totals_before['C']:.2f} | %Un={pct_before:.2f}"
    )
    logging.info(
        f"Phase totals AFTER  (kW): A={totals_after['A']:.2f}, "
        f"B={totals_after['B']:.2f}, C={totals_after['C']:.2f} | %Un={pct_after:.2f}"
    )
    logging.info(f"Δ%Unbalance = {pct_after - pct_before:+.2f} (negative = improved)")

    return {
        'before': {'pct_unbalance': pct_before, 'phase_totals_kW': totals_before},
        'after':  {'pct_unbalance': pct_after,  'phase_totals_kW': totals_after},
        'delta_pct_unbalance': pct_after - pct_before
    }

# ---------- NETWORK LOSS: before/after ----------

def _network_loss_W_for_assignment(
    lvLines, mvLines, meterLocations, transformerLocation,
    phase_loads: dict,
    *, conductorResistance: float, powerFactor: float, initialVoltage: float,
    conductorReactance: float | None = None, lvData=None, svcLines=None,
    snap_tolerance: float = None
) -> float:
    """
    Build network with given phase_loads, run unbalanced loadflow, and return total network loss (W).
    Uses your existing buildLVNetworkWithLoads + calculateUnbalancedPowerFlow.
    """
    # ถ้าอยากบังคับ SNAP_TOLERANCE จาก global
    tol = snap_tolerance if snap_tolerance is not None else globals().get('SNAP_TOLERANCE', 0.005)

    G, tNode, mNodes, nm, cm = buildLVNetworkWithLoads(
        lvLines, mvLines, meterLocations, transformerLocation, phase_loads,
        conductorResistance, conductorReactance=conductorReactance,
        use_shape_length=lvData is not None, lvData=lvData, svcLines=svcLines,
        snap_tolerance=tol
    )
    _, _, total_power_loss_W = calculateUnbalancedPowerFlow(
        G, tNode, mNodes, powerFactor, initialVoltage, max_iter=10, tol=1e-3
    )
    return float(total_power_loss_W)

def summarize_loss_change(
    lvLines, mvLines, meterLocations, transformerLocation,
    original_phase_loads: dict, new_phase_loads: dict,
    *, conductorResistance: float, powerFactor: float, initialVoltage: float,
    conductorReactance: float | None = None, lvData=None, svcLines=None,
    snap_tolerance: float = None
):
    """
    Compute network losses BEFORE vs AFTER phase reassignment and log the comparison.
    Returns a dict with loss_before_W, loss_after_W, delta_W and delta_kW.
    """
    loss_before = _network_loss_W_for_assignment(
        lvLines, mvLines, meterLocations, transformerLocation, original_phase_loads,
        conductorResistance=conductorResistance, powerFactor=powerFactor, initialVoltage=initialVoltage,
        conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines,
        snap_tolerance=snap_tolerance
    )
    loss_after = _network_loss_W_for_assignment(
        lvLines, mvLines, meterLocations, transformerLocation, new_phase_loads,
        conductorResistance=conductorResistance, powerFactor=powerFactor, initialVoltage=initialVoltage,
        conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines,
        snap_tolerance=snap_tolerance
    )

    logging.info(f"Network loss BEFORE: {loss_before:.2f} W ({loss_before/1000:.3f} kW)")
    logging.info(f"Network loss AFTER : {loss_after:.2f} W ({loss_after/1000:.3f} kW)")
    logging.info(f"ΔLoss = {loss_after - loss_before:+.2f} W ({(loss_after - loss_before)/1000:+.3f} kW)")

    return {
        'loss_before_W': loss_before,
        'loss_after_W':  loss_after,
        'delta_W':       loss_after - loss_before,
        'loss_before_kW': loss_before/1000.0,
        'loss_after_kW':  loss_after/1000.0,
        'delta_kW':       (loss_after - loss_before)/1000.0
    }

# ---------------------------------
# 17) Export & plotting
def exportPointsToShapefile(point_coords, shapefile_path, attributes_list=None):
    logging.info(f"Exporting {len(point_coords)} point(s) to shapefile: {shapefile_path}...")
    w = shapefile.Writer(shapefile_path, shapeType=shapefile.POINT)
    w.autoBalance = 1
    w.field('FID','N')
    w.field('Name', 'C')
    w.field('X', 'F', decimal=8)
    w.field('Y', 'F', decimal=8)

    for idx, (x, y) in enumerate(point_coords):
        w.point(x, y)
        if attributes_list and idx < len(attributes_list):
            rec = dict(attributes_list[idx])
            fid  = rec.get('FID', idx)
            name = rec.get('Name', f'Point_{idx+1}')
            w.record(fid, str(name), float(x), float(y))
        else:
            # ✅ เติมให้ครบ 4 ฟิลด์เสมอ
            w.record(idx, f'Point_{idx+1}', float(x), float(y))
    w.close()
    logging.info(f"Shapefile saved successfully: {shapefile_path}.shp")

def exportResultDFtoShapefile(result_df, shapefile_path="output_meters.shp"):
    """
    Exports each row of result_df as a point feature in a shapefile.
    Requires result_df to have at least:
      - 'Meter X', 'Meter Y'
      - 'Phases' (เดิม) และ 'New Phase' (ที่คำนวณใหม่) เพื่อสรุปการย้ายเฟส
    จะเพิ่มฟิลด์ NeedMove ('Y'/'N') ต่อจุด
    """
    # Helper: normalize phase string to set('A','B','C')
    def _norm_phase(s):
        if not isinstance(s, str):
            return set()
        s = ''.join(ch for ch in s.upper() if ch in 'ABC')
        return set(s)

    # เตรียมคอลัมน์ FID หากไม่มี
    if 'FID' not in result_df.columns:
        result_df['FID'] = range(len(result_df))

    total_rows = len(result_df)
    move_count = 0

    # ตัวนับต่อกลุ่ม (ภายในฟังก์ชันเท่านั้น)
    group_total = {}  # group -> all meters
    group_move  = {}  # group -> moved meters

    # สร้าง writer
    w = shapefile.Writer(shapefile_path, shapeType=shapefile.POINT)
    w.autoBalance = 1  # geometry-attribute sync

    # ฟิลด์ + NeedMove
    w.field('FID','N')
    w.field('Peano', 'C', size=20)
    w.field('VoltA', 'F', decimal=2)
    w.field('VoltB', 'F', decimal=2)
    w.field('VoltC', 'F', decimal=2)
    w.field('Group', 'C', size=10)
    w.field('Phases', 'C', size=5)
    w.field('NewPhs','C', size=5)
    w.field('LoadA','F', decimal=2)
    w.field('LoadB','F', decimal=2)
    w.field('LoadC','F', decimal=2)
    w.field('MeterX', 'F', decimal=8)
    w.field('MeterY', 'F', decimal=8)
    w.field('NeedMove','C', size=1)

    for idx, row in result_df.iterrows():
        x_coord = float(row["Meter X"])
        y_coord = float(row["Meter Y"])

        # ชื่อกลุ่ม (เว้นว่างให้เป็น 'Ungrouped')
        group_name = str(row.get('Group', '') or '').strip() or 'Ungrouped'
        group_total[group_name] = group_total.get(group_name, 0) + 1

        # ตัดสินใจว่าต้อง "ย้ายเฟส" ไหม
        orig_set = _norm_phase(row.get('Phases', ''))
        new_set  = _norm_phase(row.get('New Phase', ''))
        need_move = 'Y' if (new_set and (new_set != orig_set)) else 'N'
        if need_move == 'Y':
            move_count += 1
            group_move[group_name] = group_move.get(group_name, 0) + 1

        # Geometry
        w.point(x_coord, y_coord)

        # Record
        w.record(
            row.get('FID'),
            row.get('Peano Meter', row.get('Peano','')),
            row.get('Final Voltage A (V)', 0.0),
            row.get('Final Voltage B (V)', 0.0),
            row.get('Final Voltage C (V)', 0.0),
            group_name,
            row.get('Phases', ''),
            row.get('New Phase',''),
            row.get('New Load A', 0.0),
            row.get('New Load B', 0.0),
            row.get('New Load C', 0.0),
            x_coord,
            y_coord,
            need_move
        )

    w.close()
    print(f"Shapefile saved: {shapefile_path} (plus .shx, .dbf).")

    # ---------- LOG SUMMARY (รวม + แยกตามกลุ่ม) ----------
    try:
        pct_all = (move_count / total_rows * 100.0) if total_rows else 0.0
        logging.info(f"Total meters: {total_rows}")
        logging.info(f"Meters needing phase change (ALL): {move_count} ({pct_all:.1f}%)")

        # เรียงชื่อกลุ่ม: Group 1, Group 2, ... แล้วค่อย Ungrouped/อื่น ๆ
        import re
        def _sort_key(g):
            m = re.search(r'(\d+)', g)
            return (0, int(m.group(1))) if m else (1, g.lower())

        for g in sorted(group_total.keys(), key=_sort_key):
            g_total = group_total.get(g, 0)
            g_move  = group_move.get(g, 0)
            g_pct   = (g_move / g_total * 100.0) if g_total else 0.0
            logging.info(f"{g}: {g_move} / {g_total} need move ({g_pct:.1f}%)")

        # สรุปสั้นบรรทัดเดียวให้ชัด ๆ
        compact = ", ".join([f"{g}:{group_move.get(g,0)}" for g in sorted(group_total.keys(), key=_sort_key)])
        logging.info(f"Phase-change by group -> {compact}")
    except Exception as e:
        logging.warning(f"Logging summary failed: {e}")


def plotResults(lvLinesX, lvLinesY, mvLinesX, mvLinesY, eserviceLinesX, eserviceLinesY,
                meterLocations, initialTransformerLocation, optimizedTransformerLocation,
                group1_indices, group2_indices, splitting_point_coords=None, coord_mapping=None,
                optimizedTransformerLocationGroup1=None, optimizedTransformerLocationGroup2=None,
                transformer_losses=None, phases=None, result_df=None, G=None):
    logging.info("Plotting final results...")
    plot_path = "output_plot.png"

    # small helper
    def _is_xy(pt):
        """ตรวจสอบว่าเป็นพิกัด (x, y) ที่ถูกต้องหรือไม่"""
        try:
            # รองรับทั้ง list, tuple, numpy array
            if pt is None:
                return False
            
            # แปลงเป็น array ถ้าจำเป็น
            if isinstance(pt, (list, tuple)):
                arr = np.array(pt)
            elif isinstance(pt, np.ndarray):
                arr = pt
            else:
                return False
            
            # ต้องมี 2 มิติขึ้นไป และไม่เป็น NaN
            if arr.size >= 2:
                x, y = float(arr.flat[0]), float(arr.flat[1])
                return not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y))
            return False
        except (ValueError, TypeError, IndexError):
            return False

    # Delete existing file to prevent conflicts
    if os.path.exists(plot_path):
        os.remove(plot_path)

    if tk._default_root is None:
        _root = tk.Tk()
        _root.withdraw()  # ซ่อนหน้าต่าง root ไม่ให้โชว์ "tk"
    # สร้างหน้าต่าง Toplevel
    plot_window = tk.Toplevel()
    plot_window.title("Meter Locations, Lines, and Transformers")
    plot_window.geometry("1200x900")
       
    # กำหนด protocol เมื่อปิดหน้าต่าง
    def on_close_plot():
        try:
            plt.close(fig)
        except:
            pass
        plot_window.destroy()
    
    plot_window.protocol("WM_DELETE_WINDOW", on_close_plot)

    # สร้าง main frame
    main_frame = tk.Frame(plot_window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # สร้าง frame สำหรับปุ่มควบคุม (อยู่ด้านบน)
    control_frame = tk.Frame(main_frame, bg='lightgray', height=40)
    control_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
    control_frame.pack_propagate(False)

    # สร้าง frame สำหรับ plot
    plot_frame = tk.Frame(main_frame)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    # สร้าง figure และ canvas
    fig = plt.Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)

    # วาดเส้น
    for x, y in zip(lvLinesX or [], lvLinesY or []):
        ax.plot(x, y, color='lime', linewidth=1, linestyle='--',
                label='LV Line' if 'LV Line' not in ax.get_legend_handles_labels()[1] else "")
    for x, y in zip(mvLinesX or [], mvLinesY or []):
        ax.plot(x, y, color='maroon', linewidth=1, linestyle='-.',
                label='MV Line' if 'MV Line' not in ax.get_legend_handles_labels()[1] else "")
    for x, y in zip(eserviceLinesX or [], eserviceLinesY or []):
        ax.plot(x, y, 'm-', linewidth=2,
                label='Eservice Line to TR' if 'Eservice Line' not in ax.get_legend_handles_labels()[1] else "")

    # วาดมิเตอร์ตามกลุ่ม (กันกรณี index ว่าง/None)
    if isinstance(group1_indices, (list, tuple, np.ndarray)) and len(group1_indices) > 0:
        ax.plot(meterLocations[group1_indices, 0], meterLocations[group1_indices, 1],
                'b.', markersize=10, label='Group 1 Meters')
    if isinstance(group2_indices, (list, tuple, np.ndarray)) and len(group2_indices) > 0:
        ax.plot(meterLocations[group2_indices, 0], meterLocations[group2_indices, 1],
                'r.', markersize=10, label='Group 2 Meters')

    # แสดงค่าแรงดันบนมิเตอร์ (ถ้ามี result_df และ phases ตรงยาว)
    if result_df is not None and phases is not None:
        N = len(meterLocations)
        try:
            for i in range(N):
                x = meterLocations[i, 0]; y = meterLocations[i, 1]
                connected_phases = str(phases[i]).upper().strip() if i < len(phases) else ''
                voltage_text = ''
                for ph in ['A', 'B', 'C']:
                    if ph in connected_phases:
                        colname = f'Final Voltage {ph} (V)'
                        if colname in result_df.columns:
                            vval = result_df.iloc[i][colname]
                            try:
                                voltage_text += f'{ph}:{float(vval):.1f}V\n'
                            except Exception:
                                voltage_text += f'{ph}:N/A\n'
                        else:
                            voltage_text += f'{ph}:N/A\n'
                if voltage_text.strip():
                    ax.text(x, y, voltage_text.strip(), fontsize=6, color='black',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        except Exception as e:
            logging.warning(f"Skip meter voltages overlay (reason: {e})")

    # Initial transformer (เช็คก่อน)
    if _is_xy(initialTransformerLocation):
        ix, iy = float(initialTransformerLocation[0]), float(initialTransformerLocation[1])
        ax.plot(ix, iy, 'ko', markersize=10, label='Initial Transformer')
        ax.text(ix, iy, ' Initial Transformer',
                verticalalignment='top', horizontalalignment='left',
                fontsize=10, fontweight='bold')
    else:
        logging.info("Initial transformer location is None/invalid -> skip plotting.")

    # Splitting point
    if _is_xy(splitting_point_coords):
        sx, sy = float(splitting_point_coords[0]), float(splitting_point_coords[1])
        ax.plot(sx, sy, 'ys', markersize=12, label='Splitting Point')
        ax.text(sx, sy, ' Splitting Point', verticalalignment='bottom',
                horizontalalignment='left', fontsize=10, fontweight='bold')

    # Group transformers
    if _is_xy(optimizedTransformerLocationGroup1):
        gx1, gy1 = float(optimizedTransformerLocationGroup1[0]), float(optimizedTransformerLocationGroup1[1])
        ax.plot(gx1, gy1, 'b*', markersize=15, label='Group 1 Transformer')
        ax.text(gx1, gy1, ' Group 1 Transformer',
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=10, fontweight='bold', color='blue')
    if _is_xy(optimizedTransformerLocationGroup2):
        gx2, gy2 = float(optimizedTransformerLocationGroup2[0]), float(optimizedTransformerLocationGroup2[1])
        ax.plot(gx2, gy2, 'r*', markersize=15, label='Group 2 Transformer')
        ax.text(gx2, gy2, ' Group 2 Transformer',
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=10, fontweight='bold', color='red')

    # Loss labels (ถ้ามี)
    if transformer_losses is not None:
        try:
            if _is_xy(optimizedTransformerLocationGroup1) and 'Group 1 Transformer' in transformer_losses:
                x, y = optimizedTransformerLocationGroup1
                ax.text(x, y, f"Group 1 Transformer\nLoss: {transformer_losses['Group 1 Transformer']/1000:.2f} kW",
                        fontsize=8, color='black',
                        bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none', pad=1))
            if _is_xy(optimizedTransformerLocationGroup2) and 'Group 2 Transformer' in transformer_losses:
                x, y = optimizedTransformerLocationGroup2
                ax.text(x, y, f"Group 2 Transformer\nLoss: {transformer_losses['Group 2 Transformer']/1000:.2f} kW",
                        fontsize=8, color='black',
                        bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none', pad=1))
        except Exception as e:
            logging.warning(f"Skip transformer loss labels (reason: {e})")

    # วาด service edges (ถ้ามี)
    if G is not None and coord_mapping is not None:
        try:
            svc_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_service')]
            shown = 'Service-Line' in ax.get_legend_handles_labels()[1]
            for u, v in svc_edges:
                x1, y1 = coord_mapping[u]; x2, y2 = coord_mapping[v]
                ax.plot([x1, x2], [y1, y2], color='purple', linewidth=2,
                        label='' if shown else 'Eserviceline Meter to LVLines')
                shown = True
        except Exception as e:
            logging.warning(f"Skip drawing service edges (reason: {e})")

    # ปรับ legend/title/axes
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    ax.set_title('Meter Locations, Lines, and Transformers')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')

 # pack canvas ใน plot_frame
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # วาดครั้งแรกเพื่อเก็บขอบเขตเดิม
    canvas.draw()
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()
    
    # ใช้ tight_layout
    fig.tight_layout()

    # สร้าง matplotlib toolbar
    try:
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        has_toolbar = True
    except Exception as e:
        logging.warning(f"Could not create matplotlib toolbar: {e}")
        has_toolbar = False

    # ตัวแปรสำหรับเก็บสถานะ
    is_pan_mode = False
    is_zoom_mode = False
    
    # ฟังก์ชัน reset view
    def reset_view():
        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)
        ax.set_aspect('equal')
        canvas.draw()
        if has_toolbar:
            try:
                toolbar.push_current()
            except:
                pass

    # ฟังก์ชัน pan
    def activate_pan():
        nonlocal is_pan_mode, is_zoom_mode
        if has_toolbar:
            if not is_pan_mode:
                toolbar.pan()
                is_pan_mode = True
                is_zoom_mode = False
                pan_btn.config(relief='sunken', bg='lightblue')
                zoom_btn.config(relief='raised', bg='SystemButtonFace')
            else:
                toolbar.pan()
                is_pan_mode = False
                pan_btn.config(relief='raised', bg='SystemButtonFace')

    # ฟังก์ชัน zoom
    def activate_zoom():
        nonlocal is_pan_mode, is_zoom_mode
        if has_toolbar:
            if not is_zoom_mode:
                toolbar.zoom()
                is_zoom_mode = True
                is_pan_mode = False
                zoom_btn.config(relief='sunken', bg='lightgreen')
                pan_btn.config(relief='raised', bg='SystemButtonFace')
            else:
                toolbar.zoom()
                is_zoom_mode = False
                zoom_btn.config(relief='raised', bg='SystemButtonFace')

    # ฟังก์ชัน back
    def go_back():
        if has_toolbar:
            try:
                toolbar.back()
            except:
                pass

    # ฟังก์ชัน forward
    def go_forward():
        if has_toolbar:
            try:
                toolbar.forward()
            except:
                pass

    # สร้างปุ่มควบคุมใน control_frame
    home_btn = tk.Button(control_frame, text="🏠 Home", command=reset_view, 
                        width=8, bg='lightcoral', relief='raised', bd=2, font=('Arial', 9, 'bold'))
    home_btn.pack(side=tk.LEFT, padx=3, pady=3)

    pan_btn = tk.Button(control_frame, text="✋ Pan", command=activate_pan, 
                       width=6, bg='SystemButtonFace', relief='raised', bd=2, font=('Arial', 9, 'bold'))
    pan_btn.pack(side=tk.LEFT, padx=3, pady=3)

    zoom_btn = tk.Button(control_frame, text="🔍 Zoom", command=activate_zoom, 
                        width=7, bg='SystemButtonFace', relief='raised', bd=2, font=('Arial', 9, 'bold'))
    zoom_btn.pack(side=tk.LEFT, padx=3, pady=3)

    # ปุ่ม back และ forward (เฉพาะเมื่อมี toolbar)
    if has_toolbar:
        back_btn = tk.Button(control_frame, text="⬅️ Back", command=go_back, 
                            width=7, bg='lightgray', relief='raised', bd=2, font=('Arial', 9, 'bold'))
        back_btn.pack(side=tk.LEFT, padx=3, pady=3)

        forward_btn = tk.Button(control_frame, text="➡️ Forward", command=go_forward, 
                               width=9, bg='lightgray', relief='raised', bd=2, font=('Arial', 9, 'bold'))
        forward_btn.pack(side=tk.LEFT, padx=3, pady=3)

    # เพิ่ม mouse wheel zoom (ทำงานแม้ไม่มี toolbar)
    def on_scroll(event):
        if event.inaxes != ax:
            return
        
        scale_factor = 1.2
        if event.button == 'up':
            scale = 1 / scale_factor
        elif event.button == 'down':
            scale = scale_factor
        else:
            return
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (xlim[0] + xlim[1])/2
        ydata = event.ydata if event.ydata is not None else (ylim[0] + ylim[1])/2
        
        x_left = xdata - (xdata - xlim[0]) * scale
        x_right = xdata + (xlim[1] - xdata) * scale
        y_bottom = ydata - (ydata - ylim[0]) * scale
        y_top = ydata + (ylim[1] - ydata) * scale
        
        ax.set_xlim([x_left, x_right])
        ax.set_ylim([y_bottom, y_top])
        
        canvas.draw()
        if has_toolbar:
            try:
                toolbar.push_current()
            except:
                pass

    # เชื่อมต่อ scroll event
    canvas.mpl_connect('scroll_event', on_scroll)

    # เพิ่ม simple pan ด้วย mouse (สำรอง)
    last_x, last_y = None, None
    def on_press(event):
        nonlocal last_x, last_y
        if event.button == 2:  # middle mouse button
            last_x, last_y = event.x, event.y

    def on_drag(event):
        nonlocal last_x, last_y
        if event.button == 2 and last_x is not None and last_y is not None:
            dx = event.x - last_x
            dy = event.y - last_y
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            scale_x = (xlim[1] - xlim[0]) / canvas.get_tk_widget().winfo_width()
            scale_y = (ylim[1] - ylim[0]) / canvas.get_tk_widget().winfo_height()
            
            ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            ax.set_ylim(ylim[0] + dy * scale_y, ylim[1] + dy * scale_y)
            
            canvas.draw()
            last_x, last_y = event.x, event.y

    def on_release(event):
        nonlocal last_x, last_y
        if event.button == 2:
            last_x, last_y = None, None

    canvas.mpl_connect('button_press_event', on_press)
    canvas.mpl_connect('motion_notify_event', on_drag)
    canvas.mpl_connect('button_release_event', on_release)

    # เพิ่ม label แสดงคำแนะนำ
    if has_toolbar:
        info_text = "💡 Use buttons above | Mouse wheel = zoom | Middle mouse = pan | Full toolbar below"
    else:
        info_text = "💡 Use buttons above | Mouse wheel = zoom | Middle mouse = pan"
    
    info_label = tk.Label(control_frame, text=info_text, 
                         bg='lightyellow', font=('Arial', 8))
    info_label.pack(side=tk.RIGHT, padx=5, pady=3)

    # บันทึกไฟล์
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    plot_path = f"{folder_path}/output_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_path}")

    # ข้อมูลการทำงาน
    if has_toolbar:
        logging.info("Navigation toolbar created successfully")
    else:
        logging.info("Using basic navigation controls only")


# ---------------------------------
# 18) Transformer sizing & losses
def growthRate(g2_load, annual_growth=0.06, years=4):
    logging.debug(f"Calculating growth rate for load={g2_load:.2f}, annual={annual_growth}, years={years}")
    future_g2_load = g2_load
    for _ in range(years):
        future_g2_load *= (1 + annual_growth)
    return future_g2_load

def Lossdocument(sum_load_kW):
    transformer_table = [
        {'rating_kVA':  30, 'no_load_loss': 0.12, 'load_loss':  0.430},
        {'rating_kVA':  50, 'no_load_loss': 0.11, 'load_loss':  0.875},
        {'rating_kVA': 100, 'no_load_loss': 0.15, 'load_loss': 1.450},
        {'rating_kVA': 160, 'no_load_loss': 0.26, 'load_loss': 2.0},
        {'rating_kVA': 250, 'no_load_loss': 0.36, 'load_loss': 2.750},
        {'rating_kVA': 315, 'no_load_loss': 0.44, 'load_loss': 3.250},
        {'rating_kVA': 400, 'no_load_loss': 0.52, 'load_loss': 3.850},
        {'rating_kVA': 500, 'no_load_loss': 0.61, 'load_loss': 4.600}
    ]
    powerFactor = 0.875
    load_kVA = sum_load_kW / powerFactor
    selected = None
    for row in transformer_table:
        if row['rating_kVA'] >= load_kVA:
            selected = row
            break
    if selected is None:
        selected = transformer_table[-1]
    return selected['rating_kVA']

def get_transformer_losses(group_load_kW):
    transformer_table = [
        {'rating_kVA':  30, 'no_load_loss': 0.12, 'load_loss':  0.430},
        {'rating_kVA':  50, 'no_load_loss': 0.11, 'load_loss':  0.875},
        {'rating_kVA': 100, 'no_load_loss': 0.15, 'load_loss': 1.450},
        {'rating_kVA': 160, 'no_load_loss': 0.26, 'load_loss': 2.0},
        {'rating_kVA': 250, 'no_load_loss': 0.36, 'load_loss': 2.750},
        {'rating_kVA': 315, 'no_load_loss': 0.44, 'load_loss': 3.250},
        {'rating_kVA': 400, 'no_load_loss': 0.52, 'load_loss': 3.850},
        {'rating_kVA': 500, 'no_load_loss': 0.61, 'load_loss': 4.600}
    ]
    powerFactor = 0.875
    rating_kVA = Lossdocument(group_load_kW)
    selected = None
    for row in transformer_table:
        if row['rating_kVA'] == rating_kVA:
            selected = row
            break
    if selected is None:
        selected = transformer_table[-1]
    no_load_loss_kW = selected['no_load_loss']
    full_load_loss_kW = selected['load_loss']
    load_kVA = group_load_kW / powerFactor
    load_ratio = load_kVA / rating_kVA
    actual_copper_loss_kW = full_load_loss_kW * (load_ratio ** 2)
    total_tx_loss_kW = no_load_loss_kW + actual_copper_loss_kW
    return total_tx_loss_kW

# ---------------------------------
# 19) Graph labeling & plotting
def addNodeLabels(G, splitting_point_node, best_edge_diff):
    logging.info("Adding labels to nodes in graph G.")
    for node in G.nodes():
        if node == splitting_point_node:
            G.nodes[node]['label'] = f"Node {node}\nEdge Diff: {best_edge_diff:.2f}"
        else:
            G.nodes[node]['label'] = f"Node {node}"
    return G

# plt close ไว้อยู่
def plotGraphWithLabels(G, coord_mapping, best_edge=None, best_edge_diff=None):
    logging.info("Plotting graph with node labels...")
    pos = {node: coord_mapping[node] for node in G.nodes() if node in coord_mapping}
    labels = nx.get_node_attributes(G, 'label')

    # ปลอดภัยเมื่อ best_edge = None
    highlight_nodes = set(best_edge) if best_edge else set()
    node_colors = ['red' if node in highlight_nodes else 'lightblue' for node in G.nodes()]

    plt.figure(figsize=(10,8))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=50, edge_color='green')
    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    if best_edge:
        x1, y1 = pos[best_edge[0]]
        x2, y2 = pos[best_edge[1]]
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)

    plt.title("Graph with Node Labels")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.close()
    logging.info("Graph plotted with labels.")


# ---------------------------------
# 20) Re-execution & nested split
def rerun_process(candidate_index,
                  G, transformerNode, meterNodes, node_mapping, coord_mapping,
                  meterLocations, totalLoads, phase_loads,
                  lvLines, mvLines, filteredEserviceLines,
                  initialTransformerLocation, powerFactor,
                  initialVoltage, conductorResistance,
                  peano, phases,
                  conductorReactance=None, lvData=None, svcLines=None, snap_tolerance=0.1):

    # ใช้เฉพาะที่จำเป็นจาก global เหมือนเดิม
    global SNAP_TOLERANCE, latest_split_result
    if globals().get('tk_progress'):
        tk_progress.start(4, stage="Re-execute")

    logging.info(f"Re-executing with candidate_index={candidate_index}")

    # 1) หา splitting point ใหม่
    best_edge, sp_coord, sp_edge_diff, candidate_edges = findSplittingPoint(
        G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, candidate_index
    )
    if best_edge is None:
        logging.error(f"No valid splitting edge for candidate_index={candidate_index}")
        if globals().get('tk_progress'): tk_progress.finish("Failed")
        return None
    if globals().get('tk_progress'): tk_progress.step()

    # 2) แบ่งเครือข่ายเป็น 2 กลุ่ม
    group1_nodes, group2_nodes = partitionNetworkAtPoint(G, transformerNode, meterNodes, best_edge)
    if globals().get('tk_progress'): tk_progress.step()

    # 3) run sweep แยกโหลด (เหมือน main)
    voltages, branch_curr, group1_nodes, group2_nodes = performForwardBackwardSweepAndDivideLoads(
        G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, best_edge
    )
    if globals().get('tk_progress'): tk_progress.step()

    # 4) map เป็น index ของมิเตอร์
    nodeToIndex = {mn: i for i, mn in enumerate(meterNodes)}
    g1_idx = [nodeToIndex[n] for n in group1_nodes if n in nodeToIndex]
    g2_idx = [nodeToIndex[n] for n in group2_nodes if n in nodeToIndex]

    group1_meterLocs = meterLocations[g1_idx] if g1_idx else np.empty((0,2))
    group2_meterLocs = meterLocations[g2_idx] if g2_idx else np.empty((0,2))
    loads_g1 = totalLoads[g1_idx] if g1_idx else np.array([])
    loads_g2 = totalLoads[g2_idx] if g2_idx else np.array([])

    group1_phase_loads = {
        'A': phase_loads['A'][g1_idx] if g1_idx else np.array([]),
        'B': phase_loads['B'][g1_idx] if g1_idx else np.array([]),
        'C': phase_loads['C'][g1_idx] if g1_idx else np.array([]),
    }
    group2_phase_loads = {
        'A': phase_loads['A'][g2_idx] if g2_idx else np.array([]),
        'B': phase_loads['B'][g2_idx] if g2_idx else np.array([]),
        'C': phase_loads['C'][g2_idx] if g2_idx else np.array([]),
    }

    # สำเนา "ก่อน balance" ไว้ใช้เทียบและสรุป
    orig_group1_phase_loads = {
        'A': group1_phase_loads['A'].copy(), 'B': group1_phase_loads['B'].copy(), 'C': group1_phase_loads['C'].copy()
    }
    orig_group2_phase_loads = {
        'A': group2_phase_loads['A'].copy(), 'B': group2_phase_loads['B'].copy(), 'C': group2_phase_loads['C'].copy()
    }

    # 5) Log load balance BEFORE (ตาม main)
    def _sum3(ph): return float(np.nansum(ph['A']) + np.nansum(ph['B']) + np.nansum(ph['C']))
    def _totals(ph):
        return float(np.nansum(ph['A'])), float(np.nansum(ph['B'])), float(np.nansum(ph['C']))
    def _unb_pct(PA, PB, PC):
        S = PA + PB + PC
        if S <= 0: return 0.0
        avg = S/3.0
        return 100.0 * max(abs(PA-avg), abs(PB-avg), abs(PC-avg)) / avg

    if len(g1_idx):
        g1A, g1B, g1C = _totals(orig_group1_phase_loads)
        logging.info("Group 1 load balance before -> A: %.1f kW, B: %.1f kW, C: %.1f kW", g1A, g1B, g1C)
        logging.info("Group 1 percent unbalance before: %.2f%%", _unb_pct(g1A, g1B, g1C))

    if len(g2_idx):
        g2A, g2B, g2C = _totals(orig_group2_phase_loads)
        logging.info("Group 2 load balance before -> A: %.1f kW, B: %.1f kW, C: %.1f kW", g2A, g2B, g2C)
        logging.info("Group 2 percent unbalance before: %.2f%%", _unb_pct(g2A, g2B, g2C))

    # 6) Balance เฟสจน %Unbalance < 10 (ผลลัพธ์ AFTER) — แล้ว "ทับ" สำหรับขั้นตอนถัดไป ให้เหมือน main
    try:
        peano_g1  = peano[g1_idx] if g1_idx else np.array([])
        phases_g1 = phases[g1_idx] if g1_idx else np.array([])
        peano_g2  = peano[g2_idx] if g2_idx else np.array([])
        phases_g2 = phases[g2_idx] if g2_idx else np.array([])

        if len(group1_meterLocs) >= 1 and lvData is not None:
            new_phases_g1, new_phase_loads_g1 = optimize_phase_balance(
                group1_meterLocs, loads_g1, orig_group1_phase_loads, peano_g1, lvData, phases_g1,
                target_unbalance_pct=10.0
            )
        else:
            new_phases_g1, new_phase_loads_g1 = None, orig_group1_phase_loads

        if len(group2_meterLocs) >= 1 and lvData is not None:
            new_phases_g2, new_phase_loads_g2 = optimize_phase_balance(
                group2_meterLocs, loads_g2, orig_group2_phase_loads, peano_g2, lvData, phases_g2,
                target_unbalance_pct=10.0
            )
        else:
            new_phases_g2, new_phase_loads_g2 = None, orig_group2_phase_loads
    except Exception as e:
        logging.warning(f"Phase balance step skipped: {e}")
        new_phases_g1, new_phase_loads_g1 = None, orig_group1_phase_loads
        new_phases_g2, new_phase_loads_g2 = None, orig_group2_phase_loads

    # Log AFTER (เหมือน main)
    if len(g1_idx):
        g1A2, g1B2, g1C2 = _totals(new_phase_loads_g1)
        logging.info("Group 1 load balance after  -> A: %.1f kW, B: %.1f kW, C: %.1f kW", g1A2, g1B2, g1C2)
        logging.info("Group 1 percent unbalance after: %.2f%%", _unb_pct(g1A2, g1B2, g1C2))
        unb_g1_summary = summarize_unbalance_change(orig_group1_phase_loads, new_phase_loads_g1)
        logging.info("Group 1 %%Unbalance summary: %s", unb_g1_summary)

    if len(g2_idx):
        g2A2, g2B2, g2C2 = _totals(new_phase_loads_g2)
        logging.info("Group 2 load balance after  -> A: %.1f kW, B: %.1f kW, C: %.1f kW", g2A2, g2B2, g2C2)
        logging.info("Group 2 percent unbalance after: %.2f%%", _unb_pct(g2A2, g2B2, g2C2))
        unb_g2_summary = summarize_unbalance_change(orig_group2_phase_loads, new_phase_loads_g2)
        logging.info("Group 2 %%Unbalance summary: %s", unb_g2_summary)

    # ใช้ค่า AFTER ต่อในทุกขั้นตอนถัดไป (เหมือน main)
    group1_phase_loads = new_phase_loads_g1
    group2_phase_loads = new_phase_loads_g2

    # 7) สรุปโหลดรวมต่อกลุ่ม (AFTER)
    g1_load = _sum3(group1_phase_loads) if len(g1_idx) else 0.0
    g2_load = _sum3(group2_phase_loads) if len(g2_idx) else 0.0
    logging.info(f"Group 1 total load: {g1_load:.2f} kW | Group 2 total load: {g2_load:.2f} kW")

    # 8) Optimize TX location per group (เหมือน main) — ใช้ sp_coord เป็นจุดตั้งต้น
    opt_tr_g1 = optimizeGroup(
        group1_meterLocs, group1_phase_loads,
        calculateNetworkLoadCenter(group1_meterLocs, group1_phase_loads, lvLines, mvLines, conductorResistance,
                                   conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines),
        lvLines, mvLines, initialVoltage, conductorResistance, powerFactor,
        epsilon_junction=2.0, conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines
    ) if len(g1_idx) else None
        # อัพเดท voltage ใน result_df
        
    opt_tr_g2 = optimizeGroup(
        group2_meterLocs, group2_phase_loads,
        calculateNetworkLoadCenter(group2_meterLocs, group2_phase_loads, lvLines, mvLines, conductorResistance,
                                   conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines),
        lvLines, mvLines, initialVoltage, conductorResistance, powerFactor,
        epsilon_junction=2.0, conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines
    ) if len(g2_idx) else None
        
        
    # 9) Build network per group + คำนวณ line loss และ max distance (เหมือน main)
    line_loss_g1_kW = 0.0
    line_loss_g2_kW = 0.0
    max_dist1 = 0.0
    max_dist2 = 0.0

    if opt_tr_g1 is not None and len(g1_idx):
        G_g1, tNode_g1, mNodes_g1, nm_g1, cm_g1 = buildLVNetworkWithLoads(
            lvLines, mvLines, group1_meterLocs, opt_tr_g1, group1_phase_loads, conductorResistance,
            conductorReactance=conductorReactance, use_shape_length=True, lvData=lvData, svcLines=svcLines
        )
        # unbalanced PF -> line loss
        node_voltages_g1, branch_currents_g1, total_power_loss_g1 = calculateUnbalancedPowerFlow(
            G_g1, tNode_g1, mNodes_g1, powerFactor, initialVoltage
        )
        line_loss_g1_kW = float(total_power_loss_g1) / 1000.0
        
        # distance
        d1 = []
        for mNode in mNodes_g1:
            try:
                d1.append(nx.shortest_path_length(G_g1, tNode_g1, mNode, weight='weight'))
            except nx.NetworkXNoPath:
                d1.append(float('inf'))
        max_dist1 = max(d1) if d1 else 0.0

        
    if opt_tr_g2 is not None and len(g2_idx):
        G_g2, tNode_g2, mNodes_g2, nm_g2, cm_g2 = buildLVNetworkWithLoads(
            lvLines, mvLines, group2_meterLocs, opt_tr_g2, group2_phase_loads, conductorResistance,
            conductorReactance=conductorReactance, use_shape_length=True, lvData=lvData, svcLines=svcLines
        )
        node_voltages_g2, branch_currents_g2, total_power_loss_g2 = calculateUnbalancedPowerFlow(
            G_g2, tNode_g2, mNodes_g2, powerFactor, initialVoltage
        )
        line_loss_g2_kW = float(total_power_loss_g2) / 1000.0
        
        d2 = []
        for mNode in mNodes_g2:
            try:
                d2.append(nx.shortest_path_length(G_g2, tNode_g2, mNode, weight='weight'))
            except nx.NetworkXNoPath:
                d2.append(float('inf'))
        max_dist2 = max(d2) if d2 else 0.0

          
    
    
    # 10) Loss summary BEFORE vs AFTER (ใช้ helper ของคุณ)
    loss_g1_summary = summarize_loss_change(
        lvLines, mvLines, group1_meterLocs, opt_tr_g1,
        orig_group1_phase_loads, group1_phase_loads,
        conductorResistance=conductorResistance, powerFactor=powerFactor, initialVoltage=initialVoltage,
        conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines, snap_tolerance=snap_tolerance
    ) if len(g1_idx) else {'loss_before_kW':0.0, 'loss_after_kW':0.0}

    loss_g2_summary = summarize_loss_change(
        lvLines, mvLines, group2_meterLocs, opt_tr_g2,
        orig_group2_phase_loads, group2_phase_loads,
        conductorResistance=conductorResistance, powerFactor=powerFactor, initialVoltage=initialVoltage,
        conductorReactance=conductorReactance, lvData=lvData, svcLines=svcLines, snap_tolerance=snap_tolerance
    ) if len(g2_idx) else {'loss_before_kW':0.0, 'loss_after_kW':0.0}

    # 11) TX loss (เหมือน main ใช้โหลดปัจจุบัน)
    tx_loss_g1_kW = get_transformer_losses(g1_load) if g1_load > 0 else 0.0
    tx_loss_g2_kW = get_transformer_losses(g2_load) if g2_load > 0 else 0.0

    total_system_loss_g1 = line_loss_g1_kW + tx_loss_g1_kW
    total_system_loss_g2 = line_loss_g2_kW + tx_loss_g2_kW

    # 12) Future load & TX sizing (เหมือน main)
    future_g1_load = growthRate(g1_load, annual_growth=0.06, years=4) if g1_load > 0 else 0.0
    future_g2_load = growthRate(g2_load, annual_growth=0.06, years=4) if g2_load > 0 else 0.0
    rating_g1 = Lossdocument(future_g1_load) if future_g1_load > 0 else 0
    rating_g2 = Lossdocument(future_g2_load) if future_g2_load > 0 else 0

    # 13) Log ให้เหมือน main
    logging.info('############ Result ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW, Future growthrate(4yr@6%)={future_g1_load:.2f} kW, "
                 f"Chosen TX1={rating_g1} kVA, Max Distance Group1={max_dist1:.1f}m.")
    logging.info(f"Group 2 => Load={g2_load:.2f} kW, Future growthrate(4yr@6%)={future_g2_load:.2f} kW, "
                 f"Chosen TX2={rating_g2} kVA, Max Distance Group2={max_dist2:.1f}m.")

    logging.info('############ Loss Report ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW | LineLoss={line_loss_g1_kW:.2f} kW | "
                 f"TxLoss={tx_loss_g1_kW:.2f} kW => TOTAL={total_system_loss_g1:.2f} kW")
    logging.info("Group 1 line loss: BEFORE %.2f kW -> AFTER %.2f kW",
                 loss_g1_summary.get("loss_before_kW", 0.0), loss_g1_summary.get("loss_after_kW", 0.0))

    logging.info(f"Group 2 => Load={g2_load:.2f} kW | LineLoss={line_loss_g2_kW:.2f} kW | "
                 f"TxLoss={tx_loss_g2_kW:.2f} kW => TOTAL={total_system_loss_g2:.2f} kW")
    logging.info("Group 2 line loss: BEFORE %.2f kW -> AFTER %.2f kW",
                 loss_g2_summary.get("loss_before_kW", 0.0), loss_g2_summary.get("loss_after_kW", 0.0))

    logging.info("####### %%UnBalance Report ######")
    logging.info(
        "Group 1 %%Unbalance: BEFORE %.2f%% -> AFTER %.2f%%",
        unb_g1_summary['before']['pct_unbalance'],
        unb_g1_summary['after']['pct_unbalance']
    )
    logging.info(
        "Group 2 %%Unbalance: BEFORE %.2f%% -> AFTER %.2f%%",
        unb_g2_summary['before']['pct_unbalance'],
        unb_g2_summary['after']['pct_unbalance']
    )
    # 14) เก็บผลไว้ใน latest_split_result (เหมือนเดิม)
    result = {
        'candidate_index': candidate_index,
        'best_edge': best_edge,
        'splitting_point': sp_coord,
        'best_edge_diff': float(sp_edge_diff),
        'group1_idx': g1_idx,
        'group2_idx': g2_idx,
        'opt_tr_g1': opt_tr_g1,
        'opt_tr_g2': opt_tr_g2,
        'group1_load_now_kW': g1_load,
        'group2_load_now_kW': g2_load,
        'group1_load_future_kW': future_g1_load,
        'group2_load_future_kW': future_g2_load,
        'group1_rating_kVA': rating_g1,
        'group2_rating_kVA': rating_g2,
        'tx_losses': {
            'Group 1 Transformer': tx_loss_g1_kW,
            'Group 2 Transformer': tx_loss_g2_kW
        }
    }
    latest_split_result = result

    if globals().get('tk_progress'):
        tk_progress.finish("Done")

    logging.info("Re-execution complete.")
    # สร้างกราฟใหม่หลังจาก re-run
    g1_plot_indices = np.array(result['group1_idx'], dtype=int)
    g2_plot_indices = np.array(result['group2_idx'], dtype=int)
    
    # helper สำหรับแปลง lines เป็น X, Y lists
    def _xs(lines): return [l['X'] for l in lines] if lines else []
    def _ys(lines): return [l['Y'] for l in lines] if lines else []
    
    # เตรียมข้อมูล result_df (ถ้ามี)
    # สร้าง result_df ใหม่จากข้อมูลที่คำนวณใหม่
    result_df_new = pd.DataFrame({
        'Peano Meter': peano,
        'Distance to Transformer (m)': np.zeros(len(meterLocations)),  # จะอัพเดทภายหลัง
        'Group': ['Group 1' if i in g1_plot_indices else 'Group 2' for i in range(len(meterLocations))],
        'Phases': phases,
        'Meter X': meterLocations[:, 0],
        'Meter Y': meterLocations[:, 1],
        'Final Voltage A (V)': np.nan,
        'Final Voltage B (V)': np.nan,
        'Final Voltage C (V)': np.nan,
        'New Phase': '',
        'New Load A': np.nan,
        'New Load B': np.nan,
        'New Load C': np.nan,
    })
    
    # เพิ่มข้อมูล phase loads ใหม่
    for i in g1_plot_indices:
        local_i = np.where(g1_plot_indices == i)[0][0]
        result_df_new.at[i, 'New Phase'] = new_phases_g1[local_i] if new_phases_g1 else ''
        result_df_new.at[i, 'New Load A'] = new_phase_loads_g1['A'][local_i]
        result_df_new.at[i, 'New Load B'] = new_phase_loads_g1['B'][local_i]
        result_df_new.at[i, 'New Load C'] = new_phase_loads_g1['C'][local_i]
    
    for i in g2_plot_indices:
        local_i = np.where(g2_plot_indices == i)[0][0]
        result_df_new.at[i, 'New Phase'] = new_phases_g2[local_i] if new_phases_g2 else ''
        result_df_new.at[i, 'New Load A'] = new_phase_loads_g2['A'][local_i]
        result_df_new.at[i, 'New Load B'] = new_phase_loads_g2['B'][local_i]
        result_df_new.at[i, 'New Load C'] = new_phase_loads_g2['C'][local_i]
    
    for i, node in enumerate(mNodes_g1):
            global_idx = g1_idx[i]
            result_df_new.at[global_idx, 'Final Voltage A (V)'] = abs(node_voltages_g1[node]['A'])
            result_df_new.at[global_idx, 'Final Voltage B (V)'] = abs(node_voltages_g1[node]['B'])
            result_df_new.at[global_idx, 'Final Voltage C (V)'] = abs(node_voltages_g1[node]['C'])
    
    for i, node in enumerate(mNodes_g2):
            global_idx = g2_idx[i]
            result_df_new.at[global_idx, 'Final Voltage A (V)'] = abs(node_voltages_g2[node]['A'])
            result_df_new.at[global_idx, 'Final Voltage B (V)'] = abs(node_voltages_g2[node]['B'])
            result_df_new.at[global_idx, 'Final Voltage C (V)'] = abs(node_voltages_g2[node]['C'])
    
    # เรียก plotResults พร้อมข้อมูลใหม่
    plotResults(
        _xs(lvLines), _ys(lvLines),
        _xs(mvLines), _ys(mvLines),
        _xs(filteredEserviceLines), _ys(filteredEserviceLines),
        meterLocations,
        initialTransformerLocation=initialTransformerLocation,
        optimizedTransformerLocation=result['opt_tr_g1'],  # ใช้ Group 1 เป็นตัวหลัก
        group1_indices=g1_plot_indices,
        group2_indices=g2_plot_indices,
        splitting_point_coords=result['splitting_point'],
        coord_mapping=coord_mapping,
        optimizedTransformerLocationGroup1=result['opt_tr_g1'],
        optimizedTransformerLocationGroup2=result['opt_tr_g2'],
        phases=phases,
        result_df=result_df_new,
        G=G
    )
    
    logging.info("New plot generated after re-execution.")
    # ============ จบส่วนที่เพิ่ม ============
    
    return result



# SECOND‑LEVEL SPLIT + OPTIMISE  (re‑use existing primitives)
def refine_group_by_nested_split(group_name, meter_locs, phase_loads,
                                 lv_lines, mv_lines,
                                 init_tx_location, initial_voltage,
                                 conductor_resistance, power_factor,
                                 candidate_index=0,
                                 distance_threshold=800, max_loops=10,
                                 conductor_reactance=None, lv_data=None, svc_lines=None):

    log_hdr = f"[{group_name}]"
    logging.info(f"{log_hdr} Build graph for nested split …")

    # กราฟย่อยของกลุ่มเดิม
    Gg, tNode_g, mNodes_g, nm_g, cm_g = buildLVNetworkWithLoads(
        lv_lines, mv_lines, meter_locs,
        init_tx_location, phase_loads, conductor_resistance,
        conductorReactance=conductor_reactance,
        use_shape_length=lv_data is not None,
        lvData=lv_data,
        svcLines=svc_lines
    )

    # เลือก edge ตาม candidate_index บนกราฟย่อย
    best_edge, split_xy, edge_diff, _ = findSplittingPoint(
        Gg, tNode_g, mNodes_g, cm_g,
        power_factor, initial_voltage, candidate_index)

    if best_edge is None:
        logging.warning(f"{log_hdr} No inner split edge (idx={candidate_index}).")
        return None

    # ------- แบ่งกลุ่มแล้ว optimize ต่อ (เหมือนเดิม) -------
    A_nodes, B_nodes = partitionNetworkAtPoint(Gg, tNode_g, mNodes_g, best_edge)
    node2idx = {n: i for i, n in enumerate(mNodes_g)}
    A_idx = np.fromiter((node2idx[n] for n in A_nodes if n in node2idx), int)
    B_idx = np.fromiter((node2idx[n] for n in B_nodes if n in node2idx), int)

    def _opt(idx):
        if idx.size == 0:
            return None
        sub_loc = meter_locs[idx]
        sub_pl  = {ph: phase_loads[ph][idx] for ph in 'ABC'}
        init_guess = calculateNetworkLoadCenter(
            sub_loc, sub_pl, lv_lines, mv_lines, conductor_resistance,
            conductorReactance=conductor_reactance, lvData=lv_data, svcLines=svc_lines
        )
        return optimizeGroup(
            sub_loc, sub_pl, init_guess, lv_lines, mv_lines,
            initial_voltage, conductor_resistance, power_factor, 
            epsilon_junction=2.0, conductorReactance=conductor_reactance,
            lvData=lv_data, svcLines=svc_lines
        )

    tx_A = _opt(A_idx)
    tx_B = _opt(B_idx)
    logging.info(f"{log_hdr} idx={candidate_index}  A-meters={len(A_idx)}  B-meters={len(B_idx)}")

    return {'edge': best_edge, 'split_coord': split_xy,
            'subA': {'idx': A_idx, 'tx': tx_A},
            'subB': {'idx': B_idx, 'tx': tx_B}}

### 7. แก้ไขฟังก์ชัน summarise_subgroup

def summarise_subgroup(tag, idx_set, tx_loc,
                       meter_locs, phase_loads,
                       lv_lines, mv_lines,
                       init_V, R_cond, pf,
                       conductor_reactance=None, lv_data=None, svc_lines=None):

    if idx_set.size == 0 or tx_loc is None:
            logging.warning(f"{tag}: subgroup empty or TX not found.")
            return None, None

    sub_m_locs = meter_locs[idx_set]
    sub_pl     = {ph: phase_loads[ph][idx_set] for ph in ['A', 'B', 'C']}

    G_sub, t_sub, m_sub, _, cm_sub = buildLVNetworkWithLoads(
        lv_lines, mv_lines, sub_m_locs, tx_loc, sub_pl, R_cond,
        conductorReactance=conductor_reactance,
        use_shape_length=lv_data is not None,
        lvData=lv_data,
        svcLines=svc_lines
    )
    
    # เปลี่ยนไปใช้ calculateUnbalancedPowerFlow แทน calculatePowerLoss
    _, _, line_loss_W = calculateUnbalancedPowerFlow(
        G_sub, t_sub, m_sub, pf, init_V
    )
    
    line_loss_kW = line_loss_W / 1000.0
    P_sub        = (sub_pl['A'] + sub_pl['B'] + sub_pl['C']).sum()
    tx_loss_kW   = get_transformer_losses(P_sub)
    total_loss   = line_loss_kW + tx_loss_kW
    rating_kVA   = Lossdocument(P_sub)
    P_future     = growthRate(P_sub, 0.04, 4)

    result_line = (f"{tag} => Load={P_sub:.2f} kW, "
                   f"Future growthrate(4yr@4%)={P_future:.2f} kW, "
                   f"Chosen TX={rating_kVA} kVA")

    loss_line   = (f"{tag} => Load={P_sub:.2f} kW, "
                   f"LineLoss={line_loss_kW:.2f} kW, "
                   f"TxLoss={tx_loss_kW:.2f} kW, "
                   f"TOTAL={total_loss:.2f} kW")

    return result_line, loss_line

def run_nested_optimisation(last_result,
                            lv_lines, mv_lines,
                            initial_voltage, conductor_resistance,
                            power_factor, conductor_reactance=None,
                            lv_data=None, svc_lines=None):

    """
    split_result = dict returned by rerun_process() containing
    group1 / group2 meter sets and original TX candidates.
    """
    if last_result is None:
        messagebox.showwarning("No data",
                               "ยังไม่มีข้อมูลการแบ่งกลุ่ม – กรุณา Run Process ก่อน")
        return
 
    g1 = last_result['group1']
    g2 = last_result['group2']
            # refine Group 1
    new_tx1 = refine_group_by_nested_split(
                "Group1 Refine",
                meter_locs           = g1['meter_locs'],
                phase_loads          = g1['phase_loads'],
                lv_lines             = lv_lines,
                mv_lines             = mv_lines,
                init_tx_location     = g1['tx_loc'],
                initial_voltage      = initial_voltage,
                conductor_resistance = conductor_resistance,
                power_factor         = power_factor,
                conductor_reactance  = conductor_reactance,
                lv_data              = lv_data,
                svc_lines            = svc_lines
            )

    new_tx2 = refine_group_by_nested_split(
                "Group2 Refine",
                meter_locs           = g2['meter_locs'],
                phase_loads          = g2['phase_loads'],
                lv_lines             = lv_lines,
                mv_lines             = mv_lines,
                init_tx_location     = g2['tx_loc'],
                initial_voltage      = initial_voltage,
                conductor_resistance = conductor_resistance,
                power_factor         = power_factor,
                conductor_reactance  = conductor_reactance,
                lv_data              = lv_data,
                svc_lines            = svc_lines
            )
            
    if new_tx2 is not None:
        logging.info(f"Refined Group2 TX  {new_tx2['subA']['tx']}, "
                           f"{new_tx2['subB']['tx']}")
    else:
        logging.warning("Failed to refine Group2")
            
    result_lines = []
    loss_lines   = []

    group_data = []
    if new_tx1 is not None:
        group_data.append(("Group1-A", new_tx1['subA']['idx'], new_tx1['subA']['tx'],
                            g1['meter_locs'], g1['phase_loads']))
        group_data.append(("Group1-B", new_tx1['subB']['idx'], new_tx1['subB']['tx'],
                            g1['meter_locs'], g1['phase_loads']))
    if new_tx2 is not None:
        group_data.append(("Group2-A", new_tx2['subA']['idx'], new_tx2['subA']['tx'],
                            g2['meter_locs'], g2['phase_loads']))
        group_data.append(("Group2-B", new_tx2['subB']['idx'], new_tx2['subB']['tx'],
                            g2['meter_locs'], g2['phase_loads']))

    for tag, idx, tx, m_locs, p_loads in group_data:
        res, loss = summarise_subgroup(
            tag, idx, tx, m_locs, p_loads, lv_lines, mv_lines,
            initial_voltage, conductor_resistance, power_factor,
            conductor_reactance=conductor_reactance, lv_data=lv_data, svc_lines=svc_lines
        )
        if res:
            result_lines.append(res)
            loss_lines.append(loss)

    # ---- print the two blocks just once --------------------------------------
    if result_lines:
        logging.info("############ Result_Subgroup ############")
        for line in result_lines:
            logging.info(line)

    if loss_lines:
        logging.info("############ Loss Report_Subgroup ############")
        for line in loss_lines:
            logging.info(line)
            
    # Export subgroup
    extra_pts, extra_attrs = [], []
    nested_data = [("G1A", new_tx1), ("G1B", new_tx1), ("G2A", new_tx2), ("G2B", new_tx2)]
    
    for tag, nest in nested_data:
        if nest is None:
            continue
        if tag.endswith("A") and nest["subA"]["tx"] is not None:
            extra_pts.append(tuple(nest["subA"]["tx"]))
            extra_attrs.append({"Name": f"{tag} TX"})
        if tag.endswith("B") and nest["subB"]["tx"] is not None:
            extra_pts.append(tuple(nest["subB"]["tx"]))
            extra_attrs.append({"Name": f"{tag} TX"})

    if extra_pts:  # export only if we actually have new points
        folder_path = './testpy'
        ensure_folder_exists(folder_path)
        exportPointsToShapefile(
            extra_pts,(f"{folder_path}/nested_transformer_locations.shp"),extra_attrs
        )
        logging.info("Nested optimisation finished – shapefile exported.")

def runReoptimize():
    global latest_split_result, lvLines, mvLines, initialVoltage, conductorResistance, powerFactor, conductorReactance, lvData, svcLines
    
    if latest_split_result is None:
        messagebox.showwarning("No split result",
                               "กรุณา Run Process แล้วกด End process ก่อน")
        return
    run_nested_optimisation(
        latest_split_result,
        lvLines, mvLines,
        initialVoltage, conductorResistance, powerFactor,
        conductor_reactance=conductorReactance,
        lv_data=lvData,
        svc_lines=svcLines
    )
    print("Program finished successfully.")
    logging.info("Program finished successfully.")

def gui_candidate_input(G, transformerNode, meterNodes,
                        node_mapping, coord_mapping,
                        meterLocations, phase_loads,
                        lvLines, mvLines, filteredEserviceLines,
                        initialTransformerLocation,
                        powerFactor, initialVoltage,
                        conductorResistance,
                        peano, phases,
                        conductorReactance=None, lvData=None, svcLines=None):
        
    temp_root = None
        # ตรวจสอบว่ามี root หลักอยู่หรือไม่
    if 'root' in globals() and globals()['root'] is not None:
        try:
            # ถ้ามี root อยู่แล้ว ใช้เป็น parent
            temp_root = globals()['root']
        except:
            pass
    
    # ถ้าไม่มี root ให้สร้างแบบ withdraw (ซ่อน)
    if temp_root is None:
        temp_root = tk.Tk()
        temp_root.withdraw()  # ซ่อน root หลัก
        should_destroy_root = True
    else:
        should_destroy_root = False
    last_result = None  # start with current split

    while True:
        dialog = EdgeNavigatorDialog(
            temp_root,
            G,                 # กราฟเต็ม
            coord_mapping,     # พิกัดโหนด
            _EDGE_DF_CACHE,    # DataFrame ของ candidate‑edges
            [l['X'] for l in lvLines], [l['Y'] for l in lvLines],
            [l['X'] for l in mvLines], [l['Y'] for l in mvLines],
            meterLocations,
            start_idx=0        # (ถ้าใส่ไว้ใน __init__ ให้เป็น kwargs ก็ได้)
        )
        temp_root.wait_window(dialog)
        cand_idx = dialog.result
        if cand_idx is None:                      # End process
            break

        tmp = rerun_process(cand_idx,
                    G, transformerNode, meterNodes,
                    node_mapping, coord_mapping,
                    meterLocations, totalLoads,    
                    phase_loads,
                    lvLines, mvLines, filteredEserviceLines,
                    initialTransformerLocation,
                    powerFactor, initialVoltage,
                    conductorResistance,
                    peano, phases,
                    conductorReactance, lvData, svcLines,
                    snap_tolerance=SNAP_TOLERANCE)

        if tmp is not None:
            last_result = tmp

    # after loop ends
    if last_result is not None:
        latest_split_result = last_result
        btn = globals().get('reopt_btn', None)
        if btn is not None:
            try:
                btn.config(state=tk.NORMAL)
            except Exception:
                pass


    # ปิด temp_root เฉพาะเมื่อสร้างใหม่
    if should_destroy_root and temp_root is not None:
        try:
            temp_root.destroy()
        except:
            pass

# ---------------------------------
# 21) main, runProcess, createGUI, helper
def main():
    global meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases
    global lvLines, mvLines, initialVoltage, conductorResistance, powerFactor
    global conductorReactance, lvData, svcLines, latest_split_result, reopt_btn, SNAP_TOLERANCE
    
    logging.info("Program started with coordinate snapping.")
    
     
    # ====== JSON-based EXTRACT (drop-in) =================================
    out_json = r"D:\python\ODT\NetworkLV37-006819_with_MV.json"

    json_path = out_json  # หรือกำหนดเป็นสตริงพาธไฟล์ของคุณเอง

    # ----- ตั้งชื่อ log ให้ตรงกับ out_json (เฉพาะโหมดไม่ใช่ GUI) -----
    if not IS_GUI:
        try:
            default_log_dir = os.path.join(os.getcwd(), "logs")
            json_name = os.path.splitext(os.path.basename(out_json))[0]
            log_path = os.path.join(default_log_dir, f"{json_name}.run.log")

            setup_logging(
                log_path,
                to_console=True,          # หรือใช้ args.verbose ถ้ามีใน global
                level=logging.INFO,
                tee_stdout_to_file=True
            )
            logging.info("Switched logging in main() to: %s", log_path)
        except Exception as e:
            # ถ้าตั้ง log ไม่ได้ ให้แจ้ง แต่ไม่ให้โปรแกรมตาย
            logging.error(f"Failed to reconfigure logging for JSON '{out_json}': {e}")
    try:
        # 1) ดึง "มิเตอร์" จาก JSON
        meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases = extractMeterData_json(json_path, default_voltage=230.0, drop_zero_load=False, dedup=False)
        
        # 2) ดึง "เส้น" จาก JSON
        lvLinesX, lvLinesY, lvLines, _snap_lv, _snap_map_lv = extractLineData_json(json_path, snap_tolerance=SNAP_TOLERANCE)

        try:
            mvLinesX, mvLinesY, mvLines, _snap_mv, _snap_map_mv = extractMVLineData_json(
                json_path, snap_tolerance=None, tag_contains="MC"
            )
            logging.info(f"MV lines extracted: {len(mvLines)}")
        except Exception as e:
            logging.warning(f"Extract MV failed: {e}")
            mvLinesX, mvLinesY, mvLines = [], [], []
            _snap_mv, _snap_map_mv = None, {}


        try:
        # ดึง eservice ทั้งหมดจาก JSON ไม่กรอง tag_prefix
            eserviceLinesX, eserviceLinesY, filteredEserviceLines, _snap_svc, _snap_map_svc = extractServiceLines_json(
                json_path,
                snap_tolerance=SNAP_TOLERANCE,
                )
            if not filteredEserviceLines:
                logging.warning("No Eservice lines found in JSON.")
        except Exception as e:
            logging.error(f"Error processing Eservice lines from JSON: {e}")
            eserviceLinesX, eserviceLinesY, filteredEserviceLines = [], [], []
            _snap_svc, _snap_map_svc = None, {}
         
        
        
        snap_map = _snap_map_lv
        # สร้าง per-segment length จาก JSON (ใช้ SHAPE.LEN/สำรอง) โดยอิง snap_map เดิม
        try:
            length_map = build_line_length_map_from_json(
            json_input=out_json,
            coord_snap_map=snap_map,
            length_field="SHAPE.LEN",
            fallback_fields=("Shape_Leng", "SHAPE_Leng"),
            unit_factor=1.0
            )
            logging.info(f"[LEN] length_map ready: {len(length_map)//2} segments")
        except Exception as e:
            logging.warning(f"[LEN] build_line_length_map_from_json failed: {e}")
            length_map = None  # จะ fallback ไปใช้ระยะยูคลิด

        # 3) คำนวณ SNAP_TOLERANCE อัตโนมัติจากชุดข้อมูลจริง (มิเตอร์ + LV + MV)
        SNAP_TOLERANCE = auto_determine_snap_tolerance(
            meterLocations,
            lvLines,
            mvLines,
            reduction_ratio=0.98,
            use_analysis=True
        )
        logging.info(f"Automatically determined SNAP_TOLERANCE: {SNAP_TOLERANCE:.8f} m")

        logging.info(f"Applied coordinate snapping with tolerance: {SNAP_TOLERANCE} m")
        # ดึง (x,y) หม้อแปลงจาก JSON โดยตรง
        _fac = None
        try:
           # ถ้ามี argparse อยู่ในไฟล์ และรันแบบ --fac ให้ใช้เป็นตัวช่วย match FACILITYID
            _fac = args.facility_id  # ถ้าไม่มี args ก็ไม่เป็นไร
        except:
            pass

        # ตรวจสอบสายที่อาจมีปัญหา
        logging.info("Checking for problematic lines after snap...")
        
        # วิเคราะห์ LV lines
        lv_analysis = identify_failed_snap_lines(lvLines, SNAP_TOLERANCE)
        if lv_analysis['total_issues'] > 0:
            logging.warning(f"Found {lv_analysis['total_issues']} problematic LV lines")
            
            # Export สายที่มีปัญหาเพื่อตรวจสอบ
            folder_path = './testpy'
            ensure_folder_exists(folder_path)
            export_failed_lines_shapefile(
                lv_analysis, 
                lvLines, 
                f"{folder_path}/problematic_lv_lines.shp"
            )
            
            # พยายามแก้ไข
            fixed_lv_lines, fix_log = fix_failed_snap_lines(lv_analysis, lvLines, SNAP_TOLERANCE)
            logging.info(f"Attempted {len(fix_log)} fixes on LV lines")
        try:
            tx_xy = get_transformer_xy_from_json(out_json, facilityid=_fac)
        except Exception as e:
            logging.error(f"หา TR (x,y) จาก JSON ไม่ได้: {e}")
            return

        # 4) อ่านขนาดหม้อแปลงจาก JSON ตามที่คุณขอให้ย้ายมาใช้ RATEKVA
        try:
            # ถ้ามี FACILITYID ของงานนี้อยู่ในสโคป ให้ส่งเป็น facility_id; ไม่มีก็ปล่อย None
            transformerCapacity_kVA, transformerCapacity, powerFactor = \
                get_transformer_capacity_from_json(json_path, facilityid=None, default_pf=0.875)
            logging.info(
                f"[TR] RATEKVA={transformerCapacity_kVA} kVA  -> capacity≈{transformerCapacity:.2f} kW (pf={powerFactor})"
            )
        except Exception as e:
            logging.error(f"อ่าน RATEKVA จาก JSON ล้มเหลว: {e}")
            return

        # 5) เซ็ตค่าคงที่สาย (ถ้าคุณมีค่ามาตรฐานอื่น ให้ปรับตรงนี้)
        conductorResistance = 0.77009703
        conductorReactance  = 0.3497764
        initialVoltage      = 230

        # 6) (สำคัญ) ให้ตัวแปรที่ส่วนล่างใช้ชื่อเดิมได้
        svcLines = filteredEserviceLines

    except Exception as e:
        logging.error(f"An error occurred during JSON extract: {e}")
        return
    # ====== END JSON-based EXTRACT ======================================
   
    ############ Build initial network (with snapping) ############
    logging.info("Building initial network with snapping …")   
    
        # 1) สร้าง coord_snap_map จาก LV+MV ด้วย SNAP_TOLERANCE
    all_line_coords = []
    for L in (lvLines, mvLines):
        for line in L:
            all_line_coords.extend([(x, y) for x, y in zip(line['X'], line['Y'])
                                    if not (np.isnan(x) or np.isnan(y))])
    coord_snap_map = snap_coordinates_to_tolerance(list(set(all_line_coords)), SNAP_TOLERANCE)

    # 2) สร้าง per-segment length_map จาก JSON (SHAPE.LEN หรือสำรอง)
    length_map = build_line_length_map_from_json(
        json_input=out_json,               # path/dict ของ JSON
        coord_snap_map=coord_snap_map,
        length_field="SHAPE.LEN",
        fallback_fields=("Shape_Leng", "SHAPE_Leng"),
        unit_factor=1.0
    )
    
    G_init, tNode_init, mNodes_init, nm_init, cm_init = buildLVNetworkWithLoads(
    lvLines=lvLines,
    mvLines=mvLines,
    meterLocations=meterLocations,
    transformerLocation=tx_xy,                # มาจาก get_transformer_xy_from_json(...)
    phase_loads=phase_loads,
    conductorResistance=conductorResistance,
    conductorReactance=conductorReactance,
    svcLines=svcLines,
    use_shape_length=True,                    # ใช้ความยาวจาก JSON
    lvData=None,                              # ไม่ต้องส่ง shapefile แล้ว
    length_field="SHAPE.LEN",
    snap_tolerance=SNAP_TOLERANCE,
    line_length_map=length_map,               # << สำคัญ
    coord_snap_map=coord_snap_map             # << สำคัญ
    )

    
    if not nx.is_connected(G_init):
        logging.warning("Initial network not fully connected.")
        components = list(nx.connected_components(G_init))
        logging.warning(f"Network has {len(components)} connected components")
    else:
        logging.info("Initial network is connected.")
    
    # For this run we use LV-based optimization
    optimizedTransformerLocation_LV = optimizeTransformerLocationOnLVCond3(
        meterLocations, phase_loads, initialTransformerLocation,
        lvLines, initialVoltage, conductorResistance, powerFactor,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    
    logging.info(f"Optimized (LV) => {optimizedTransformerLocation_LV}")
    optimizedTransformerLocation = optimizedTransformerLocation_LV
    
    # ใช้ฟังก์ชันที่แก้ไขแล้ว พร้อม coordinate snapping
    G, transformerNode, meterNodes, node_mapping, coord_mapping = buildLVNetworkWithLoads(
    lvLines=lvLines,
    mvLines=mvLines,
    meterLocations=meterLocations,
    transformerLocation=optimizedTransformerLocation,   # หรือ tx_xy / tx_group
    phase_loads=phase_loads,
    conductorResistance=conductorResistance,
    conductorReactance=conductorReactance,
    svcLines=svcLines,
    use_shape_length=True,
    lvData=None,
    length_field="SHAPE.LEN",
    snap_tolerance=SNAP_TOLERANCE,
    line_length_map=length_map,
    coord_snap_map=coord_snap_map
    )

    # ตรวจสอบความสมบูรณ์ของ network
    validation = validate_network_after_snap(G, coord_mapping, meterNodes, transformerNode)

    if not validation['summary']['network_complete']:
        logging.error("Network is incomplete after snap!")
        logging.error(f"Unreachable meters: {len(validation['unreachable_meters'])}")
        
        # อาจต้องปรับ tolerance หรือแก้ไขข้อมูล
        if len(validation['unreachable_meters']) > 0:
            # ลองใช้ tolerance ที่สูงขึ้น
            new_tolerance = SNAP_TOLERANCE * 2
            logging.info(f"Retrying with higher tolerance: {new_tolerance}")
        if not nx.is_connected(G):
            logging.warning("Post-optimization network not fully connected.")
            components = list(nx.connected_components(G))
            logging.warning(f"Network has {len(components)} connected components")
        else:
            logging.info("Post-optimization network is connected.")
    
    splitting_edge, sp_coord, sp_edge_diff, candidate_edges = findSplittingPoint(
        G, transformerNode, meterNodes, coord_mapping,
        powerFactor, initialVoltage, candidate_index=0
    )
    
    if splitting_edge is None:
        logging.warning("No splitting edge found; single group only. End.")
        return
        
    group1_nodes, group2_nodes = partitionNetworkAtPoint(G, transformerNode, meterNodes, splitting_edge)
    
    voltages, branch_curr, group1_nodes, group2_nodes = performForwardBackwardSweepAndDivideLoads(
        G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, splitting_edge
    )

    nodeToIndex = {mn: i for i, mn in enumerate(meterNodes)}
    group1_meter_nodes = [n for n in group1_nodes if n in nodeToIndex]
    group2_meter_nodes = [n for n in group2_nodes if n in nodeToIndex]
    g1_idx = [nodeToIndex[n] for n in group1_meter_nodes]
    g2_idx = [nodeToIndex[n] for n in group2_meter_nodes]

    # 1) สร้างข้อมูลเฉพาะกลุ่ม 1
    group1_meterLocs = meterLocations[g1_idx]
    loads_g1 = totalLoads[g1_idx]
    # โหลดตามเฟสของ มิเตอร์กลุ่ม 1
    group1_phase_loads = {
        'A': phase_loads['A'][g1_idx],
        'B': phase_loads['B'][g1_idx],
        'C': phase_loads['C'][g1_idx],
    }
        # ข้อมูล peano และ phases ตามกลุ่ม
    peano_g1  = peano[g1_idx]
    phases_g1 = phases[g1_idx]
    
    phase_indexer = build_phase_indexer_from_json(
    json_input=json_path,
    candidate_fields=("PHASEDESIGNATION","PHASEDESIG","PHASE","PHASETYPE"),
    tag_prefix_for_lv=("22LC", "2244LC"),  # ครอบทั้ง TAG "22LCEA..." และ "2244LC..." :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
    subtype_allow=(1,),
    opvolt_max=1000.0
    )

    # เรียก balance Group 1
    new_phases_g1, new_phase_loads_g1 = optimize_phase_balance(
    group1_meterLocs, loads_g1, group1_phase_loads, peano[g1_idx],
    lvData=None,                                  # ไม่ต้องใช้ shapefile
    original_phases=phases[g1_idx].tolist(),
    phase_indexer=phase_indexer,                  # << ใช้อันนี้
    target_unbalance_pct=10.0
    )
    logging.info(
        "Group 1 load balance before -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
        group1_phase_loads['A'].sum(),
        group1_phase_loads['B'].sum(),
        group1_phase_loads['C'].sum()
    )
    # Calculate %Unbalance Before Group1
    g1_a = group1_phase_loads['A'].sum()
    g1_b = group1_phase_loads['B'].sum()
    g1_c = group1_phase_loads['C'].sum()
    g1_avg = (g1_a + g1_b + g1_c) / 3.0
    g1_unb_before = max(abs(g1_a - g1_avg), abs(g1_b - g1_avg), abs(g1_c - g1_avg)) / g1_avg * 100
    logging.info("Group 1 percent unbalance before: %.2f%%", g1_unb_before)
    
    logging.info(
        "Group 1 load balance after  -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
        new_phase_loads_g1['A'].sum(),
        new_phase_loads_g1['B'].sum(),
        new_phase_loads_g1['C'].sum()
    )
    # Calculate %Unbalance After Group1
    new_g1_a = new_phase_loads_g1['A'].sum()
    new_g1_b = new_phase_loads_g1['B'].sum()
    new_g1_c = new_phase_loads_g1['C'].sum()
    new_g1_avg = (new_g1_a + new_g1_b + new_g1_c) / 3.0
    g1_unb_after = max(abs(new_g1_a - new_g1_avg), abs(new_g1_b - new_g1_avg), abs(new_g1_c - new_g1_avg)) / new_g1_avg * 100
    logging.info("Group 1 percent unbalance after: %.2f%%", g1_unb_after)
    # ---- %Unbalance summary (หลังคำนวณก่อน-หลังเสร็จ) ----
    unb_g1_summary = summarize_unbalance_change(group1_phase_loads, new_phase_loads_g1)
    logging.info("Group 1 %%Unbalance summary: %s", unb_g1_summary)
    
    
    
    # 2) กลุ่ม 2
    group2_meterLocs = meterLocations[g2_idx]
    loads_g2 = totalLoads[g2_idx]
    group2_phase_loads = {
        'A': phase_loads['A'][g2_idx],
        'B': phase_loads['B'][g2_idx],
        'C': phase_loads['C'][g2_idx],
    }
    peano_g2  = peano[g2_idx]
    phases_g2 = phases[g2_idx]

    new_phases_g2, new_phase_loads_g2 = optimize_phase_balance(
    group2_meterLocs, loads_g2, group2_phase_loads, peano[g2_idx],
    lvData=None,                                  # ไม่ต้องใช้ shapefile
    original_phases=phases[g2_idx].tolist(),
    phase_indexer=phase_indexer,                  # << ใช้อันนี้
    target_unbalance_pct=10.0
    )
    logging.info(
    "Group 2 load balance before -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
    group2_phase_loads['A'].sum(),
    group2_phase_loads['B'].sum(),
    group2_phase_loads['C'].sum()
    )
    # Calculate %Unbalance Before Group2
    g2_a = group2_phase_loads['A'].sum()
    g2_b = group2_phase_loads['B'].sum()
    g2_c = group2_phase_loads['C'].sum()
    g2_avg = (g2_a + g2_b + g2_c) / 3.0
    g2_unb_before = max(abs(g2_a - g2_avg), abs(g2_b - g2_avg), abs(g2_c - g2_avg)) / g2_avg * 100
    logging.info("Group 2 percent unbalance before: %.2f%%", g2_unb_before)
    
    logging.info(
        "Group 2 load balance after  -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
        new_phase_loads_g2['A'].sum(),
        new_phase_loads_g2['B'].sum(),
        new_phase_loads_g2['C'].sum()
    )
    # Calculate %Unbalance After Group2
    new_g2_a = new_phase_loads_g2['A'].sum()
    new_g2_b = new_phase_loads_g2['B'].sum()
    new_g2_c = new_phase_loads_g2['C'].sum()
    new_g2_avg = (new_g2_a + new_g2_b + new_g2_c) / 3.0
    g2_unb_after = max(abs(new_g2_a - new_g2_avg), abs(new_g2_b - new_g2_avg), abs(new_g2_c - new_g2_avg)) / new_g2_avg * 100
    logging.info("Group 2 percent unbalance after: %.2f%%", g2_unb_after)
    unb_g2_summary = summarize_unbalance_change(group2_phase_loads, new_phase_loads_g2)
    logging.info("Group 2 %%Unbalance summary: %s", unb_g2_summary)
    
    
    
    dist_arr = []
    for n in meterNodes:
        try:
            dval = nx.shortest_path_length(G, transformerNode, n, weight='weight')
        except nx.NetworkXNoPath:
            dval = float('inf')
        dist_arr.append(dval)
        
    voltA = {n: voltages[n]['A'] for n in voltages}
    voltB = {n: voltages[n]['B'] for n in voltages}
    voltC = {n: voltages[n]['C'] for n in voltages}
    
    result_df = pd.DataFrame({
        'Peano Meter': peano,
        'Final Voltage A (V)': [voltA.get(n, np.nan) for n in meterNodes],
        'Final Voltage B (V)': [voltB.get(n, np.nan) for n in meterNodes],
        'Final Voltage C (V)': [voltC.get(n, np.nan) for n in meterNodes],
        'Distance to Transformer (m)': dist_arr,
        'Load A (kW)': phase_loads['A'],
        'Load B (kW)': phase_loads['B'],
        'Load C (kW)': phase_loads['C'],
        'Group': ['Group 1' if n in group1_nodes else 'Group 2' for n in meterNodes],
        'Phases': phases
    })
    result_df["Meter X"] = meterLocations[:, 0]
    result_df["Meter Y"] = meterLocations[:, 1]
    # 3) เอาค่าที่ได้ไปอัปเดตใน result_df
    for local_i, global_i in enumerate(g1_idx):
        result_df.at[global_i, 'New Phase']  = new_phases_g1[local_i]
        result_df.at[global_i, 'New Load A'] = new_phase_loads_g1['A'][local_i]
        result_df.at[global_i, 'New Load B'] = new_phase_loads_g1['B'][local_i]
        result_df.at[global_i, 'New Load C'] = new_phase_loads_g1['C'][local_i]

    for local_i, global_i in enumerate(g2_idx):
        result_df.at[global_i, 'New Phase']  = new_phases_g2[local_i]
        result_df.at[global_i, 'New Load A'] = new_phase_loads_g2['A'][local_i]
        result_df.at[global_i, 'New Load B'] = new_phase_loads_g2['B'][local_i]
        result_df.at[global_i, 'New Load C'] = new_phase_loads_g2['C'][local_i]

    for n in G.nodes():
        if n in group1_nodes:
            G.nodes[n]['group'] = 1
        elif n in group2_nodes:
            G.nodes[n]['group'] = 2
        else:
            G.nodes[n]['group'] = 0
            
    g1_load = sum(G.nodes[n]['load_A'] + G.nodes[n]['load_B'] + G.nodes[n]['load_C'] for n in group1_nodes)
    g2_load = sum(G.nodes[n]['load_A'] + G.nodes[n]['load_B'] + G.nodes[n]['load_C'] for n in group2_nodes)
    logging.info(f"Group 1 total load: {g1_load:.2f} kW | Group 2 total load: {g2_load:.2f} kW")

    group1_meterLocs = meterLocations[g1_idx]
    group1_phase_loads = {
        'A': phase_loads['A'][g1_idx],
        'B': phase_loads['B'][g1_idx],
        'C': phase_loads['C'][g1_idx]
    }
    group2_meterLocs = meterLocations[g2_idx]
    group2_phase_loads = {
        'A': phase_loads['A'][g2_idx],
        'B': phase_loads['B'][g2_idx],
        'C': phase_loads['C'][g2_idx]
    }
    
    logging.info("Calculate LoadCenter each Group...")
    logging.info("Optimizing Transformer Location each Group...")
    
    
    lc_g1 = calculateNetworkLoadCenter(
        group1_meterLocs, 
        group1_phase_loads, 
        lvLines, 
        mvLines, 
        conductorResistance,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    lc_g2 = calculateNetworkLoadCenter(
        group2_meterLocs, 
        group2_phase_loads, 
        lvLines, 
        mvLines, 
        conductorResistance,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    logging.info(f"Group 1 load center = {lc_g1}")
    logging.info(f"Group 2 load center = {lc_g2}")

    logging.info("Optimizing Transformer Location each Group...")
    optimizedTransformerLocationGroup1 = optimizeGroup(
        group1_meterLocs,
        group1_phase_loads,
        lc_g1,
        lvLines, 
        mvLines,
        initialVoltage, 
        conductorResistance,
        powerFactor, 
        epsilon_junction=2.0,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    optimizedTransformerLocationGroup2 = optimizeGroup(
        group2_meterLocs,
        group2_phase_loads,
        lc_g2,
        lvLines, 
        mvLines,
        initialVoltage, 
        conductorResistance,
        powerFactor, 
        epsilon_junction=2.0,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    logging.info(f"Group 1 optimized TR = {optimizedTransformerLocationGroup1}")
    logging.info(f"Group 2 optimized TR = {optimizedTransformerLocationGroup2}")

      
    if optimizedTransformerLocationGroup1 is not None:
        G_g1, tNode_g1, mNodes_g1, nm_g1, cm_g1 = buildLVNetworkWithLoads(
            lvLines, mvLines,
            group1_meterLocs,
            optimizedTransformerLocationGroup1,
            group1_phase_loads,
            conductorResistance,
            conductorReactance=conductorReactance,
            use_shape_length=True,
            lvData=lvData,
            svcLines=svcLines
        )
        dist_g1 = []
        for i, mNode in enumerate(mNodes_g1):
            try:
                dist_g1.append(nx.shortest_path_length(G_g1, tNode_g1, mNode, weight='weight'))
            except nx.NetworkXNoPath:
                dist_g1.append(float('inf'))
                
        # เปลี่ยนไปใช้ calculateUnbalancedPowerFlow แทน calculatePowerLoss
        node_voltages_g1, branch_currents_g1, total_power_loss_g1 = calculateUnbalancedPowerFlow(
        G_g1, tNode_g1, mNodes_g1, powerFactor, initialVoltage
        )
        
        for i, node in enumerate(mNodes_g1):
            global_idx = g1_idx[i]
            result_df.at[global_idx, 'Distance to Transformer (m)'] = dist_g1[i]
            result_df.at[global_idx, 'Final Voltage A (V)'] = abs(node_voltages_g1[node]['A'])
            result_df.at[global_idx, 'Final Voltage B (V)'] = abs(node_voltages_g1[node]['B'])
            result_df.at[global_idx, 'Final Voltage C (V)'] = abs(node_voltages_g1[node]['C'])
            
    if optimizedTransformerLocationGroup2 is not None:
        G_g2, tNode_g2, mNodes_g2, nm_g2, cm_g2 = buildLVNetworkWithLoads(
            lvLines, mvLines,
            group2_meterLocs,
            optimizedTransformerLocationGroup2,
            group2_phase_loads,
            conductorResistance,
            conductorReactance=conductorReactance,
            use_shape_length=True,
            lvData=lvData,
            svcLines=svcLines
        )
        dist_g2 = []
        for i, mNode in enumerate(mNodes_g2):
            try:
                dist_g2.append(nx.shortest_path_length(G_g2, tNode_g2, mNode, weight='weight'))
            except nx.NetworkXNoPath:
                dist_g2.append(float('inf'))
                
        # เปลี่ยนไปใช้ calculateUnbalancedPowerFlow แทน calculatePowerLoss
        node_voltages_g2, branch_currents_g2, total_power_loss_g2 = calculateUnbalancedPowerFlow(
        G_g2, tNode_g2, mNodes_g2, powerFactor, initialVoltage
        )
        
        for i, node in enumerate(mNodes_g2):
            global_idx = g2_idx[i]
            result_df.at[global_idx, 'Distance to Transformer (m)'] = dist_g2[i]
            result_df.at[global_idx, 'Final Voltage A (V)'] = abs(node_voltages_g2[node]['A'])
            result_df.at[global_idx, 'Final Voltage B (V)'] = abs(node_voltages_g2[node]['B'])
            result_df.at[global_idx, 'Final Voltage C (V)'] = abs(node_voltages_g2[node]['C'])

    # Calculate Total Loss
    line_loss_g1_kW = total_power_loss_g1 / 1000.0
    tx_loss_g1_kW = get_transformer_losses(g1_load)
    total_system_loss_g1 = line_loss_g1_kW + tx_loss_g1_kW
    # ---- Loss summary ต่อกลุ่ม (ใช้ TX ที่ optimize แล้ว) ----
    loss_g1_summary = summarize_loss_change(
        lvLines, mvLines, group1_meterLocs, optimizedTransformerLocationGroup1,
        group1_phase_loads, new_phase_loads_g1,
        conductorResistance=conductorResistance,
        powerFactor=powerFactor,
        initialVoltage=initialVoltage,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )
    
    line_loss_g2_kW = total_power_loss_g2 / 1000.0
    tx_loss_g2_kW = get_transformer_losses(g2_load)
    total_system_loss_g2 = line_loss_g2_kW + tx_loss_g2_kW
    loss_g2_summary = summarize_loss_change(
        lvLines, mvLines, group2_meterLocs, optimizedTransformerLocationGroup2,
        group2_phase_loads, new_phase_loads_g2,
        conductorResistance=conductorResistance,
        powerFactor=powerFactor,
        initialVoltage=initialVoltage,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )
    
    # Forecast Future Load
    future_g1_load = growthRate(g1_load, annual_growth=0.06, years=4)
    future_g2_load = growthRate(g2_load, annual_growth=0.06, years=4)
    
    # Select Transformer size for each group (using your document)
    rating_g1 = Lossdocument(future_g1_load)
    rating_g2 = Lossdocument(future_g2_load)

    
    
    # Max Distance from Group 1,2
    max_dist1 = max(dist_g1)
    max_dist2 = max(dist_g2)

    logging.info('############ Result ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW, Future growthrate(4yr@6%)={future_g1_load:.2f} kW, Chosen TX1={rating_g1} kVA, Max Distance Group1={max_dist1:.1f}m.")
    logging.info(f"Group 2 => Load={g2_load:.2f} kW, Future growthrate(4yr@6%)={future_g2_load:.2f} kW, Chosen TX2={rating_g2} kVA, Max Distance Group2={max_dist2:.1f}m.")
    logging.info('########## Loss Report #########')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW | LineLoss={line_loss_g1_kW:.2f} kW | TxLoss={tx_loss_g1_kW:.2f} kW => TOTAL={total_system_loss_g1:.2f} kW")
    logging.info("Group 1 line loss: BEFORE %.2f kW -> AFTER %.2f kW",
             loss_g1_summary["loss_before_kW"], loss_g1_summary["loss_after_kW"])
    logging.info(f"Group 2 => Load={g2_load:.2f} kW | LineLoss={line_loss_g2_kW:.2f} kW | TxLoss={tx_loss_g2_kW:.2f} kW => TOTAL={total_system_loss_g2:.2f} kW")
    logging.info("Group 2 line loss: BEFORE %.2f kW -> AFTER %.2f kW",
             loss_g2_summary["loss_before_kW"], loss_g2_summary["loss_after_kW"])
    logging.info("####### %UnBalance Report ######")
    logging.info("Group 1 %%Unbalance: BEFORE %.2f%% -> AFTER %.2f%%", g1_unb_before, g1_unb_after)
    logging.info("Group 2 %%Unbalance: BEFORE %.2f%% -> AFTER %.2f%%", g2_unb_before, g2_unb_after)

    point_coords = []
    attributes_list = []
    if sp_coord is not None:
        # Convert sp_coord to tuple for consistency
        point_coords.append(tuple(sp_coord))
        attributes_list.append({'Name': 'Splitting Point'})
    if optimizedTransformerLocationGroup1 is not None:
        point_coords.append(tuple(optimizedTransformerLocationGroup1))
        attributes_list.append({'Name': 'Group 1 Transformer'})
    if optimizedTransformerLocationGroup2 is not None:
        point_coords.append(tuple(optimizedTransformerLocationGroup2))
        attributes_list.append({'Name': 'Group 2 Transformer'})
    if point_coords:
        folder_path = './testpy'
        ensure_folder_exists(folder_path)
        exportPointsToShapefile(point_coords, f"{folder_path}/optimized_transformer_locations.shp", attributes_list)
        g1_plot_indices = np.array(g1_idx, dtype=int)
        g2_plot_indices = np.array(g2_idx, dtype=int)
    
    # Export to CSV
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    csv_path = f"{folder_path}/optimized_transformer_locations.csv"
    result_df.to_csv(csv_path, index=False)
    logging.info(f"Result CSV saved: {csv_path}")
    
    # Export to shapefile
    exportResultDFtoShapefile(result_df, f"{folder_path}/result_meters.shp")
    
        # --- เตรียมค่าแบบปลอดภัยก่อนเรียก plotResults ---
    def _xs(lines): return [l['X'] for l in lines] if lines else []
    def _ys(lines): return [l['Y'] for l in lines] if lines else []

    lvX = _xs(lvLines)
    lvY = _ys(lvLines)
    mvX = _xs(mvLines)
    mvY = _ys(mvLines)
    svcX = _xs(filteredEserviceLines)
    svcY = _ys(filteredEserviceLines)

    # initial TX: ใช้ tx_xy ที่อ่านจาก JSON; ถ้าไม่มี ให้เป็น None (plotResults เวอร์ชันที่แก้แล้วจะข้ามให้)
    initial_tx = tx_xy if (tx_xy and len(tx_xy) == 2) else None

    # indices ของกลุ่ม แปลงเป็น array int ถ้ายังไม่ใช่
    g1_plot_indices = np.asarray(g1_plot_indices, dtype=int) if 'g1_plot_indices' in locals() else np.array([], dtype=int)
    g2_plot_indices = np.asarray(g2_plot_indices, dtype=int) if 'g2_plot_indices' in locals() else np.array([], dtype=int)

    # --- เรียก plotResults ---
    plotResults(
    [l['X'] for l in lvLines], [l['Y'] for l in lvLines],
    [l['X'] for l in mvLines], [l['Y'] for l in mvLines],   # << MV ต้องส่งแบบนี้
    [l['X'] for l in filteredEserviceLines],
    [l['Y'] for l in filteredEserviceLines],
    meterLocations,
    initialTransformerLocation=tx_xy,                       # ถ้ามี
    optimizedTransformerLocation=optimizedTransformerLocation_LV,
    group1_indices=g1_plot_indices,
    group2_indices=g2_plot_indices,
    splitting_point_coords=sp_coord,
    coord_mapping=coord_mapping,
    optimizedTransformerLocationGroup1=optimizedTransformerLocationGroup1,
    optimizedTransformerLocationGroup2=optimizedTransformerLocationGroup2,
    transformer_losses=None,
    phases=phases,
    result_df=result_df,
    G=G
    )


    G = addNodeLabels(G, None, sp_edge_diff)
    plotGraphWithLabels(G, coord_mapping, best_edge_diff=sp_edge_diff, best_edge=splitting_edge)
    
    logging.info("Initial processing complete. Proceeding with group-level optimization and output.")
    
    # 5‑A)  build initial split dict so the button works FIRST time
    global latest_split_result, reopt_btn
    latest_split_result = {
        'best_edge': splitting_edge,
        'group1'   : {
            'idx'        : np.array(g1_idx, dtype=int),
            'meter_locs' : group1_meterLocs,
            'phase_loads': group1_phase_loads,
            'tx_loc'     : optimizedTransformerLocationGroup1
        },
        'group2'   : {
            'idx'        : np.array(g2_idx, dtype=int),
            'meter_locs' : group2_meterLocs,
            'phase_loads': group2_phase_loads,
            'tx_loc'     : optimizedTransformerLocationGroup2
        }
    }
    # แทนบล็อกเดิม:
    # if reopt_btn is not None:
    #     reopt_btn.config(state=tk.NORMAL)

    btn = globals().get('reopt_btn', None)
    if btn is not None:
        try:
            btn.config(state=tk.NORMAL)
        except Exception:
            # เผื่อกรณีไม่มี Tk context ในโหมด headless
            pass


    # 5‑B)  open the candidate‑edge dialog
    gui_candidate_input(G, transformerNode, meterNodes,
                        node_mapping, coord_mapping,
                        meterLocations, phase_loads,
                        lvLines, mvLines, filteredEserviceLines,
                        initialTransformerLocation,
                        powerFactor, initialVoltage,
                        conductorResistance,
                        peano, phases,
                        conductorReactance, lvData, svcLines)

    logging.info("Program finished successfully.")
    
   
   

def runProcess():
    """Continue the process after shapefiles have been loaded."""
    global meterData, lvData, mvData, transformerData, eserviceData
    
    if meterData is None or lvData is None or mvData is None or transformerData is None or eserviceData is None:
        logging.error("Shapefiles have not been loaded yet!")
        return
    
    # Call the main function to continue the process
    main()



# Creating the UI and Buttons
# def createGUI():
#     """Create the GUI with 'Run Process' and 'Import ShapeFile' buttons."""
#     global root, transformerFileName, reopt_btn, lvData, conductorReactance
    
#     root = tk.Tk()
#     root.title("Transformer Optimization GUI")

#     # Create a frame at the top for the buttons
#     top_frame = tk.Frame(root)
#     top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

#     # Add the "Import ShapeFile" Button
#     import_btn = tk.Button(top_frame, text="Import ShapeFile", command=lambda: loadShapefiles(root))
#     import_btn.pack(side=tk.LEFT, padx=5)

#     # Add the "Run Process" Button
#     run_btn = tk.Button(top_frame, text="Run Process", command=runProcess)
#     run_btn.pack(side=tk.LEFT, padx=5)
    
#     # Add the "Re-optimize" Button
#     reopt_btn = tk.Button(top_frame, text="Re-optimize subgroup",
#                           state=tk.DISABLED,      # disabled until we have data
#                           command=runReoptimize)
#     reopt_btn.pack(side=tk.LEFT, padx=5)

#     # Create a frame to hold the log text widget and scrollbar
#     text_frame = tk.Frame(root)
#     text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#     # Create the scrollbar
#     scrollbar = tk.Scrollbar(text_frame)
#     scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

#     # Create the text widget for logging
#     log_text = tk.Text(text_frame, wrap="word", height=20, yscrollcommand=scrollbar.set)
#     log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#     # Configure the scrollbar to scroll the text widget
#     scrollbar.config(command=log_text.yview)
    
#     # ---------- Progress bar -------------
#     progress_frame = tk.Frame(root)
#     progress_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

#     progress_bar = ttk.Progressbar(progress_frame, orient='horizontal',
#                                    mode='determinate', length=400)
#     progress_bar.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

#     progress_label = tk.Label(progress_frame, text="", anchor="w")
#     progress_label.pack(side=tk.LEFT)

#     # ทำให้ทุกที่เรียกใช้ได้
#     global tk_progress
#     tk_progress = TkProgress(progress_bar, progress_label)


       
#     if 'transformerFileName' in globals() and transformerFileName:
#         base_name = os.path.splitext(os.path.basename(transformerFileName))[0]
#         folder_path = './testpy'
#         ensure_folder_exists(folder_path)
#         log_filename = os.path.join(folder_path, f"Optimization_{base_name}_log.txt")
#     else:
#         folder_path = './testpy'
#         ensure_folder_exists(folder_path)
#         log_filename = f"{folder_path}/Optimization_Transformer_log.txt"
    
#     # Configure the logger to log messages to the text widget
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
#     text_handler = TextHandler(log_text)
#     text_handler.setFormatter(fmt)
#     text_handler.setLevel(logging.INFO)
#     text_handler.addFilter(SummaryFilter())  # Filtering logs
#     logger.addHandler(text_handler)

#     # Configure a FileHandler to also log to a file
#     file_handler = logging.FileHandler(log_filename)
#     file_handler.setFormatter(fmt)
#     file_handler.setLevel(logging.INFO)
#     logger.addHandler(file_handler)
#     root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))

#     root.mainloop()

#############################################################################################
# if __name__ == "__main__":
#     import glob  # ใช้กับ auto-detect JSON

#     # ---------- 1) parse args ----------
#     parser = argparse.ArgumentParser(
#         description="Transformer Optimization runner (headless by default; GUI only with --gui)"
#     )
#     parser.add_argument("--gui", action="store_true",
#                         help="เปิด GUI (จะไม่เปิด GUI เว้นแต่ใส่ --gui)")
#     parser.add_argument("--fac", dest="facility_id",
#                         help="รัน non-GUI จาก FACILITYID (จะสร้างไฟล์ JSON ให้ก่อน)")
#     parser.add_argument("--json", dest="json_path",
#                         help="รัน non-GUI จากไฟล์ JSON ที่มีอยู่แล้ว")
#     parser.add_argument("--verbose", "-v", action="store_true",
#                         help="เพิ่มระดับ log เป็น INFO และพิมพ์ลงคอนโซล")
#     args = parser.parse_args()

#     # หลัง parser.parse_args()
#     IS_GUI = bool(getattr(args, "gui", False))

#     # ---------- 2) resolve inputs (fac / jpth / out_json) ----------
#     fac  = args.facility_id
#     jpth = args.json_path
#     out_json = None  # จะถูกตั้งค่าทีหลังถ้ามี pipeline

#     # ENV fallback (ใช้เฉพาะกรณีไม่ส่ง args มาเลย)
#     if not fac and not jpth:
#         fac  = os.environ.get("FACILITY_ID") or None
#         jpth = os.environ.get("JSON_PATH") or None

#     # auto-detect ไฟล์ JSON ล่าสุดในโฟลเดอร์ (เฉพาะเมื่อไม่ใช้ GUI และยังไม่มีอินพุต)
#     if not args.gui and not fac and not jpth:
#         candidates = sorted(
#             glob.glob("TRwitmeter*_with_mv*.json") + glob.glob("TRwitmeter*.json"),
#             key=lambda p: os.path.getmtime(p),
#             reverse=True
#         )
#         if candidates:
#             jpth = candidates[0]

#     # ---------- 3) setup logging รอบแรก (ชื่อกลาง ๆ) ----------
#     default_log_dir = os.path.join(os.getcwd(), "logs")
#     log_path = os.path.join(default_log_dir, "run.run.log")

#     logger = setup_logging(
#         log_path,
#         to_console=getattr(args, "verbose", False),  # ใช้ -v คุมว่าจะโชว์ console ไหม
#         level=logging.INFO,
#         tee_stdout_to_file=True      # print() ลง log นี้ด้วย
#     )

#     logging.info("args => gui=%s, fac=%s, json=%s", bool(args.gui), fac, jpth)
#     print("Create Log Already")

#     # ---------- 4) headless / GUI flow ----------
#     if fac or jpth:
#         try:
#             # 4.1 ถ้ามี FACILITYID -> call pipeline ก่อน ให้มันสร้าง JSON ใหม่
#             if fac:
#                 if "run_pipeline_for_facilityid" not in globals():
#                     raise SystemExit("run_pipeline_for_facilityid ยังไม่ได้ import/ประกาศ")

#                 res = run_pipeline_for_facilityid(fac)
#                 if not res or not res.get("out_json") or not os.path.exists(res["out_json"]):
#                     raise SystemExit(f"Pipeline error: ไม่ได้ไฟล์ JSON จาก FACILITYID={fac}")
#                 out_json = res["out_json"]
#                 logging.info("[pipeline] out_json => %s", out_json)

#             # 4.2 ถ้า user ส่ง --json มาโดยตรง
#             if jpth:
#                 if not os.path.exists(jpth):
#                     raise SystemExit(f"ไม่พบไฟล์ JSON: {jpth}")
#                 out_json = jpth
#                 logging.info("[direct-json] out_json => %s", out_json)

#             # ตอนนี้ out_json ต้องมีแล้ว
#             if not out_json:
#                 raise SystemExit("ไม่สามารถหาไฟล์ JSON ที่จะใช้รัน main() ได้")

#             # 4.3 สลับ logging ไปผูกกับชื่อ out_json จริง
#             json_name = os.path.splitext(os.path.basename(out_json))[0]
#             new_log = os.path.join(default_log_dir, f"{json_name}.run.log")

#             logger = setup_logging(
#                 new_log,
#                 to_console=getattr(args, "verbose", False),
#                 level=logging.INFO,
#                 tee_stdout_to_file=True      # สำคัญ! ให้ print() ไปลงไฟล์ใหม่นี้
#             )
#             logging.info("Switch logging to: %s", new_log)

#             # กันไลบรารีภายในสับสน args
#             sys.argv = [sys.argv[0]]

#             logging.info("[main] starting …")
#             try:
#                 # main() ของคุณต้องใช้ out_json จาก global (หรือคุณจะส่งเป็นพารามิเตอร์เองก็ได้)
#                 main()
#             except Exception:
#                 logging.exception("[main] crashed with exception")
#                 raise SystemExit(1)

#             logging.info("[main] finished successfully.")
#             sys.exit(0)

#         except SystemExit as se:
#             logging.error("EXIT: %s", se)
#             raise
#         except Exception:
#             logging.exception("Fatal error in headless run")
#             sys.exit(1)

#     # ---------- 5) GUI mode ----------
#     if args.gui:
#         createGUI()
#         sys.exit(0)

#     # ---------- 6) ไม่มี input / ไม่ gui ----------
#     logging.error(
#         "ไม่มี FACILITYID / JSON ให้รัน และไม่ได้ระบุ --gui\n"
#         "วิธีรันตัวอย่าง:\n"
#         "  python %s --fac 67-005991\n"
#         "  python %s --json D:\\path\\TRwitmeter67-005991_with_mv.json\n"
#         "หรือกำหนด ENV: FACILITY_ID / JSON_PATH แล้วสั่ง: python %s -v",
#         os.path.basename(sys.argv[0]),
#         os.path.basename(sys.argv[0]),
#         os.path.basename(sys.argv[0]),
#     )
#     sys.exit(2)


def main_pipeline(project_id: str, facility_id: str, sp_index: int=0):

    logging.info(f"[PIPELINE] Start analysis project={project_id}, fac={facility_id}")

    # ====== PATH ======
    base_dir = os.path.join("pea_no_projects", "input", str(project_id))
    out_json = os.path.join(
        base_dir,
        f"{project_id}_NetworkLV{facility_id}_with_MV.json"
    )

    if not os.path.exists(out_json):
        raise FileNotFoundError(f"Input JSON not found: {out_json}")

    json_path = out_json

    # ----- ตั้งชื่อ log ให้ตรงกับ out_json (เฉพาะโหมดไม่ใช่ GUI) -----
    if not IS_GUI:
        try:
            default_log_dir = os.path.join(os.getcwd(), "logs")
            json_name = os.path.splitext(os.path.basename(out_json))[0]
            log_path = os.path.join(default_log_dir, f"{json_name}.run.log")

            setup_logging(
                log_path,
                to_console=True,          # หรือใช้ args.verbose ถ้ามีใน global
                level=logging.INFO,
                tee_stdout_to_file=True
            )
            logging.info("Switched logging in main() to: %s", log_path)
        except Exception as e:
            # ถ้าตั้ง log ไม่ได้ ให้แจ้ง แต่ไม่ให้โปรแกรมตาย
            logging.error(f"Failed to reconfigure logging for JSON '{out_json}': {e}")
    try:
        # 1) ดึง "มิเตอร์" จาก JSON
        meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases = extractMeterData_json(json_path, default_voltage=230.0, drop_zero_load=False, dedup=False)
        
        # 2) ดึง "เส้น" จาก JSON
        lvLinesX, lvLinesY, lvLines, _snap_lv, _snap_map_lv = extractLineData_json(json_path, snap_tolerance=None)

        try:
            mvLinesX, mvLinesY, mvLines, _snap_mv, _snap_map_mv = extractMVLineData_json(
                json_path, snap_tolerance=None, tag_contains="MC"
            )
            logging.info(f"MV lines extracted: {len(mvLines)}")
        except Exception as e:
            logging.warning(f"Extract MV failed: {e}")
            mvLinesX, mvLinesY, mvLines = [], [], []
            _snap_mv, _snap_map_mv = None, {}


        try:
        # ดึง eservice ทั้งหมดจาก JSON ไม่กรอง tag_prefix
            eserviceLinesX, eserviceLinesY, filteredEserviceLines, _snap_svc, _snap_map_svc = extractServiceLines_json(
                json_path,
                snap_tolerance=SNAP_TOLERANCE,
                )
            if not filteredEserviceLines:
                logging.warning("No Eservice lines found in JSON.")
        except Exception as e:
            logging.error(f"Error processing Eservice lines from JSON: {e}")
            eserviceLinesX, eserviceLinesY, filteredEserviceLines = [], [], []
            _snap_svc, _snap_map_svc = None, {}
         
        
        
        snap_map = _snap_map_lv
        # สร้าง per-segment length จาก JSON (ใช้ SHAPE.LEN/สำรอง) โดยอิง snap_map เดิม
        try:
            length_map = build_line_length_map_from_json(
            json_input=out_json,
            coord_snap_map=snap_map,
            length_field="SHAPE.LEN",
            fallback_fields=("Shape_Leng", "SHAPE_Leng"),
            unit_factor=1.0
            )
            logging.info(f"[LEN] length_map ready: {len(length_map)//2} segments")
        except Exception as e:
            logging.warning(f"[LEN] build_line_length_map_from_json failed: {e}")
            length_map = None  # จะ fallback ไปใช้ระยะยูคลิด

        # 3) คำนวณ SNAP_TOLERANCE อัตโนมัติจากชุดข้อมูลจริง (มิเตอร์ + LV + MV)
        SNAP_TOLERANCE = auto_determine_snap_tolerance(
            meterLocations,
            lvLines,
            mvLines,
            reduction_ratio=0.98,
            use_analysis=True
        )
        logging.info(f"Automatically determined SNAP_TOLERANCE: {SNAP_TOLERANCE:.8f} m")

        logging.info(f"Applied coordinate snapping with tolerance: {SNAP_TOLERANCE} m")
        

        # ตรวจสอบสายที่อาจมีปัญหา
        logging.info("Checking for problematic lines after snap...")
        
        # วิเคราะห์ LV lines
        lv_analysis = identify_failed_snap_lines(lvLines, SNAP_TOLERANCE)
        if lv_analysis['total_issues'] > 0:
            logging.warning(f"Found {lv_analysis['total_issues']} problematic LV lines")
            
            # Export สายที่มีปัญหาเพื่อตรวจสอบ
            base_dir = os.path.join("pea_no_projects", "output", str(project_id), "downloads")
            folder_path = base_dir
            ensure_folder_exists(folder_path)
            export_failed_lines_shapefile(
                lv_analysis, 
                lvLines, 
                f"{folder_path}/problematic_lv_lines.shp"
            )
            
            # พยายามแก้ไข
            fixed_lv_lines, fix_log = fix_failed_snap_lines(lv_analysis, lvLines, SNAP_TOLERANCE)
            logging.info(f"Attempted {len(fix_log)} fixes on LV lines")
        try:
            tx_xy = get_transformer_xy_from_json(out_json, facilityid=facility_id)
        except Exception as e:
            logging.error(f"หา TR (x,y) จาก JSON ไม่ได้: {e}")
            return

        # 4) อ่านขนาดหม้อแปลงจาก JSON ตามที่คุณขอให้ย้ายมาใช้ RATEKVA
        try:
            # ถ้ามี FACILITYID ของงานนี้อยู่ในสโคป ให้ส่งเป็น facility_id; ไม่มีก็ปล่อย None
            transformerCapacity_kVA, transformerCapacity, powerFactor = \
                get_transformer_capacity_from_json(json_path, facilityid=None, default_pf=0.875)
            logging.info(
                f"[TR] RATEKVA={transformerCapacity_kVA} kVA  -> capacity≈{transformerCapacity:.2f} kW (pf={powerFactor})"
            )
        except Exception as e:
            logging.error(f"อ่าน RATEKVA จาก JSON ล้มเหลว: {e}")
            return

        # 5) เซ็ตค่าคงที่สาย (ถ้าคุณมีค่ามาตรฐานอื่น ให้ปรับตรงนี้)
        conductorResistance = 0.77009703
        conductorReactance  = 0.3497764
        initialVoltage      = 230

        # 6) (สำคัญ) ให้ตัวแปรที่ส่วนล่างใช้ชื่อเดิมได้
        svcLines = filteredEserviceLines

    except Exception as e:
        logging.error(f"An error occurred during JSON extract: {e}")
        return
    # ====== END JSON-based EXTRACT ======================================
   
    ############ Build initial network (with snapping) ############
    logging.info("Building initial network with snapping …")   
    
        # 1) สร้าง coord_snap_map จาก LV+MV ด้วย SNAP_TOLERANCE
    all_line_coords = []
    for L in (lvLines, mvLines):
        for line in L:
            all_line_coords.extend([(x, y) for x, y in zip(line['X'], line['Y'])
                                    if not (np.isnan(x) or np.isnan(y))])
    coord_snap_map = snap_coordinates_to_tolerance(list(set(all_line_coords)), SNAP_TOLERANCE)

    # 2) สร้าง per-segment length_map จาก JSON (SHAPE.LEN หรือสำรอง)
    length_map = build_line_length_map_from_json(
        json_input=out_json,               # path/dict ของ JSON
        coord_snap_map=coord_snap_map,
        length_field="SHAPE.LEN",
        fallback_fields=("Shape_Leng", "SHAPE_Leng"),
        unit_factor=1.0
    )
    
    G_init, tNode_init, mNodes_init, nm_init, cm_init = buildLVNetworkWithLoads(
    lvLines=lvLines,
    mvLines=mvLines,
    meterLocations=meterLocations,
    transformerLocation=tx_xy,                # มาจาก get_transformer_xy_from_json(...)
    phase_loads=phase_loads,
    conductorResistance=conductorResistance,
    conductorReactance=conductorReactance,
    svcLines=svcLines,
    use_shape_length=True,                    # ใช้ความยาวจาก JSON
    lvData=None,                              # ไม่ต้องส่ง shapefile แล้ว
    length_field="SHAPE.LEN",
    snap_tolerance=SNAP_TOLERANCE,
    line_length_map=length_map,               # << สำคัญ
    coord_snap_map=coord_snap_map             # << สำคัญ
    )

    
    if not nx.is_connected(G_init):
        logging.warning("Initial network not fully connected.")
        components = list(nx.connected_components(G_init))
        logging.warning(f"Network has {len(components)} connected components")
    else:
        logging.info("Initial network is connected.")
    
    # For this run we use LV-based optimization
    optimizedTransformerLocation_LV = optimizeTransformerLocationOnLVCond3(
        meterLocations, phase_loads, initialTransformerLocation,
        lvLines, initialVoltage, conductorResistance, powerFactor,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    
    logging.info(f"Optimized (LV) => {optimizedTransformerLocation_LV}")
    optimizedTransformerLocation = optimizedTransformerLocation_LV
    
    # ใช้ฟังก์ชันที่แก้ไขแล้ว พร้อม coordinate snapping
    G, transformerNode, meterNodes, node_mapping, coord_mapping = buildLVNetworkWithLoads(
    lvLines=lvLines,
    mvLines=mvLines,
    meterLocations=meterLocations,
    transformerLocation=optimizedTransformerLocation,   # หรือ tx_xy / tx_group
    phase_loads=phase_loads,
    conductorResistance=conductorResistance,
    conductorReactance=conductorReactance,
    svcLines=svcLines,
    use_shape_length=True,
    lvData=None,
    length_field="SHAPE.LEN",
    snap_tolerance=SNAP_TOLERANCE,
    line_length_map=length_map,
    coord_snap_map=coord_snap_map
    )

    # ตรวจสอบความสมบูรณ์ของ network
    validation = validate_network_after_snap(G, coord_mapping, meterNodes, transformerNode)

    if not validation['summary']['network_complete']:
        logging.error("Network is incomplete after snap!")
        logging.error(f"Unreachable meters: {len(validation['unreachable_meters'])}")
        
        # อาจต้องปรับ tolerance หรือแก้ไขข้อมูล
        if len(validation['unreachable_meters']) > 0:
            # ลองใช้ tolerance ที่สูงขึ้น
            new_tolerance = SNAP_TOLERANCE * 2
            logging.info(f"Retrying with higher tolerance: {new_tolerance}")
        if not nx.is_connected(G):
            logging.warning("Post-optimization network not fully connected.")
            components = list(nx.connected_components(G))
            logging.warning(f"Network has {len(components)} connected components")
        else:
            logging.info("Post-optimization network is connected.")
    
    splitting_edge, sp_coord, sp_edge_diff, candidate_edges = findSplittingPoint(
        G, transformerNode, meterNodes, coord_mapping,
        powerFactor, initialVoltage,project_id, candidate_index=sp_index
    )
    
    if splitting_edge is None:
        logging.warning("No splitting edge found; single group only. End.")
        return
        
    group1_nodes, group2_nodes = partitionNetworkAtPoint(G, transformerNode, meterNodes, splitting_edge)
    
    voltages, branch_curr, group1_nodes, group2_nodes = performForwardBackwardSweepAndDivideLoads(
        G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, splitting_edge
    )

    nodeToIndex = {mn: i for i, mn in enumerate(meterNodes)}
    group1_meter_nodes = [n for n in group1_nodes if n in nodeToIndex]
    group2_meter_nodes = [n for n in group2_nodes if n in nodeToIndex]
    g1_idx = [nodeToIndex[n] for n in group1_meter_nodes]
    g2_idx = [nodeToIndex[n] for n in group2_meter_nodes]

    # 1) สร้างข้อมูลเฉพาะกลุ่ม 1
    group1_meterLocs = meterLocations[g1_idx]
    loads_g1 = totalLoads[g1_idx]
    # โหลดตามเฟสของ มิเตอร์กลุ่ม 1
    group1_phase_loads = {
        'A': phase_loads['A'][g1_idx],
        'B': phase_loads['B'][g1_idx],
        'C': phase_loads['C'][g1_idx],
    }
        # ข้อมูล peano และ phases ตามกลุ่ม
    peano_g1  = peano[g1_idx]
    phases_g1 = phases[g1_idx]
    
    phase_indexer = build_phase_indexer_from_json(
    json_input=json_path,
    candidate_fields=("PHASEDESIGNATION","PHASEDESIG","PHASE","PHASETYPE"),
    tag_prefix_for_lv=("22LC", "2244LC"),  # ครอบทั้ง TAG "22LCEA..." และ "2244LC..." :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
    subtype_allow=(1,),
    opvolt_max=1000.0
    )

    # เรียก balance Group 1
    new_phases_g1, new_phase_loads_g1 = optimize_phase_balance(
    group1_meterLocs, loads_g1, group1_phase_loads, peano[g1_idx],
    lvData=None,                                  # ไม่ต้องใช้ shapefile
    original_phases=phases[g1_idx].tolist(),
    phase_indexer=phase_indexer,                  # << ใช้อันนี้
    target_unbalance_pct=10.0
    )
    logging.info(
        "Group 1 load balance before -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
        group1_phase_loads['A'].sum(),
        group1_phase_loads['B'].sum(),
        group1_phase_loads['C'].sum()
    )
    # Calculate %Unbalance Before Group1
    g1_a = group1_phase_loads['A'].sum()
    g1_b = group1_phase_loads['B'].sum()
    g1_c = group1_phase_loads['C'].sum()
    g1_avg = (g1_a + g1_b + g1_c) / 3.0
    g1_unb_before = max(abs(g1_a - g1_avg), abs(g1_b - g1_avg), abs(g1_c - g1_avg)) / g1_avg * 100
    logging.info("Group 1 percent unbalance before: %.2f%%", g1_unb_before)
    
    logging.info(
        "Group 1 load balance after  -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
        new_phase_loads_g1['A'].sum(),
        new_phase_loads_g1['B'].sum(),
        new_phase_loads_g1['C'].sum()
    )
    # Calculate %Unbalance After Group1
    new_g1_a = new_phase_loads_g1['A'].sum()
    new_g1_b = new_phase_loads_g1['B'].sum()
    new_g1_c = new_phase_loads_g1['C'].sum()
    new_g1_avg = (new_g1_a + new_g1_b + new_g1_c) / 3.0
    g1_unb_after = max(abs(new_g1_a - new_g1_avg), abs(new_g1_b - new_g1_avg), abs(new_g1_c - new_g1_avg)) / new_g1_avg * 100
    logging.info("Group 1 percent unbalance after: %.2f%%", g1_unb_after)
    # ---- %Unbalance summary (หลังคำนวณก่อน-หลังเสร็จ) ----
    unb_g1_summary = summarize_unbalance_change(group1_phase_loads, new_phase_loads_g1)
    logging.info("Group 1 %%Unbalance summary: %s", unb_g1_summary)
    
    
    
    # 2) กลุ่ม 2
    group2_meterLocs = meterLocations[g2_idx]
    loads_g2 = totalLoads[g2_idx]
    group2_phase_loads = {
        'A': phase_loads['A'][g2_idx],
        'B': phase_loads['B'][g2_idx],
        'C': phase_loads['C'][g2_idx],
    }
    peano_g2  = peano[g2_idx]
    phases_g2 = phases[g2_idx]

    new_phases_g2, new_phase_loads_g2 = optimize_phase_balance(
    group2_meterLocs, loads_g2, group2_phase_loads, peano[g2_idx],
    lvData=None,                                  # ไม่ต้องใช้ shapefile
    original_phases=phases[g2_idx].tolist(),
    phase_indexer=phase_indexer,                  # << ใช้อันนี้
    target_unbalance_pct=10.0
    )
    logging.info(
    "Group 2 load balance before -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
    group2_phase_loads['A'].sum(),
    group2_phase_loads['B'].sum(),
    group2_phase_loads['C'].sum()
    )
    # Calculate %Unbalance Before Group2
    g2_a = group2_phase_loads['A'].sum()
    g2_b = group2_phase_loads['B'].sum()
    g2_c = group2_phase_loads['C'].sum()
    g2_avg = (g2_a + g2_b + g2_c) / 3.0
    g2_unb_before = max(abs(g2_a - g2_avg), abs(g2_b - g2_avg), abs(g2_c - g2_avg)) / g2_avg * 100
    logging.info("Group 2 percent unbalance before: %.2f%%", g2_unb_before)
    
    logging.info(
        "Group 2 load balance after  -> A: %.1f kW, B: %.1f kW, C: %.1f kW",
        new_phase_loads_g2['A'].sum(),
        new_phase_loads_g2['B'].sum(),
        new_phase_loads_g2['C'].sum()
    )
    # Calculate %Unbalance After Group2
    new_g2_a = new_phase_loads_g2['A'].sum()
    new_g2_b = new_phase_loads_g2['B'].sum()
    new_g2_c = new_phase_loads_g2['C'].sum()
    new_g2_avg = (new_g2_a + new_g2_b + new_g2_c) / 3.0
    g2_unb_after = max(abs(new_g2_a - new_g2_avg), abs(new_g2_b - new_g2_avg), abs(new_g2_c - new_g2_avg)) / new_g2_avg * 100
    logging.info("Group 2 percent unbalance after: %.2f%%", g2_unb_after)
    unb_g2_summary = summarize_unbalance_change(group2_phase_loads, new_phase_loads_g2)
    logging.info("Group 2 %%Unbalance summary: %s", unb_g2_summary)
    
    
    
    dist_arr = []
    for n in meterNodes:
        try:
            dval = nx.shortest_path_length(G, transformerNode, n, weight='weight')
        except nx.NetworkXNoPath:
            dval = float('inf')
        dist_arr.append(dval)
        
    voltA = {n: voltages[n]['A'] for n in voltages}
    voltB = {n: voltages[n]['B'] for n in voltages}
    voltC = {n: voltages[n]['C'] for n in voltages}
    
    result_df = pd.DataFrame({
        'Peano Meter': peano,
        'Final Voltage A (V)': [voltA.get(n, np.nan) for n in meterNodes],
        'Final Voltage B (V)': [voltB.get(n, np.nan) for n in meterNodes],
        'Final Voltage C (V)': [voltC.get(n, np.nan) for n in meterNodes],
        'Distance to Transformer (m)': dist_arr,
        'Load A (kW)': phase_loads['A'],
        'Load B (kW)': phase_loads['B'],
        'Load C (kW)': phase_loads['C'],
        'Group': ['Group 1' if n in group1_nodes else 'Group 2' for n in meterNodes],
        'Phases': phases
    })
    result_df["Meter X"] = meterLocations[:, 0]
    result_df["Meter Y"] = meterLocations[:, 1]
    # 3) เอาค่าที่ได้ไปอัปเดตใน result_df
    for local_i, global_i in enumerate(g1_idx):
        result_df.at[global_i, 'New Phase']  = new_phases_g1[local_i]
        result_df.at[global_i, 'New Load A'] = new_phase_loads_g1['A'][local_i]
        result_df.at[global_i, 'New Load B'] = new_phase_loads_g1['B'][local_i]
        result_df.at[global_i, 'New Load C'] = new_phase_loads_g1['C'][local_i]

    for local_i, global_i in enumerate(g2_idx):
        result_df.at[global_i, 'New Phase']  = new_phases_g2[local_i]
        result_df.at[global_i, 'New Load A'] = new_phase_loads_g2['A'][local_i]
        result_df.at[global_i, 'New Load B'] = new_phase_loads_g2['B'][local_i]
        result_df.at[global_i, 'New Load C'] = new_phase_loads_g2['C'][local_i]

    for n in G.nodes():
        if n in group1_nodes:
            G.nodes[n]['group'] = 1
        elif n in group2_nodes:
            G.nodes[n]['group'] = 2
        else:
            G.nodes[n]['group'] = 0
            
    g1_load = sum(G.nodes[n]['load_A'] + G.nodes[n]['load_B'] + G.nodes[n]['load_C'] for n in group1_nodes)
    g2_load = sum(G.nodes[n]['load_A'] + G.nodes[n]['load_B'] + G.nodes[n]['load_C'] for n in group2_nodes)
    logging.info(f"Group 1 total load: {g1_load:.2f} kW | Group 2 total load: {g2_load:.2f} kW")

    group1_meterLocs = meterLocations[g1_idx]
    group1_phase_loads = {
        'A': phase_loads['A'][g1_idx],
        'B': phase_loads['B'][g1_idx],
        'C': phase_loads['C'][g1_idx]
    }
    group2_meterLocs = meterLocations[g2_idx]
    group2_phase_loads = {
        'A': phase_loads['A'][g2_idx],
        'B': phase_loads['B'][g2_idx],
        'C': phase_loads['C'][g2_idx]
    }
    
    logging.info("Calculate LoadCenter each Group...")
    logging.info("Optimizing Transformer Location each Group...")
    
    
    lc_g1 = calculateNetworkLoadCenter(
        group1_meterLocs, 
        group1_phase_loads, 
        lvLines, 
        mvLines, 
        conductorResistance,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    lc_g2 = calculateNetworkLoadCenter(
        group2_meterLocs, 
        group2_phase_loads, 
        lvLines, 
        mvLines, 
        conductorResistance,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    logging.info(f"Group 1 load center = {lc_g1}")
    logging.info(f"Group 2 load center = {lc_g2}")

    logging.info("Optimizing Transformer Location each Group...")
    optimizedTransformerLocationGroup1 = optimizeGroup(
        group1_meterLocs,
        group1_phase_loads,
        lc_g1,
        lvLines, 
        mvLines,
        initialVoltage, 
        conductorResistance,
        powerFactor, 
        epsilon_junction=2.0,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    optimizedTransformerLocationGroup2 = optimizeGroup(
        group2_meterLocs,
        group2_phase_loads,
        lc_g2,
        lvLines, 
        mvLines,
        initialVoltage, 
        conductorResistance,
        powerFactor, 
        epsilon_junction=2.0,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    logging.info(f"Group 1 optimized TR = {optimizedTransformerLocationGroup1}")
    logging.info(f"Group 2 optimized TR = {optimizedTransformerLocationGroup2}")

      
    if optimizedTransformerLocationGroup1 is not None:
        G_g1, tNode_g1, mNodes_g1, nm_g1, cm_g1 = buildLVNetworkWithLoads(
            lvLines, mvLines,
            group1_meterLocs,
            optimizedTransformerLocationGroup1,
            group1_phase_loads,
            conductorResistance,
            conductorReactance=conductorReactance,
            use_shape_length=True,
            lvData=lvData,
            svcLines=svcLines
        )
        dist_g1 = []
        for i, mNode in enumerate(mNodes_g1):
            try:
                dist_g1.append(nx.shortest_path_length(G_g1, tNode_g1, mNode, weight='weight'))
            except nx.NetworkXNoPath:
                dist_g1.append(float('inf'))
                
        # เปลี่ยนไปใช้ calculateUnbalancedPowerFlow แทน calculatePowerLoss
        node_voltages_g1, branch_currents_g1, total_power_loss_g1 = calculateUnbalancedPowerFlow(
        G_g1, tNode_g1, mNodes_g1, powerFactor, initialVoltage
        )
        
        for i, node in enumerate(mNodes_g1):
            global_idx = g1_idx[i]
            result_df.at[global_idx, 'Distance to Transformer (m)'] = dist_g1[i]
            result_df.at[global_idx, 'Final Voltage A (V)'] = abs(node_voltages_g1[node]['A'])
            result_df.at[global_idx, 'Final Voltage B (V)'] = abs(node_voltages_g1[node]['B'])
            result_df.at[global_idx, 'Final Voltage C (V)'] = abs(node_voltages_g1[node]['C'])
            
    if optimizedTransformerLocationGroup2 is not None:
        G_g2, tNode_g2, mNodes_g2, nm_g2, cm_g2 = buildLVNetworkWithLoads(
            lvLines, mvLines,
            group2_meterLocs,
            optimizedTransformerLocationGroup2,
            group2_phase_loads,
            conductorResistance,
            conductorReactance=conductorReactance,
            use_shape_length=True,
            lvData=lvData,
            svcLines=svcLines
        )
        dist_g2 = []
        for i, mNode in enumerate(mNodes_g2):
            try:
                dist_g2.append(nx.shortest_path_length(G_g2, tNode_g2, mNode, weight='weight'))
            except nx.NetworkXNoPath:
                dist_g2.append(float('inf'))
                
        # เปลี่ยนไปใช้ calculateUnbalancedPowerFlow แทน calculatePowerLoss
        node_voltages_g2, branch_currents_g2, total_power_loss_g2 = calculateUnbalancedPowerFlow(
        G_g2, tNode_g2, mNodes_g2, powerFactor, initialVoltage
        )
        
        for i, node in enumerate(mNodes_g2):
            global_idx = g2_idx[i]
            result_df.at[global_idx, 'Distance to Transformer (m)'] = dist_g2[i]
            result_df.at[global_idx, 'Final Voltage A (V)'] = abs(node_voltages_g2[node]['A'])
            result_df.at[global_idx, 'Final Voltage B (V)'] = abs(node_voltages_g2[node]['B'])
            result_df.at[global_idx, 'Final Voltage C (V)'] = abs(node_voltages_g2[node]['C'])

    # Calculate Total Loss
    line_loss_g1_kW = total_power_loss_g1 / 1000.0
    tx_loss_g1_kW = get_transformer_losses(g1_load)
    total_system_loss_g1 = line_loss_g1_kW + tx_loss_g1_kW
    # ---- Loss summary ต่อกลุ่ม (ใช้ TX ที่ optimize แล้ว) ----
    loss_g1_summary = summarize_loss_change(
        lvLines, mvLines, group1_meterLocs, optimizedTransformerLocationGroup1,
        group1_phase_loads, new_phase_loads_g1,
        conductorResistance=conductorResistance,
        powerFactor=powerFactor,
        initialVoltage=initialVoltage,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )
    
    line_loss_g2_kW = total_power_loss_g2 / 1000.0
    tx_loss_g2_kW = get_transformer_losses(g2_load)
    total_system_loss_g2 = line_loss_g2_kW + tx_loss_g2_kW
    loss_g2_summary = summarize_loss_change(
        lvLines, mvLines, group2_meterLocs, optimizedTransformerLocationGroup2,
        group2_phase_loads, new_phase_loads_g2,
        conductorResistance=conductorResistance,
        powerFactor=powerFactor,
        initialVoltage=initialVoltage,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines,
        snap_tolerance=SNAP_TOLERANCE
    )
    
    # Forecast Future Load
    future_g1_load = growthRate(g1_load, annual_growth=0.06, years=4)
    future_g2_load = growthRate(g2_load, annual_growth=0.06, years=4)
    
    # Select Transformer size for each group (using your document)
    rating_g1 = Lossdocument(future_g1_load)
    rating_g2 = Lossdocument(future_g2_load)

    
    
    # Max Distance from Group 1,2
    max_dist1 = max(dist_g1)
    max_dist2 = max(dist_g2)

    logging.info('############ Result ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW, Future growthrate(4yr@6%)={future_g1_load:.2f} kW, Chosen TX1={rating_g1} kVA, Max Distance Group1={max_dist1:.1f}m.")
    logging.info(f"Group 2 => Load={g2_load:.2f} kW, Future growthrate(4yr@6%)={future_g2_load:.2f} kW, Chosen TX2={rating_g2} kVA, Max Distance Group2={max_dist2:.1f}m.")
    logging.info('########## Loss Report #########')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW | LineLoss={line_loss_g1_kW:.2f} kW | TxLoss={tx_loss_g1_kW:.2f} kW => TOTAL={total_system_loss_g1:.2f} kW")
    logging.info("Group 1 line loss: BEFORE %.2f kW -> AFTER %.2f kW",
             loss_g1_summary["loss_before_kW"], loss_g1_summary["loss_after_kW"])
    logging.info(f"Group 2 => Load={g2_load:.2f} kW | LineLoss={line_loss_g2_kW:.2f} kW | TxLoss={tx_loss_g2_kW:.2f} kW => TOTAL={total_system_loss_g2:.2f} kW")
    logging.info("Group 2 line loss: BEFORE %.2f kW -> AFTER %.2f kW",
             loss_g2_summary["loss_before_kW"], loss_g2_summary["loss_after_kW"])
    logging.info("####### %UnBalance Report ######")
    logging.info("Group 1 %%Unbalance: BEFORE %.2f%% -> AFTER %.2f%%", g1_unb_before, g1_unb_after)
    logging.info("Group 2 %%Unbalance: BEFORE %.2f%% -> AFTER %.2f%%", g2_unb_before, g2_unb_after)

    base_dir = os.path.join("pea_no_projects", "output", str(project_id))
    output_folder = base_dir

    results = {
        # "tr_pea_no": transformerPEA_No,
        # "tr_kva": round(transformerCapacity_kVA,2),
        # "tr_loss": round(transformerLoss,2),
        # "tr_Line_lengh": round(transformerLine_lengh,2),
        # "tr_PLOADPEAK" : transformerPLOADPEAK,

        "tr_pea_no": "transformerPEA_No",
        "tr_kva": round(transformerCapacity_kVA,2),
        "tr_loss": 1,
        "tr_Line_lengh": 1,
        "tr_PLOADPEAK" : "transformerPLOADPEAK",

        "g1_load": round(g1_load, 2),
        "future_g1_load": round(future_g1_load, 2),
        "rating_g1": rating_g1,
        "line_loss_g1_kW": round(line_loss_g1_kW, 2),
        "tx_loss_g1_kW": round(tx_loss_g1_kW, 2),
        "total_system_loss_g1": round(total_system_loss_g1, 2),

        "g2_load": round(g2_load, 2),
        "future_g2_load": round(future_g2_load, 2),
        "rating_g2": rating_g2,
        "line_loss_g2_kW": round(line_loss_g2_kW, 2),
        "tx_loss_g2_kW": round(tx_loss_g2_kW, 2),
        "total_system_loss_g2": round(total_system_loss_g2, 2),

        "Group_1_percent_unbalance_before": round(g1_unb_before, 1),
        "Group_1_percent_unbalance_after":round(g1_unb_after, 1),

        "Group_2_percent_unbalance_before": round(g2_unb_before, 1),
        "Group_2_percent_unbalance_after":round(g2_unb_after, 1),

        "Max_Distance_Group1": round(max_dist1, 1),
        "Max_Distance_Group2": round(max_dist2, 1),

        "group1_load_balance_before": {
            "A": round(group1_phase_loads['A'].sum(), 1),
            "B": round(group1_phase_loads['B'].sum(), 1),
            "C": round(group1_phase_loads['C'].sum(), 1),
        },
        "group1_load_balance_after": {
            "A": round(new_phase_loads_g1['A'].sum(), 1),
            "B": round(new_phase_loads_g1['B'].sum(), 1),
            "C": round(new_phase_loads_g1['C'].sum(), 1),
        },
        "group2_load_balance_before": {
            "A": round(group2_phase_loads['A'].sum(), 1),
            "B": round(group2_phase_loads['B'].sum(), 1),
            "C": round(group2_phase_loads['C'].sum(), 1),
        },
        "group2_load_balance_after": {
            "A": round(new_phase_loads_g2['A'].sum(), 1),
            "B": round(new_phase_loads_g2['B'].sum(), 1),
            "C": round(new_phase_loads_g2['C'].sum(), 1),
        },
    }

    # บันทึกลงไฟล์ results.json
    with open(os.path.join(output_folder, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    point_coords = []
    attributes_list = []
    if sp_coord is not None:
        # Convert sp_coord to tuple for consistency
        point_coords.append(tuple(sp_coord))
        attributes_list.append({'Name': 'Splitting Point'})
    if optimizedTransformerLocationGroup1 is not None:
        point_coords.append(tuple(optimizedTransformerLocationGroup1))
        attributes_list.append({'Name': 'Group 1 Transformer'})
    if optimizedTransformerLocationGroup2 is not None:
        point_coords.append(tuple(optimizedTransformerLocationGroup2))
        attributes_list.append({'Name': 'Group 2 Transformer'})
    if point_coords:
        base_dir = os.path.join("pea_no_projects", "output", str(project_id), "downloads")
        folder_path = base_dir
        ensure_folder_exists(folder_path)
        exportPointsToShapefile(point_coords, f"{folder_path}/optimized_transformer_locations.shp", attributes_list)
        g1_plot_indices = np.array(g1_idx, dtype=int)
        g2_plot_indices = np.array(g2_idx, dtype=int)
    
    # Export to CSV
    base_dir = os.path.join("pea_no_projects", "output", str(project_id), "downloads")
    folder_path = base_dir
    ensure_folder_exists(folder_path)
    csv_path = f"{folder_path}/optimized_transformer_locations.csv"
    result_df.to_csv(csv_path, index=False)
    logging.info(f"Result CSV saved: {csv_path}")
    
    # Export to shapefile
    exportResultDFtoShapefile(result_df, f"{folder_path}/result_meters.shp")
    
        # --- เตรียมค่าแบบปลอดภัยก่อนเรียก plotResults ---
    def _xs(lines): return [l['X'] for l in lines] if lines else []
    def _ys(lines): return [l['Y'] for l in lines] if lines else []

    lvX = _xs(lvLines)
    lvY = _ys(lvLines)
    mvX = _xs(mvLines)
    mvY = _ys(mvLines)
    svcX = _xs(filteredEserviceLines)
    svcY = _ys(filteredEserviceLines)

    # initial TX: ใช้ tx_xy ที่อ่านจาก JSON; ถ้าไม่มี ให้เป็น None (plotResults เวอร์ชันที่แก้แล้วจะข้ามให้)
    initial_tx = tx_xy if (tx_xy and len(tx_xy) == 2) else None

    # indices ของกลุ่ม แปลงเป็น array int ถ้ายังไม่ใช่
    g1_plot_indices = np.asarray(g1_plot_indices, dtype=int) if 'g1_plot_indices' in locals() else np.array([], dtype=int)
    g2_plot_indices = np.asarray(g2_plot_indices, dtype=int) if 'g2_plot_indices' in locals() else np.array([], dtype=int)

    # --- เรียก plotResults ---
    plotResults_NGUI(
    project_id,
    [l['X'] for l in lvLines], [l['Y'] for l in lvLines],
    [l['X'] for l in mvLines], [l['Y'] for l in mvLines],   # << MV ต้องส่งแบบนี้
    [l['X'] for l in filteredEserviceLines],
    [l['Y'] for l in filteredEserviceLines],
    meterLocations,
    initialTransformerLocation=tx_xy,                       # ถ้ามี
    optimizedTransformerLocation=optimizedTransformerLocation_LV,
    group1_indices=g1_plot_indices,
    group2_indices=g2_plot_indices,
    splitting_point_coords=sp_coord,
    coord_mapping=coord_mapping,
    optimizedTransformerLocationGroup1=optimizedTransformerLocationGroup1,
    optimizedTransformerLocationGroup2=optimizedTransformerLocationGroup2,
    transformer_losses=None,
    phases=phases,
    result_df=result_df,
    G=G
    )


    G = addNodeLabels(G, None, sp_edge_diff)
    # plotGraphWithLabels(G, coord_mapping, best_edge_diff=sp_edge_diff, best_edge=splitting_edge)
    
    logging.info("Initial processing complete. Proceeding with group-level optimization and output.")
    
    # 5‑A)  build initial split dict so the button works FIRST time
    # latest_split_result, reopt_btn
    # latest_split_result = {
    #     'best_edge': splitting_edge,
    #     'group1'   : {
    #         'idx'        : np.array(g1_idx, dtype=int),
    #         'meter_locs' : group1_meterLocs,
    #         'phase_loads': group1_phase_loads,
    #         'tx_loc'     : optimizedTransformerLocationGroup1
    #     },
    #     'group2'   : {
    #         'idx'        : np.array(g2_idx, dtype=int),
    #         'meter_locs' : group2_meterLocs,
    #         'phase_loads': group2_phase_loads,
    #         'tx_loc'     : optimizedTransformerLocationGroup2
    #     }
    # }
    # แทนบล็อกเดิม:
    # if reopt_btn is not None:
    #     reopt_btn.config(state=tk.NORMAL)

    # btn = globals().get('reopt_btn', None)
    # if btn is not None:
    #     try:
    #         btn.config(state=tk.NORMAL)
    #     except Exception:
    #         # เผื่อกรณีไม่มี Tk context ในโหมด headless
    #         pass


    # 5‑B)  open the candidate‑edge dialog
    # gui_candidate_input(G, transformerNode, meterNodes,
    #                     node_mapping, coord_mapping,
    #                     meterLocations, phase_loads,
    #                     lvLines, mvLines, filteredEserviceLines,
    #                     initialTransformerLocation,
    #                     powerFactor, initialVoltage,
    #                     conductorResistance,
    #                     peano, phases,
    #                     conductorReactance, lvData, svcLines)

    logging.info("Program finished successfully.")

def plotResults_NGUI(project_id, lvLinesX, lvLinesY, mvLinesX, mvLinesY, eserviceLinesX, eserviceLinesY,
                meterLocations, initialTransformerLocation, optimizedTransformerLocation,
                group1_indices, group2_indices, splitting_point_coords=None, coord_mapping=None,
                optimizedTransformerLocationGroup1=None, optimizedTransformerLocationGroup2=None,
                transformer_losses=None, phases=None, result_df=None, G=None):
    
    # Step 1: โปรเจกชัน TM3 48N (ใส่พิกัดเริ่มต้นตามของคุณ)
    # TM3 Thailand 48N = EPSG:32648, หรือใช้ Proj string
    proj_tm3 = pyproj.CRS.from_proj4("+proj=utm +zone=47 +datum=WGS84 +units=m +no_defs")
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(proj_tm3, proj_wgs84, always_xy=True)

    base_dir = os.path.join("pea_no_projects", "output", str(project_id))
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    # ตัวอย่างข้อมูลจาก lvLinesX, lvLinesY (list of list)
    # สมมุติว่า lvLinesX = [[x1, x2], [x3, x4]], lvLinesY = [[y1, y2], [y3, y4]]
    featuresLV = []    
    
    for x_list, y_list in zip(lvLinesX, lvLinesY):
        latlons = [transformer.transform(x, y) for x, y in zip(x_list, y_list)]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": latlons
            },
            "properties": {
                "type": "LV Line"
            }
        }
        featuresLV.append(feature)

    geojson_output = {
        "type": "FeatureCollection",
        "features": featuresLV
    }

    # เขียนลงไฟล์หรือ return
    output_path = os.path.join(output_dir, "lv_lines.geojson")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_output, f, indent=2)    

    featuresMV = []

    for mvx_list, mvy_list in zip(mvLinesX, mvLinesY):
        latlons = [transformer.transform(x, y) for x, y in zip(mvx_list, mvy_list)]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": latlons
            },
            "properties": {
                "type": "MV Line"
            }
        }
        featuresMV.append(feature)

    geojson_output = {
        "type": "FeatureCollection",
        "features": featuresMV
    }

    # เขียนลงไฟล์หรือ return
    output_path = os.path.join(output_dir, "mv_lines.geojson")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_output, f, indent=2)

    # 1. สร้าง transformer สำหรับแปลงจาก TM3 Zone 48N → WGS84
    proj_tm3 = pyproj.CRS.from_proj4("+proj=utm +zone=47 +datum=WGS84 +units=m +no_defs")
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(proj_tm3, proj_wgs84, always_xy=True)

    # 2. สมมุติว่ามีตัวแปรเหล่านี้จากโค้ดเดิม
    # meterLocations: numpy array shape (N,2)
    # group1_indices, group2_indices: list of indices

    # 3. แปลงพิกัดแต่ละกลุ่ม
    group1_lonlat = [
        transformer.transform(x, y)
        for x, y in zip(
            meterLocations[group1_indices, 0],
            meterLocations[group1_indices, 1]
        )
    ]

    group2_lonlat = [
        transformer.transform(x, y)
        for x, y in zip(
            meterLocations[group2_indices, 0],
            meterLocations[group2_indices, 1]
        )
    ]

    # สร้างฟังก์ชันช่วยสร้าง voltage_text ต่อมิเตอร์
    def get_voltage_text(i):
        connected_phases = phases[i].upper().strip()
        voltage_text = ''
        for ph in ['A', 'B', 'C']:
            if ph in connected_phases:
                colname = f'Final Voltage {ph} (V)'
                if colname in result_df.columns:
                    vval = result_df.iloc[i][colname]
                    if pd.notnull(vval):
                        voltage_text += f'{ph}:{vval:.1f}V\n'
                    else:
                        voltage_text += f'{ph}:N/A\n'
        return voltage_text.strip()

    # แปลงพิกัดและเพิ่ม voltage text

    # 4. สร้าง GeoJSON FeatureCollection
    features = []

    for idx in group1_indices:
        x, y = meterLocations[idx]
        lon, lat = transformer.transform(x, y)
        voltage_text = get_voltage_text(idx)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "group": 1,
                "voltage_text": voltage_text
            }
        })

    for idx in group2_indices:
        x, y = meterLocations[idx]
        lon, lat = transformer.transform(x, y)
        voltage_text = get_voltage_text(idx)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "group": 2,
                "voltage_text": voltage_text
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # บันทึกลงไฟล์
    output_path = os.path.join(output_dir, "meter_groups.geojson")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    features = []

    # Initial Transformer
    if initialTransformerLocation is not None:
        lon, lat = transformer.transform(initialTransformerLocation[0], initialTransformerLocation[1])
        features.append(Feature(
            geometry=Point((lon, lat)),
            properties={"name": "Initial Transformer", "group": "initial"}
        ))

    # Splitting Point
    if splitting_point_coords is not None:
        lon, lat = transformer.transform(splitting_point_coords[0], splitting_point_coords[1])
        features.append(Feature(
            geometry=Point((lon, lat)),
            properties={"name": "Splitting Point", "group": "splitting"}
        ))

    # Group 1 Transformer
    if optimizedTransformerLocationGroup1 is not None:
        lon, lat = transformer.transform(optimizedTransformerLocationGroup1[0], optimizedTransformerLocationGroup1[1])
        features.append(Feature(
            geometry=Point((lon, lat)),
            properties={"name": "Group 1 Transformer", "group": "group1"}
        ))

    # Group 2 Transformer
    if optimizedTransformerLocationGroup2 is not None:
        lon, lat = transformer.transform(optimizedTransformerLocationGroup2[0], optimizedTransformerLocationGroup2[1])
        features.append(Feature(
            geometry=Point((lon, lat)),
            properties={"name": "Group 2 Transformer", "group": "group2"}
        ))

    # สร้าง FeatureCollection
    feature_collection = FeatureCollection(features)

    # บันทึกไฟล์
    output_path = os.path.join(output_dir, "feature_groups.geojson")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, ensure_ascii=False, indent=2)

    
    print("Export OK!")