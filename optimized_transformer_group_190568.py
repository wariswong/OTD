import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import shapefile
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog, messagebox
from tqdm import tqdm
import logging
import sys
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk



##############################################################################
# 1) A CUSTOM LOG HANDLER THAT WRITES TO A TKINTER TEXT WIDGET
##############################################################################
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

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

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
        self.ax.clear()
        # วาด LV & MV lines
        for x, y in zip(self.lvLinesX, self.lvLinesY):
            plt.plot(x, y, color='lime', linewidth=1, linestyle='--', label='LV Line' if 'LV Line' not in plt.gca().get_legend_handles_labels()[1] else "")
        for x, y in zip(self.mvLinesX, self.mvLinesY):
            plt.plot(x, y, color='maroon', linewidth=1, linestyle='-.', label='MV Line' if 'MV Line' not in plt.gca().get_legend_handles_labels()[1] else "")
        # วาด meter locations
        self.ax.plot(self.meterLocs[:,0], self.meterLocs[:,1], 'k.', markersize=3)
        # ไฮไลต์ current edge
        u, v = self.edge_df.iloc[self.curr_idx]['Edge']
        x1, y1 = self.coord[u]
        x2, y2 = self.coord[v]
        self.ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3)
        self.ax.set_aspect('equal')
        self.canvas.draw()

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
                    
##############################################################################
# 2) YOUR EXISTING FUNCTIONS
#    NOTE: loadShapefiles() is updated so it DOES NOT reconfigure the logger.
##############################################################################

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

# Global variables to store the shapefile data
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
reopt_btn            = None 
lvLines              = None   # becomes a list of LV line dicts after main()
mvLines              = None   # list of MV line dicts
initialVoltage       = None   # 230 V (set in main)
conductorResistance  = None   # Ω/km (set in main)
powerFactor          = None   # 0.875 (set in main)
_EDGE_DF_CACHE       = None  

   

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
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logging.info(f"Logger file handler updated: {log_filename}")

    logging.info("All shapefiles loaded successfully.")
    



def extractLineData(lineData):
    logging.info("Extracting line data (simple version, no filter).")
    shapes = lineData.shapes()
    linesX = []
    linesY = []
    lines = []
    for shape in shapes:
        x = [point[0] for point in shape.points]
        y = [point[1] for point in shape.points]
        linesX.append(x)
        linesY.append(y)
        lines.append({'X': x, 'Y': y})
    return linesX, linesY, lines

def extractLineDataWithAttributes(lineData, required_field, required_value):
    logging.info(f"Extracting line data with attribute filter: {required_field} == {required_value}")
    shapes = lineData.shapes()
    records = lineData.records()
    fields = [field[0] for field in lineData.fields[1:]]
    linesX = []
    linesY = []
    lines = []
    for shape, record in zip(shapes, records):
        attributes = dict(zip(fields, record))
        if attributes.get(required_field) == required_value:
            x = [p[0] for p in shape.points]
            y = [p[1] for p in shape.points]
            linesX.append(x)
            linesY.append(y)
            lines.append({'X': x, 'Y': y, 'ATTRIBUTES': attributes})
    logging.info(f"Found {len(lines)} lines matching {required_field}={required_value}.")
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

########################################
# 2. BUILDING THE NETWORK
########################################

def buildLVNetworkWithLoads(lvLines, mvLines, meterLocations, transformerLocation, phase_loads, conductorResistance, 
                          conductorReactance=None,*,svcLines=None, use_shape_length=False, lvData=None, length_field="Shape_Leng"):
    
    if svcLines is None:          # ถ้า caller ไม่ส่ง → ใช้ list ว่าง
        svcLines = []
    logging.info("Building LV network with loads (optimized)...")
    G = nx.Graph()
    node_id = 0
    node_mapping = {}
    coord_mapping = {}
    lv_nodes = set()
    
    # ตรวจสอบพารามิเตอร์
    if use_shape_length and lvData is None:
        logging.error("lvData must be provided when use_shape_length=True")
        raise ValueError("lvData must be provided when use_shape_length=True")

    # สร้างการจับคู่ระหว่างเส้นใน lvLines และข้อมูลใน lvData
    if use_shape_length:
        # ดึงข้อมูล shape และ records จาก lvData
        lv_shapes = lvData.shapes()
        lv_records = lvData.records()
        lv_fields = [field[0] for field in lvData.fields[1:]]
        
        # ตรวจสอบว่ามีฟิลด์ length_field หรือไม่
        if length_field not in lv_fields:
            logging.warning(f"Field '{length_field}' not found in lvData. Falling back to coordinate-based distance.")
            use_shape_length = False
        else:
            # สร้าง dictionary สำหรับการค้นหาความยาวของเส้นอย่างรวดเร็ว
            line_length_map = {}
            for i, (shape, record) in enumerate(zip(lv_shapes, lv_records)):
                # สร้างคีย์จากจุดเริ่มต้นและจุดสิ้นสุด
                if len(shape.points) >= 2:
                    start_point = tuple(shape.points[0])
                    end_point = tuple(shape.points[-1])
                    key = (start_point, end_point)
                    
                    # ดึงค่าความยาวจาก record
                    try:
                        attrs = dict(zip(lv_fields, record))
                        length = float(attrs[length_field])
                        line_length_map[key] = length
                        # เพิ่มอีกคีย์หนึ่งโดยสลับจุดเริ่มต้นและจุดสิ้นสุด (กรณีเส้นไม่มีทิศทาง)
                        line_length_map[(end_point, start_point)] = length
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Error converting length value for shape {i}: {e}")
            
            logging.info(f"Created mapping for {len(line_length_map)} line segments using field '{length_field}'.")

    # เพิ่ม LV lines (ปรับปรุงให้รองรับ use_shape_length)
    for line in lvLines:
        x = line['X']
        y = line['Y']
        # สร้าง list of tuples แทนการใช้ zip แล้วค่อย loop
        coords = [(x[i], y[i]) for i in range(len(x)) if not np.isnan(x[i])]
        
        prev_node = None
        for i, coord in enumerate(coords):
            if coord not in node_mapping:
                node_mapping[coord] = node_id
                coord_mapping[node_id] = coord
                node_id += 1
            current_node = node_mapping[coord]
            lv_nodes.add(current_node)
            
            if prev_node is not None:
                # ตรวจสอบว่าจะใช้ความยาวจาก Shape_Leng หรือไม่
                if use_shape_length:
                    prev_coord = coords[i-1]
                    # พยายามหาความยาวจาก line_length_map
                    segment_key = (prev_coord, coord)
                    if segment_key in line_length_map:
                        # ใช้ความยาวจาก Shape_Leng
                        distance = line_length_map[segment_key]
                        logging.debug(f"Using shape length {distance} for segment {prev_coord} to {coord}")
                    else:
                        # ถ้าไม่พบในการจับคู่ ให้คำนวณจากพิกัด
                        distance = np.hypot(prev_coord[0] - coord[0], prev_coord[1] - coord[1])
                        logging.debug(f"Shape length not found, using coordinate distance {distance} for segment {prev_coord} to {coord}")
                else:
                    # คำนวณระยะทางโดยตรงจากพิกัด (วิธีเดิม)
                    prev_coord = coords[i-1]
                    distance = np.hypot(prev_coord[0] - coord[0], prev_coord[1] - coord[1])
                
                resistance = distance / 1000 * conductorResistance
                # กำหนดค่า reactance
                reactance = (distance / 1000 * conductorReactance) if conductorReactance is not None else 0.1 * resistance
                G.add_edge(prev_node, current_node, weight=distance, resistance=resistance, reactance=reactance)
            
            prev_node = current_node
            
    # เพิ่ม MV lines (ทำแบบเดียวกับ LV)
    for line in mvLines:
        x = line['X']
        y = line['Y']
        coords = [(x[i], y[i]) for i in range(len(x)) if not np.isnan(x[i])]
        
        prev_node = None
        for i, coord in enumerate(coords):
            if coord not in node_mapping:
                node_mapping[coord] = node_id
                coord_mapping[node_id] = coord
                node_id += 1
            current_node = node_mapping[coord]
            
            if prev_node is not None:
                prev_coord = coords[i-1]
                distance = np.hypot(prev_coord[0] - coord[0], prev_coord[1] - coord[1])
                resistance = distance / 1000 * conductorResistance
                reactance = (distance / 1000 * conductorReactance) if conductorReactance is not None else 0.1 * resistance
                G.add_edge(prev_node, current_node, weight=distance, resistance=resistance, reactance=reactance)
            prev_node = current_node

    # 4) CONNECT METERS (ใช้ service‑line ถ้ามี ไม่งั้น KD‑Tree)
    # ------------------------------------------------------------------
    from scipy.spatial import cKDTree

    # -- 4.1 เตรียม lookup ปลาย service‑line (SUBTYPE 5) --
    svc_tol = 0.05      # หน่วยเดียวกับพิกัด (≈ 5 ซม. ปรับได้)
    svc_ends = []       # (ปลาย‑A, ปลาย‑B, pts ทั้งเส้น)
    for line in svcLines:
        pts = [(x, y) for x, y in zip(line['X'], line['Y']) if not np.isnan(x)]
        if len(pts) >= 2:
            svc_ends.append((np.array(pts[0]), np.array(pts[-1]), pts))

    # -- 4.2 KD‑Tree ของโหนด LV (ใช้ต่อได้ทั้งสองกรณี) --
    if lv_nodes:
        lv_coords = np.array([coord_mapping[n] for n in lv_nodes])
        lv_kdtree = cKDTree(lv_coords)
        lv_nodes_list = list(lv_nodes)
        def nearest_lv(pt):
            dist, idx = lv_kdtree.query(pt)
            return lv_nodes_list[idx], dist
    else:
        lv_kdtree = None
        def nearest_lv(pt): return (None, None)

    # -- 4.3 loop มิเตอร์ทั้งหมด --
    meterNodes = []
    for idx, m_xy in enumerate(meterLocations):
        meterNode = node_id
        node_mapping[tuple(m_xy)] = meterNode
        coord_mapping[meterNode]  = tuple(m_xy)
        node_id += 1
        G.add_node(meterNode)

        for ph in 'ABC':
            G.nodes[meterNode][f'load_{ph}'] = phase_loads[ph][idx]

        # --- หา service‑line ที่ชนมิเตอร์ ---
        svc_hit = None
        for p0, p1, pts in svc_ends:
            if np.linalg.norm(m_xy - p0) < svc_tol:
                svc_hit = (p0, p1, pts); break
            if np.linalg.norm(m_xy - p1) < svc_tol:
                svc_hit = (p1, p0, pts); break

        if svc_hit:      # ↳ ใช้ service‑line
            m_end, lv_end, pts = svc_hit
            lv_node, _ = nearest_lv(lv_end)
            if lv_node is None:
                logging.error("LV node not found for service‑line endpoint.")
                continue
            svc_len = np.sum(np.hypot(np.diff([p[0] for p in pts]),
                                    np.diff([p[1] for p in pts])))
            R = svc_len/1000 * conductorResistance
            X = svc_len/1000 * (conductorReactance if conductorReactance
                                else 0.1*conductorResistance)
            G.add_edge(meterNode, lv_node,
                    weight=svc_len, resistance=R, reactance=X,
                    is_service=True)          
        else:            
            lv_node, dist = nearest_lv(m_xy)
            if lv_node is None:
                logging.error("No LV line to snap meter %d", idx)
                continue
            R = dist/1000 * conductorResistance
            X = dist/1000 * (conductorReactance if conductorReactance
                            else 0.1*conductorResistance)
            G.add_edge(meterNode, lv_node,
                    weight=dist, resistance=R, reactance=X,
                    is_service=False)         
        meterNodes.append(meterNode)


        # เพิ่ม Transformer node
        transformerLocationTuple = tuple(transformerLocation)
        if transformerLocationTuple in node_mapping:
            transformerNode = node_mapping[transformerLocationTuple]
        else:
            transformerNode = node_id
            node_mapping[transformerLocationTuple] = transformerNode
            coord_mapping[transformerNode] = transformerLocationTuple
            G.add_node(transformerNode)
            G.nodes[transformerNode]['load_A'] = 0.0
            G.nodes[transformerNode]['load_B'] = 0.0
            G.nodes[transformerNode]['load_C'] = 0.0
            node_id += 1
            
            # เชื่อม transformer กับโหนด LV ที่ใกล้ที่สุด
            if lv_nodes:
                # ใช้ numpy vectorization เพื่อหาจุดที่ใกล้ที่สุดเร็วขึ้น
                lv_coords_array = np.array([coord_mapping[n] for n in lv_nodes])
                tx_loc_array = np.array(transformerLocation)
                
                # คำนวณระยะทางทั้งหมดพร้อมกัน
                distances = np.sqrt(np.sum((lv_coords_array - tx_loc_array)**2, axis=1))
                min_index = np.argmin(distances)
                closest_node = list(lv_nodes)[min_index]
                min_dist = distances[min_index]
                
                resistance = min_dist / 1000 * conductorResistance
                reactance = (min_dist / 1000 * conductorReactance) if conductorReactance is not None else 0.1 * resistance
                G.add_edge(transformerNode, closest_node, weight=min_dist, resistance=resistance, reactance=reactance)
            else:
                logging.warning("No LV lines to connect the transformer.")
    
    
    
    # เพิ่มค่าเริ่มต้นให้กับทุกโหนดที่อาจจะไม่มีข้อมูลโหลด
    for n in G.nodes:
        for ph in ['A','B','C']:
            if f'load_{ph}' not in G.nodes[n]:
                G.nodes[n][f'load_{ph}'] = 0.0
    
    logging.info(f"LV network with loads built successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return (G,transformerNode,[node_mapping[tuple(m)] for m in meterLocations],node_mapping, coord_mapping)

########################################
# NEW: BUILD LV NETWORK USING LINE ATTRIBUTE
########################################

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

########################################
# 3. NETWORK-BASED LOAD CENTER
########################################
def calculateNetworkLoadCenter(meterLocations, phase_loads, lvLines, mvLines, conductorResistance,
                              conductorReactance=None, lvData=None, svcLines=None):
    """
    ปรับปรุงฟังก์ชัน calculateNetworkLoadCenter โดยใช้อัลกอริทึมที่เร็วขึ้น
    """
    logging.info("Calculating network load center...")
    if len(meterLocations) == 0:
        logging.warning("No meter locations; returning [0, 0].")
        return np.array([0, 0], dtype=float)
    
    # สร้างกราฟเครือข่าย
    G, tNode, mNodes, node_mapping, coord_mapping = buildLVNetworkWithLoads(
        lvLines, mvLines, meterLocations, meterLocations[0], phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=lvData is not None,
        lvData=lvData,
        svcLines=svcLines
    )
    
    # คำนวณโหลดรวมของแต่ละมิเตอร์
    total_loads = phase_loads['A'] + phase_loads['B'] + phase_loads['C']
    
    # สร้าง array สำหรับเก็บผลรวมระยะทาง
    sum_distances = np.zeros(len(G.nodes))
    node_id_to_index = {nid: idx for idx, nid in enumerate(G.nodes)}
    
    # ตรวจว่าเป็นโครงข่ายแบบต้นไม้หรือไม่ (ช่วยเลือกอัลกอริทึมที่เหมาะสม)
    is_tree_like = len(G.edges()) <= len(G.nodes()) + 5  # ให้ tolerance นิดหน่อย
    
    # กรองมิเตอร์ที่ไม่มีโหลด
    valid_meters = [(i, meter_node) for i, meter_node in enumerate(mNodes) if total_loads[i] > 0]
    
    # สร้าง distance cache
    distance_cache = {}
    
    # ฟังก์ชันคำนวณระยะทางแบบเร็ว
    def fast_distance_calc(graph, source, weight='weight'):
        # กรณีกราฟเป็นแบบต้นไม้หรือใกล้เคียง
        if is_tree_like:
            # สร้าง spanning tree
            T = nx.dfs_tree(graph, source=source)
            
            # เริ่มต้นด้วยค่าระยะทางเป็นอนันต์
            distances = {node: float('inf') for node in graph.nodes()}
            distances[source] = 0.0
            
            # ใช้ BFS คำนวณระยะทาง
            queue = [source]
            visited = {source}
            
            while queue:
                current = queue.pop(0)
                
                for neighbor in T.successors(current):
                    if neighbor not in visited:
                        # คำนวณระยะทางสะสม
                        edge_weight = graph[current][neighbor].get(weight, 1.0)
                        distances[neighbor] = distances[current] + edge_weight
                        
                        # เพิ่มเข้า queue
                        queue.append(neighbor)
                        visited.add(neighbor)
            
            # ตรวจสอบโหนดที่ไม่สามารถเข้าถึงได้
            unreachable = [node for node in graph.nodes() if distances[node] == float('inf')]
            if unreachable:
                # ใช้ Dijkstra เฉพาะกับโหนดที่เข้าไม่ถึง
                dij_distances = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
                for node in unreachable:
                    if node in dij_distances:
                        distances[node] = dij_distances[node]
            
            return distances
        else:
            # ใช้ Dijkstra ถ้าไม่เป็นแบบต้นไม้ (เพื่อความถูกต้อง)
            return nx.single_source_dijkstra_path_length(graph, source, weight=weight)
    
    # คำนวณระยะทาง
    for i, meter_node in tqdm(valid_meters, desc="Calculating distances"):
        load = total_loads[i]
        
        try:
            # ใช้ cache ถ้ามี
            if meter_node in distance_cache:
                distances = distance_cache[meter_node]
            else:
                # คำนวณระยะทางด้วยอัลกอริทึมที่เหมาะสม
                distances = fast_distance_calc(G, meter_node, weight='weight')
                # เก็บใน cache
                distance_cache[meter_node] = distances
                
            # คำนวณผลรวมระยะทาง
            for n_id, dist in distances.items():
                if n_id in node_id_to_index:
                    sum_distances[node_id_to_index[n_id]] += load * dist
                
        except nx.NetworkXError:
            logging.error(f"No path for meter_node={meter_node}, skipping.")
            continue
    
    # หาโหนดที่มีผลรวมระยะทางต่ำสุด
    best_node_idx = np.argmin(sum_distances)
    best_node_id = list(G.nodes)[best_node_idx]
    bestCoord = np.array(coord_mapping[best_node_id])
    
    logging.info(f"Best network load center: {bestCoord}")
    return bestCoord


########################################
# 4. POWER FLOW & LOSS CALCULATION
########################################

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
            S_A = P_A * (powerFactor + 1j*tan_phi)
            S_B = P_B * (powerFactor + 1j*tan_phi)
            S_C = P_C * (powerFactor + 1j*tan_phi)
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
########################################
# 5. OBJECTIVES & CONSTRAINTS
########################################

def objectiveFunction(transformerLocation, meterLocations, phase_loads, initialVoltage,
                     conductorResistance, lvLines, powerFactor, load_center_only=False, 
                     conductorReactance=None, lvData=None, svcLines=None):
    logging.debug(f"Evaluating objective function at location {transformerLocation}...")
    # Build the network using the updated function
    G, tNode, mNodes, nm, cm = buildLVNetworkWithLoads(
        lvLines, [], meterLocations, transformerLocation, phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=lvData is not None,
        lvData=lvData,
        svcLines=svcLines
    )
    # Use the iterative unbalanced powerflow calculation
    node_voltages, power_flow, totalPowerLoss = calculateUnbalancedPowerFlow(
        G, tNode, mNodes, powerFactor, initialVoltage, max_iter=10, tol=1e-3
    )
    # Compute total voltage drop over meter nodes (difference in magnitude from initialVoltage)
    totalVoltageDrop = 0.0
    for node in mNodes:
        for ph in ['A','B','C']:
            totalVoltageDrop += (initialVoltage - abs(node_voltages[node][ph]))
    
    # Calculate network load center as before (using your original function)
    netCenter = calculateNetworkLoadCenter(
        meterLocations,
        phase_loads,
        lvLines,
        [],
        conductorResistance,
        conductorReactance=conductorReactance,
        lvData=lvData,
        svcLines=svcLines
    )
    distanceFromNetworkCenter = np.linalg.norm(transformerLocation - netCenter)
    
    if load_center_only:
        voltage_drop_weight = 1.0
        power_loss_weight = 0.1
        load_center_weight = 5000.0
    else:
        voltage_drop_weight = 2.0
        power_loss_weight = 0.2
        load_center_weight = 1000.0
    score = (voltage_drop_weight * totalVoltageDrop) + (power_loss_weight * totalPowerLoss) + (load_center_weight * distanceFromNetworkCenter)
    logging.debug(f"Objective function value = {score:.4f}")
    return score

def voltageConstraint(transformerLocation, meterLocations, phase_loads, initialVoltage,
                     conductorResistance, lvLines, powerFactor, conductorReactance=None, 
                     lvData=None, svcLines=None):
    logging.debug(f"Checking voltage constraint at location={transformerLocation}...")
    G, tNode, mNodes, nm, cm = buildLVNetworkWithLoads(
        lvLines, [], meterLocations, transformerLocation, phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=lvData is not None,
        lvData=lvData,
        svcLines=svcLines
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

########################################
# 6. ADDITIONAL UTILS
########################################

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

########################################
# 6B. NEW UTILS FOR JUNCTION CONSTRAINT
########################################

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

########################################
# 7. SPLITTING POINT & PARTITIONING
########################################

# Modified findSplittingPoint() now accepts candidate_index and returns candidate_edges


def findSplittingPoint(G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, candidate_index=0):
    """
    ปรับปรุงฟังก์ชัน findSplittingPoint เพื่อเพิ่มประสิทธิภาพ:
    1. ใช้ memoization สำหรับการคำนวณ cumulative loads
    2. ใช้ set แทน list ในการเก็บข้อมูลและตรวจสอบการมีอยู่
    3. คำนวณและจัดเก็บข้อมูลอย่างมีประสิทธิภาพ
    """
    global _EDGE_DF_CACHE
    logging.info("Finding splitting point by load balance difference on edges...")
    
    # สร้าง spanning tree
    T = nx.dfs_tree(G, source=transformerNode)
    
    # รวบรวมโหลดสำหรับแต่ละโหนด
    node_loads = {n: (G.nodes[n].get('load_A', 0) +
                      G.nodes[n].get('load_B', 0) +
                      G.nodes[n].get('load_C', 0))
                  for n in G.nodes()}
    
    # ใช้ dictionary เพื่อเก็บค่าโหลดสะสม (memoization)
    cumulative_loads = {}
    
    def computeCumulativeLoad(node):
        # ถ้าเคยคำนวณแล้ว ส่งคืนค่าที่เก็บไว้
        if node in cumulative_loads:
            return cumulative_loads[node]
            
        # คำนวณค่าใหม่
        load = node_loads[node]
        for child in T.successors(node):
            load += computeCumulativeLoad(child)
        
        # บันทึกค่าลงใน dictionary
        cumulative_loads[node] = load
        return load
    
    # คำนวณโหลดทั้งหมด
    total_load = computeCumulativeLoad(transformerNode)

    # คำนวณจำนวนมิเตอร์ในแต่ละ sub-tree
    meter_counts = {}
    meter_set = set(meterNodes)  # แปลงเป็น set เพื่อการค้นหาที่เร็วขึ้น
    
    def cum_meter(n):
        # ตรวจสอบว่าเคยคำนวณแล้วหรือไม่
        if n in meter_counts:
            return meter_counts[n]
            
        # ตรวจสอบว่าเป็นมิเตอร์หรือไม่ โดยใช้ set
        cnt = 1 if n in meter_set else 0
        for c in T.successors(n):
            cnt += cum_meter(c)
        
        # บันทึกค่า
        meter_counts[n] = cnt
        return cnt
    
    # คำนวณจำนวนมิเตอร์ทั้งหมด
    total_meters = cum_meter(transformerNode)
    
    # รวบรวม edges และความแตกต่างของโหลด
    edge_diffs = []
    
    # สร้าง set ของมิเตอร์โนดเพื่อการตรวจสอบที่เร็วขึ้น
    meter_node_set = set(meterNodes)
    
    for edge in T.edges():
        n1, n2 = edge
        # ข้ามเส้นเชื่อมที่เชื่อมกับมิเตอร์
        if n1 in meter_node_set or n2 in meter_node_set:
            continue
        
        # คำนวณความแตกต่างของโหลด
        load_child_side = cumulative_loads[n2]
        load_parent_side = total_load - load_child_side
        diff = abs(load_child_side - load_parent_side)
        edge_diffs.append((edge, diff))
    
    # เรียงลำดับตามความแตกต่าง
    edge_diffs.sort(key=lambda x: x[1])
    
    # กำหนดเกณฑ์ขั้นต่ำของมิเตอร์
    min_meters = max(1, int(0.20 * total_meters))
    chosen_idx = None
    
    # หา edge ที่ผ่านเกณฑ์
    for idx, (edge, diff) in enumerate(edge_diffs):
        meters_child = meter_counts[edge[1]]
        meters_parent = total_meters - meters_child
        if meters_child >= min_meters and meters_parent >= min_meters:
            chosen_idx = idx
            break
    
    # ถ้าไม่มี edge ที่ผ่านเกณฑ์ ใช้ index 0
    if chosen_idx is None:
        chosen_idx = 0

    # ใช้ candidate_index หรือ chosen_idx
    candidate_index = chosen_idx if candidate_index == 0 else candidate_index
    
    
    # 1) เตรียม lists ทั้งพิกัดและโหลด
    edges_list   = []
    diffs_list   = []
    n1x_list     = []
    n1y_list     = []
    n2x_list     = []
    n2y_list     = []
    loads_parent = []
    loads_child  = []

    # 2) เติมข้อมูลในลูปเดียว
    for (edge, diff) in edge_diffs:
        n1, n2 = edge

        # เก็บ edge กับ ΔLoad
        edges_list.append(edge)
        diffs_list.append(diff)

        # เก็บพิกัดเดิม
        x1, y1 = coord_mapping.get(n1, (np.nan, np.nan))
        x2, y2 = coord_mapping.get(n2, (np.nan, np.nan))
        n1x_list.append(x1)
        n1y_list.append(y1)
        n2x_list.append(x2)
        n2y_list.append(y2)

        # คำนวณโหลดฝั่ง child & parent
        child_load  = cumulative_loads[n2]
        parent_load = total_load - child_load
        loads_child.append(child_load)
        loads_parent.append(parent_load)

    # 3) สร้าง DataFrame พร้อมทั้ง 6 คอลัมน์พิกัด + 2 คอลัมน์โหลด
    edge_diffs_df = pd.DataFrame({
        'Edge'     : edges_list,
        'Edge_Diff': diffs_list,
        'N1_X'     : n1x_list,
        'N1_Y'     : n1y_list,
        'N2_X'     : n2x_list,
        'N2_Y'     : n2y_list,
        'Load_G1'  : loads_parent,
        'Load_G2'  : loads_child,
    })

    # 4) สร้าง splitting_index
    edge_diffs_df.reset_index(inplace=True)
    edge_diffs_df.rename(columns={'index': 'splitting_index'}, inplace=True)
    _EDGE_DF_CACHE = edge_diffs_df
    
    # บันทึกไฟล์ CSV
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    csv_path = f"{folder_path}/edgediff.csv"
    edge_diffs_df.to_csv(csv_path, index=True, index_label="splitting_index")
    logging.info(f"Splitting edges info saved to CSV: {csv_path}. Found {len(edge_diffs)} edges.")

    # ถ้าไม่มี edge ให้คืนค่า None
    if edge_diffs_df.empty:
        _EDGE_DF_CACHE = None
        logging.warning("No valid edges for splitting.")
        return None, None, None, []

    # ตรวจสอบความถูกต้องของ candidate_index
    if candidate_index < 0 or candidate_index >= len(edge_diffs):
        logging.error("Candidate index out of range, using index 0 instead.")
        candidate_index = 0

    # หา edge ที่ดีที่สุดและความแตกต่าง
    best_edge = edge_diffs[candidate_index][0]
    best_edge_diff = edge_diffs[candidate_index][1]
    u, v = best_edge

    # คำนวณจุดแยกเป็นจุดกึ่งกลาง
    u_coord = np.array(coord_mapping[u])
    v_coord = np.array(coord_mapping[v])
    splitting_point_coord = (u_coord + v_coord) / 2

    logging.info(f"Splitting edge chosen (candidate={candidate_index}): {best_edge}, diff={best_edge_diff:.2f}")

    # สร้าง shapefile ของ edges
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    shp_path = f"{folder_path}/edgediff.shp"
    w = shapefile.Writer(shp_path, shapeType=shapefile.POLYLINE)
    w.field("FID","N")
    w.field("Index", "N")
    w.field("Edge_Diff", "F", decimal=2)

    for i, row in edge_diffs_df.iterrows():
        # สร้างเส้นสำหรับแต่ละ edge
        w.line([[
            [row['N1_X'], row['N1_Y']],
            [row['N2_X'], row['N2_Y']]
        ]])
        w.record(i, row['Edge_Diff'])

    w.close()
    logging.info(f"Shapefile of edges exported: {shp_path}")

    # คืนค่าผลลัพธ์
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

########################################
# 8. HELPER: GET BOUNDING BOX OF A SET OF POINTS
########################################

def get_bounding_box(points):
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
    return [(min_x, max_x), (min_y, max_y)]

########################################
# 9. OPTIMIZATION FUNCTIONS
########################################

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

def optimizeGroup(meterLocations, phase_loads, initialTransformerLocation,
                  lvLines, mvLines, initialVoltage, conductorResistance,
                  powerFactor, epsilon_junction=1.0, conductorReactance=None, 
                  lvData=None, svcLines=None):
    """
    Optimizing group-level transformer location using only existing LV line nodes,
    explicitly skipping meter nodes. The function name/signature is unchanged, but
    the internal approach is discrete enumeration over conductor nodes.
    """
    logging.info("Optimizing group-level transformer location on existing LV nodes (skipping meter nodes)...")

    if len(meterLocations) == 0:
        logging.info("No meters in this group => skip optimization.")
        return None

    # Build a temporary graph for this group's meters + lines
    G_temp, tNode_temp, mNodes_temp, node_mapping_temp, coord_mapping_temp = buildLVNetworkWithLoads(
        lvLines, mvLines,
        meterLocations,
        initialTransformerLocation,
        phase_loads,
        conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=lvData is not None,
        lvData=lvData,
        svcLines=svcLines
    )

    # We'll iterate over all nodes, but skip the meter nodes so we stay on the actual line nodes
    best_coord = None
    best_score = float('inf')

    for node_id in G_temp.nodes():
        # Skip nodes if they don't have a coordinate or if they are meter nodes
        if node_id not in coord_mapping_temp:
            continue
        if node_id in mNodes_temp:
            # This node was added specifically for a meter location
            continue

        node_xy = np.array(coord_mapping_temp[node_id], dtype=float)

        # Evaluate your existing objective function at this node
        score = objectiveFunction(
            node_xy,
            meterLocations,
            phase_loads,
            initialVoltage,
            conductorResistance,
            lvLines,
            powerFactor,
            load_center_only=True,
            conductorReactance=conductorReactance,
            lvData=lvData,
            svcLines=svcLines
        )

        if score < best_score:
            best_score = score
            best_coord = node_xy

    if best_coord is None:
        logging.warning("No valid line node found for discrete LV optimization. Returning initial location.")
        return initialTransformerLocation

    logging.info(f"Discrete node-based optimization picked node => {best_coord}, score={best_score:.2f}")
    return best_coord

########################################
# 10. ITERATIVE DISTANCE CHECK
########################################



########################################
# 11. EXPORT & PLOTTING
########################################

def exportPointsToShapefile(point_coords, shapefile_path, attributes_list=None):
    logging.info(f"Exporting {len(point_coords)} point(s) to shapefile: {shapefile_path}...")
    w = shapefile.Writer(shapefile_path, shapeType=shapefile.POINT)
    w.autoBalance = 1
    # Define fields for name and coordinates.
    w.field('FID','N')
    w.field('Name', 'C')
    w.field('X', 'F', decimal=8)
    w.field('Y', 'F', decimal=8)
    for idx, (x, y) in enumerate(point_coords):
        w.point(x, y)
        if attributes_list and idx < len(attributes_list):
            record = attributes_list[idx]
            record.setdefault('FID', idx)
            record.setdefault('Name', f'Point_{idx+1}')
            # Ensure that coordinate fields are included.
            record.setdefault('X', x)
            record.setdefault('Y', y)
            w.record(
                record['FID'],
                record['Name'], 
                record['X'], 
                record['Y'])
        else:
            w.record(f"Point{idx+1}", x, y)
    w.close()
    logging.info(f"Shapefile saved successfully: {shapefile_path}.shp")

def exportResultDFtoShapefile(result_df, shapefile_path="output_meters.shp"):
    """
    Exports each row of result_df as a point feature in a shapefile.
    Requires result_df to have at least 'Meter X' and 'Meter Y' columns.
    """
    if 'FID' not in result_df.columns:
        result_df['FID'] = range(len(result_df))
        
    w = shapefile.Writer(shapefile_path, shapeType=shapefile.POINT)
    w.autoBalance = 1  # Ensures geometry and attributes sync

    # Define fields. Adjust field names as needed for your data.
    w.field('FID','N')
    w.field('Peano', 'C', size=20)
    w.field('VoltA', 'F', decimal=2)
    w.field('VoltB', 'F', decimal=2)
    w.field('VoltC', 'F', decimal=2)
    w.field('Group', 'C', size=10)
    w.field('Phases', 'C', size=5)
    w.field('MeterX', 'F', decimal=8)
    w.field('MeterY', 'F', decimal=8)

    for idx, row in result_df.iterrows():
        # Make sure these columns exist in result_df
        x_coord = float(row["Meter X"])
        y_coord = float(row["Meter Y"])

        w.point(x_coord, y_coord)

        w.record(
            row.get('FID'),
            row.get('Peano Meter', ''),             # e.g. the meter's Peano
            row.get('Final Voltage A (V)', 0),      # final voltage A
            row.get('Final Voltage B (V)', 0),      # final voltage B
            row.get('Final Voltage C (V)', 0),      # final voltage C
            row.get('Group', ''),                   # group label
            row.get('Phases', ''),                  # e.g. 'A', 'B', 'C'
            x_coord,
            y_coord
        )

    w.close()
    print(f"Shapefile saved: {shapefile_path} (plus .shx, .dbf).")
    

def plotResults(lvLinesX, lvLinesY, mvLinesX, mvLinesY, eserviceLinesX, eserviceLinesY,
                meterLocations, initialTransformerLocation, optimizedTransformerLocation,
                group1_indices, group2_indices, splitting_point_coords=None, coord_mapping=None,
                optimizedTransformerLocationGroup1=None, optimizedTransformerLocationGroup2=None,
                transformer_losses=None, phases=None, result_df=None,G=None):
    logging.info("Plotting final results...")
    plot_path = "output_plot.png"

    
    # Delete existing file to prevent conflicts
    if os.path.exists(plot_path):
        os.remove(plot_path)
    
    # สร้างหน้าต่าง Toplevel
    plot_window = tk.Toplevel()
    plot_window.title("Meter Locations, Lines, and Transformers")
    plot_window.geometry("1000x800")
    
    # สร้าง frame สำหรับใส่ canvas และ toolbar
    frame = tk.Frame(plot_window)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # สร้าง figure และ embed ใน Tkinter window
    fig = plt.Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # วาดข้อมูลทั้งหมด (เหมือนเดิมทุกประการ)
    for x, y in zip(lvLinesX, lvLinesY):
        ax.plot(x, y, color='lime', linewidth=1, linestyle='--', label='LV Line' if 'LV Line' not in ax.get_legend_handles_labels()[1] else "")
    for x, y in zip(mvLinesX, mvLinesY):
        ax.plot(x, y, color='maroon', linewidth=1, linestyle='-.', label='MV Line' if 'MV Line' not in ax.get_legend_handles_labels()[1] else "")
    for x, y in zip(eserviceLinesX, eserviceLinesY):
        ax.plot(x, y, 'm-', linewidth=2, label='Eservice Line to TR' if 'Eservice Line' not in ax.get_legend_handles_labels()[1] else "")
    if len(group1_indices) > 0:
        ax.plot(meterLocations[group1_indices, 0], meterLocations[group1_indices, 1], 'b.', markersize=10, label='Group 1 Meters')
    if len(group2_indices) > 0:
        ax.plot(meterLocations[group2_indices, 0], meterLocations[group2_indices, 1], 'r.', markersize=10, label='Group 2 Meters')
    if result_df is not None:
        for i in range(len(meterLocations)):
            x = meterLocations[i, 0]
            y = meterLocations[i, 1]
            connected_phases = phases[i].upper().strip()
            voltage_text = ''
            for ph in ['A','B','C']:
                if ph in connected_phases:
                    colname = f'Final Voltage {ph} (V)'
                    if colname in result_df.columns:
                        vval = result_df.iloc[i][colname]
                        voltage_text += f'{ph}:{vval:.1f}V\n'
                    else:
                        voltage_text += f'{ph}:N/A\n'
            if voltage_text.strip():
                ax.text(x, y, voltage_text.strip(), fontsize=6, color='black',
                         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
    ax.plot(initialTransformerLocation[0], initialTransformerLocation[1], 'ko', markersize=10, label='Initial Transformer')
    ax.text(initialTransformerLocation[0], initialTransformerLocation[1], ' Initial Transformer',
             verticalalignment='top', horizontalalignment='left', fontsize=10, fontweight='bold')
    if splitting_point_coords is not None:
        ax.plot(splitting_point_coords[0], splitting_point_coords[1], 'ys', markersize=12, label='Splitting Point')
        ax.text(splitting_point_coords[0], splitting_point_coords[1], ' Splitting Point',
                 verticalalignment='bottom', horizontalalignment='left', fontsize=10, fontweight='bold')
    if optimizedTransformerLocationGroup1 is not None:
        ax.plot(optimizedTransformerLocationGroup1[0], optimizedTransformerLocationGroup1[1], 'b*', markersize=15, label='Group 1 Transformer')
        ax.text(optimizedTransformerLocationGroup1[0], optimizedTransformerLocationGroup1[1], ' Group 1 Transformer',
                 verticalalignment='bottom', horizontalalignment='right', fontsize=10, fontweight='bold', color='blue')
    if optimizedTransformerLocationGroup2 is not None:
        ax.plot(optimizedTransformerLocationGroup2[0], optimizedTransformerLocationGroup2[1], 'r*', markersize=15, label='Group 2 Transformer')
        ax.text(optimizedTransformerLocationGroup2[0], optimizedTransformerLocationGroup2[1], ' Group 2 Transformer',
                 verticalalignment='bottom', horizontalalignment='right', fontsize=10, fontweight='bold', color='red')
    if transformer_losses is not None and coord_mapping is not None:
        for tx, loss in transformer_losses.items():
            if tx == 'Group 1 Transformer' and optimizedTransformerLocationGroup1 is not None:
                x, y = optimizedTransformerLocationGroup1
            elif tx == 'Group 2 Transformer' and optimizedTransformerLocationGroup2 is not None:
                x, y = optimizedTransformerLocationGroup2
            else:
                continue
            ax.text(x, y, f"{tx}\nLoss: {loss/1000:.2f} kW", fontsize=8, color='black',
                     bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none', pad=1))
            
    if G is not None and coord_mapping is not None:
        svc_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_service')]
        for u, v in svc_edges:
            x1, y1 = coord_mapping[u]
            x2, y2 = coord_mapping[v]
            ax.plot([x1, x2], [y1, y2], color='purple', linewidth=2,
                    label='Eserviceline Meter to LVLines' if 'Service‑Line'
                    not in ax.get_legend_handles_labels()[1] else "")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    ax.set_title('Meter Locations, Lines, and Transformers')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # ตั้งค่าเริ่มต้นและบันทึกไว้
    ax.autoscale(True)
    # บันทึกขอบเขตเริ่มต้นไว้
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()
    
    fig.tight_layout()
    canvas.draw()
    
    # ฟังก์ชันสำหรับรีเซ็ตมุมมอง
    def reset_view():
        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)
        ax.set_aspect('equal')
        canvas.draw()
    
    # เพิ่ม NavigationToolbar2Tk แบบกำหนดเอง
    class CustomToolbar(NavigationToolbar2Tk):
        def home(self, *args):
            # เมื่อกดปุ่ม Home ให้เรียกใช้ฟังก์ชัน reset_view
            reset_view()
    
    toolbar = CustomToolbar(canvas, frame)
    toolbar.update()
    
      
    # เพิ่มความสามารถ pan/zoom ด้วย mouse (เสริมจากปุ่มใน toolbar)
    pan_enabled = False
    pan_start = None
    
    def on_button_press(event):
        nonlocal pan_enabled, pan_start
        if event.xdata is not None and event.ydata is not None:
            if event.button == 1:  # คลิกซ้าย - เริ่ม pan
                pan_enabled = True
                pan_start = (event.xdata, event.ydata)
            elif event.button == 3:  # คลิกขวา - ยกเลิก pan
                pan_enabled = False
                pan_start = None
    
    def on_button_release(event):
        nonlocal pan_enabled, pan_start
        if event.button == 1:  # คลิกซ้าย
            pan_enabled = False
            pan_start = None
    
    def on_mouse_move(event):
        nonlocal pan_enabled, pan_start
        if pan_enabled and pan_start is not None:
            if event.xdata is not None and event.ydata is not None:
                # คำนวณระยะทางที่เลื่อน
                dx = event.xdata - pan_start[0]
                dy = event.ydata - pan_start[1]
                
                # ปรับขอบเขตใหม่
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.set_xlim(xmin - dx, xmax - dx)
                ax.set_ylim(ymin - dy, ymax - dy)
                
                canvas.draw()
                pan_start = (event.xdata, event.ydata)
    
    def on_scroll(event):
        if event.xdata is not None and event.ydata is not None:
            # ทิศทางของ scroll
            if event.button == 'up':
                scale_factor = 0.9  # zoom in
            else:
                scale_factor = 1.1  # zoom out
                
            # คำนวณขอบเขตใหม่สำหรับ zoom
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xmin = event.xdata - (event.xdata - xmin) * scale_factor
            xmax = event.xdata + (xmax - event.xdata) * scale_factor
            ymin = event.ydata - (event.ydata - ymin) * scale_factor
            ymax = event.ydata + (ymax - event.ydata) * scale_factor
            
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            canvas.draw()
    
    # ลงทะเบียน event handlers
    canvas.mpl_connect('button_press_event', on_button_press)
    canvas.mpl_connect('button_release_event', on_button_release)
    canvas.mpl_connect('motion_notify_event', on_mouse_move)
    canvas.mpl_connect('scroll_event', on_scroll)
    
    # คงการบันทึกไฟล์เหมือนเดิม
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    plot_path = f"{folder_path}/output_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_path}")

########################################
# 12. TRANSFORMER SIZING FUNCTION
########################################

def growthRate(g2_load, annual_growth=0.04, years=4):
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

########################################
# 13. NODE LABELS & GRAPH PLOTTING
########################################

def addNodeLabels(G, splitting_point_node, best_edge_diff):
    logging.info("Adding labels to nodes in graph G.")
    for node in G.nodes():
        if node == splitting_point_node:
            G.nodes[node]['label'] = f"Node {node}\nEdge Diff: {best_edge_diff:.2f}"
        else:
            G.nodes[node]['label'] = f"Node {node}"
    return G

def plotGraphWithLabels(G, coord_mapping, best_edge_diff=None, best_edge=None):
    logging.info("Plotting graph with node labels...")
    node_colors = ['red' if node in best_edge else 'lightblue' for node in G.nodes()]
    pos = {node: coord_mapping[node] for node in G.nodes() if node in coord_mapping}
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(10,8))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=50, edge_color='green')
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    if best_edge is not None:
        edge_pos = [pos[best_edge[0]], pos[best_edge[1]]]
        plt.plot([edge_pos[0][0], edge_pos[1][0]], [edge_pos[0][1], edge_pos[1][1]], color='red', linewidth=2)
    plt.title("Graph with Node Labels")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.close()
    
    logging.info("Graph plotted with labels.")

########################################
# 14. RE-EXECUTION FUNCTION (POST-PROCESS)
########################################

def rerun_process(candidate_index, G, transformerNode, meterNodes, node_mapping, coord_mapping,
                  meterLocations, phase_loads, lvLines, mvLines, filteredEserviceLines,
                  initialTransformerLocation, powerFactor, initialVoltage, conductorResistance,
                  peano, phases, conductorReactance=None, lvData=None, svcLines=None):
    logging.info(f"Re-executing post-process steps with new splitting candidate index: {candidate_index}")
    best_edge, sp_coord, sp_edge_diff, candidate_edges = findSplittingPoint(G, transformerNode, meterNodes,
                                                                           coord_mapping, powerFactor, initialVoltage,
                                                                           candidate_index)
    if best_edge is None:
        logging.error("No valid splitting edge found with candidate index {}.".format(candidate_index))
        return None
    
    group1_nodes, group2_nodes = partitionNetworkAtPoint(G, transformerNode, meterNodes, best_edge)
    voltages, branch_curr, group1_nodes, group2_nodes = performForwardBackwardSweepAndDivideLoads(
        G, transformerNode, meterNodes, coord_mapping, powerFactor, initialVoltage, best_edge
    )
    nodeToIndex = {mn: i for i, mn in enumerate(meterNodes)}
    group1_meter_nodes = [n for n in group1_nodes if n in nodeToIndex]
    group2_meter_nodes = [n for n in group2_nodes if n in nodeToIndex]
    g1_idx = [nodeToIndex[n] for n in group1_meter_nodes]
    g2_idx = [nodeToIndex[n] for n in group2_meter_nodes]
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
        'Distance to Transformer (m)': dist_arr,
        'Load A (kW)': phase_loads['A'],
        'Load B (kW)': phase_loads['B'],
        'Load C (kW)': phase_loads['C'],
        'Group': ['Group 1' if n in group1_nodes else 'Group 2' for n in meterNodes],
        'Phases': phases,
        'Final Voltage A (V)': [np.nan]*len(meterNodes),
        'Final Voltage B (V)': [np.nan]*len(meterNodes),
        'Final Voltage C (V)': [np.nan]*len(meterNodes)
    })
    result_df["Meter X"] = meterLocations[:, 0]
    result_df["Meter Y"] = meterLocations[:, 1]

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
    optimizedTransformerLocationGroup1 = optimizeGroup(
        group1_meterLocs,
        group1_phase_loads,
        calculateNetworkLoadCenter(
            group1_meterLocs, 
            group1_phase_loads, 
            lvLines, 
            mvLines, 
            conductorResistance,
            conductorReactance=conductorReactance,
            lvData=lvData,
            svcLines=svcLines
        ),
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
        calculateNetworkLoadCenter(
            group2_meterLocs, 
            group2_phase_loads, 
            lvLines, 
            mvLines, 
            conductorResistance,
            conductorReactance=conductorReactance,
            lvData=lvData,
            svcLines=svcLines
        ),
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
   
    if optimizedTransformerLocationGroup1 is not None:
        G_g1, tNode_g1, mNodes_g1, nm_g1, cm_g1 = buildLVNetworkWithLoads(
            lvLines, mvLines,
            group1_meterLocs,
            optimizedTransformerLocationGroup1,
            group1_phase_loads,
            conductorResistance,
            conductorReactance=conductorReactance,
            use_shape_length=lvData is not None,
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
            use_shape_length=lvData is not None,
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

    line_loss_g2_kW = total_power_loss_g2 / 1000.0
    tx_loss_g2_kW = get_transformer_losses(g2_load)
    total_system_loss_g2 = line_loss_g2_kW + tx_loss_g2_kW

    # Select Transformer size for each group (using your document)
    rating_g1 = Lossdocument(g1_load)
    rating_g2 = Lossdocument(g2_load)

    # Forecast Future Load
    future_g1_load = growthRate(g1_load, annual_growth=0.04, years=4)
    future_g2_load = growthRate(g2_load, annual_growth=0.04, years=4)

    logging.info('############ Result ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW, Future growthrate(4yr@4%)={future_g1_load:.2f} kW, Chosen TX1={rating_g1} kVA")
    logging.info(f"Group 2 => Load={g2_load:.2f} kW, Future growthrate(4yr@4%)={future_g2_load:.2f} kW, Chosen TX2={rating_g2} kVA")
    logging.info('############ Loss Report ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW | LineLoss={line_loss_g1_kW:.2f} kW | TxLoss={tx_loss_g1_kW:.2f} kW => TOTAL={total_system_loss_g1:.2f} kW")
    logging.info(f"Group 2 => Load={g2_load:.2f} kW | LineLoss={line_loss_g2_kW:.2f} kW | TxLoss={tx_loss_g2_kW:.2f} kW => TOTAL={total_system_loss_g2:.2f} kW")
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    csv_path = f"{folder_path}/optimized_transformer_locations.csv"
    result_df.to_csv(csv_path, index=False)
    logging.info(f"Result CSV saved: {csv_path}")

    # Export to shapefile (the new part)
    folder_path = './testpy'
    ensure_folder_exists(folder_path)
    exportResultDFtoShapefile(result_df, f"{folder_path}/result_meters.shp")
    
    
    # Export Splitting Point, Transformer Group Point to Shapefile #
    point_coords = []
    attributes_list = []
    if sp_coord is not None:
        # Convert sp_coord to tuple before exporting
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
        
    # Plot results and export shapefile (using same functions as before)
    g1_plot_indices = np.array(g1_idx, dtype=int)
    g2_plot_indices = np.array(g2_idx, dtype=int)
    plotResults(
        [l['X'] for l in lvLines], [l['Y'] for l in lvLines],
        [l['X'] for l in mvLines], [l['Y'] for l in mvLines],
        [l['X'] for l in filteredEserviceLines],
        [l['Y'] for l in filteredEserviceLines],
        meterLocations,
        initialTransformerLocation,
        optimizedTransformerLocationGroup1,  # For demonstration
        g1_plot_indices,
        g2_plot_indices,
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
    plotGraphWithLabels(G, coord_mapping, best_edge_diff=sp_edge_diff, best_edge=best_edge)
    logging.info("Re-execution finished.")
    
    result = {
        'best_edge': best_edge,            # edge ที่ผู้ใช้เลือก
        'group1': {
            'idx'        : np.array(g1_idx, dtype=int),
            'meter_locs' : group1_meterLocs,
            'phase_loads': group1_phase_loads,
            'tx_loc'     : optimizedTransformerLocationGroup1
        },
        'group2': {
            'idx'        : np.array(g2_idx, dtype=int),
            'meter_locs' : group2_meterLocs,
            'phase_loads': group2_phase_loads,
            'tx_loc'     : optimizedTransformerLocationGroup2
        }
    }
    
    return result
##############################################################################
# SECOND‑LEVEL SPLIT + OPTIMISE  (re‑use existing primitives)
##############################################################################

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
    """Dialog loop to test alternative splitting indices."""
    global latest_split_result, reopt_btn

    temp_root = tk.Tk();  temp_root.withdraw()

    last_result = latest_split_result  # start with current split

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
                            meterLocations, phase_loads,
                            lvLines, mvLines, filteredEserviceLines,
                            initialTransformerLocation,
                            powerFactor, initialVoltage,
                            conductorResistance,
                            peano, phases,
                            conductorReactance, lvData, svcLines)
        if tmp is not None:
            last_result = tmp

    # after loop ends
    if last_result is not None:
        latest_split_result = last_result
        if reopt_btn is not None:
            reopt_btn.config(state=tk.NORMAL)

    temp_root.destroy()


########################################
# 15. MAIN + TKINTER UI
########################################

def main():
    global lvLines, mvLines, initialVoltage, conductorResistance, powerFactor
    global conductorReactance, lvData, svcLines, latest_split_result, reopt_btn
    
    logging.info("Program started.")
    try:
        meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases = extractMeterData(meterData)
        lvLinesX, lvLinesY, lvLines = extractLineData(lvData)
        mvLinesX, mvLinesY, mvLines = extractLineData(mvData)
        svcLinesX, svcLinesY, svcLines = extractLineDataWithAttributes(eserviceData, 'SUBTYPECOD', 5)
        numbercond_priority = [3, 2, 1]
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return
        
    try:
        transformerRecords = transformerData.records()
        transformerFields = [f[0] for f in transformerData.fields[1:]]
        transformer_df = pd.DataFrame(transformerRecords, columns=transformerFields)
        if 'OPSA_TRS_3' in transformerFields:
            transformerCapacity_kVA = transformer_df['OPSA_TRS_3'].values[0]
        else:
            raise KeyError("Field 'OPSA_TRS_3' not found in transformer shapefile.")
        powerFactor = 0.875
        transformerCapacity = transformerCapacity_kVA * powerFactor
    except Exception as e:
        logging.error(e)
        return
    
    conductorResistance = 0.77009703
    conductorReactance = 0.3497764 
    initialVoltage = 230
    try:
        eserviceLinesX, eserviceLinesY, filteredEserviceLines = extractLineDataWithAttributes(eserviceData, 'SUBTYPECOD', 2)
        if not filteredEserviceLines:
            logging.warning("No Eservice lines with SUBTYPECOD=2 found.")
    except Exception as e:
        logging.error(f"Error processing Eservice lines: {e}")
        return
    
    try:
        t_shapes = transformerData.shapes()
        if not t_shapes:
            logging.error("Transformer shapefile has no shapes.")
            return
        initialTransformerLocation = np.array([t_shapes[0].points[0][0], t_shapes[0].points[0][1]])
        logging.info(f"Initial transformer location => {initialTransformerLocation}")
    except Exception as e:
        logging.error(f"Error extracting transformer location: {e}")
        return
    
    G_init, tNode_init, mNodes_init, nm_init, cm_init = buildLVNetworkWithLoads(
        lvLines, filteredEserviceLines, meterLocations, initialTransformerLocation, phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=True,
        lvData=lvData,
        svcLines=svcLines
    )
    
    if not nx.is_connected(G_init):
        logging.warning("Initial network not fully connected.")
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
    
    G, transformerNode, meterNodes, node_mapping, coord_mapping = buildLVNetworkWithLoads(
        lvLines, filteredEserviceLines, meterLocations, optimizedTransformerLocation, phase_loads, conductorResistance,
        conductorReactance=conductorReactance,
        use_shape_length=True,
        lvData=lvData,
        svcLines=svcLines
    )
    
    if not nx.is_connected(G):
        logging.warning("Post-optimization network not fully connected.")
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
    
    optimizedTransformerLocationGroup1 = optimizeGroup(
        group1_meterLocs,
        group1_phase_loads,
        calculateNetworkLoadCenter(
            group1_meterLocs, 
            group1_phase_loads, 
            lvLines, 
            mvLines, 
            conductorResistance,
            conductorReactance=conductorReactance,
            lvData=lvData,
            svcLines=svcLines
        ),
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
        calculateNetworkLoadCenter(
            group2_meterLocs, 
            group2_phase_loads, 
            lvLines, 
            mvLines, 
            conductorResistance,
            conductorReactance=conductorReactance,
            lvData=lvData,
            svcLines=svcLines
        ),
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

    line_loss_g2_kW = total_power_loss_g2 / 1000.0
    tx_loss_g2_kW = get_transformer_losses(g2_load)
    total_system_loss_g2 = line_loss_g2_kW + tx_loss_g2_kW

    # Select Transformer size for each group (using your document)
    rating_g1 = Lossdocument(g1_load)
    rating_g2 = Lossdocument(g2_load)

    # Forecast Future Load
    future_g1_load = growthRate(g1_load, annual_growth=0.04, years=4)
    future_g2_load = growthRate(g2_load, annual_growth=0.04, years=4)

    logging.info('############ Result ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW, Future growthrate(4yr@4%)={future_g1_load:.2f} kW, Chosen TX1={rating_g1} kVA")
    logging.info(f"Group 2 => Load={g2_load:.2f} kW, Future growthrate(4yr@4%)={future_g2_load:.2f} kW, Chosen TX2={rating_g2} kVA")
    logging.info('############ Loss Report ############')
    logging.info(f"Group 1 => Load={g1_load:.2f} kW | LineLoss={line_loss_g1_kW:.2f} kW | TxLoss={tx_loss_g1_kW:.2f} kW => TOTAL={total_system_loss_g1:.2f} kW")
    logging.info(f"Group 2 => Load={g2_load:.2f} kW | LineLoss={line_loss_g2_kW:.2f} kW | TxLoss={tx_loss_g2_kW:.2f} kW => TOTAL={total_system_loss_g2:.2f} kW")

    # Export Splitting Point, Transformer Group Point to Shapefile #
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
    
    plotResults(
        [l['X'] for l in lvLines], [l['Y'] for l in lvLines],
        [l['X'] for l in mvLines], [l['Y'] for l in mvLines],
        [l['X'] for l in filteredEserviceLines],
        [l['Y'] for l in filteredEserviceLines],
        meterLocations,
        initialTransformerLocation,
        optimizedTransformerLocation_LV,  # For demonstration
        g1_plot_indices,
        g2_plot_indices,
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
    if reopt_btn is not None:
        reopt_btn.config(state=tk.NORMAL)

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
def createGUI():
    """Create the GUI with 'Run Process' and 'Import ShapeFile' buttons."""
    global root, transformerFileName, reopt_btn, lvData, conductorReactance
    
    root = tk.Tk()
    root.title("Transformer Optimization GUI")

    # Create a frame at the top for the buttons
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    # Add the "Import ShapeFile" Button
    import_btn = tk.Button(top_frame, text="Import ShapeFile", command=lambda: loadShapefiles(root))
    import_btn.pack(side=tk.LEFT, padx=5)

    # Add the "Run Process" Button
    run_btn = tk.Button(top_frame, text="Run Process", command=runProcess)
    run_btn.pack(side=tk.LEFT, padx=5)
    
    # Add the "Re-optimize" Button
    reopt_btn = tk.Button(top_frame, text="Re-optimize subgroup",
                          state=tk.DISABLED,      # disabled until we have data
                          command=runReoptimize)
    reopt_btn.pack(side=tk.LEFT, padx=5)

    # Create a frame to hold the log text widget and scrollbar
    text_frame = tk.Frame(root)
    text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create the scrollbar
    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create the text widget for logging
    log_text = tk.Text(text_frame, wrap="word", height=20, yscrollcommand=scrollbar.set)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure the scrollbar to scroll the text widget
    scrollbar.config(command=log_text.yview)
    
    if 'transformerFileName' in globals() and transformerFileName:
        base_name = os.path.splitext(os.path.basename(transformerFileName))[0]
        folder_path = './testpy'
        ensure_folder_exists(folder_path)
        log_filename = os.path.join(folder_path, f"Optimization_{base_name}_log.txt")
    else:
        folder_path = './testpy'
        ensure_folder_exists(folder_path)
        log_filename = f"{folder_path}/Optimization_Transformer_log.txt"
    
    # Configure the logger to log messages to the text widget
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    text_handler = TextHandler(log_text)
    text_handler.setFormatter(fmt)
    text_handler.setLevel(logging.INFO)
    text_handler.addFilter(SummaryFilter())  # Filtering logs
    logger.addHandler(text_handler)

    # Configure a FileHandler to also log to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))

    root.mainloop()

#############################################################################################
# Other Functions                                                                            
#############################################################################################

# Create folder 
def ensure_folder_exists(folder_path):
    """
    สร้างโฟลเดอร์ถ้ายังไม่มีอยู่
    :param folder_path: path ของโฟลเดอร์ที่จะสร้าง
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Create a new folder: {folder_path}")
    else:
        print(f"folder {folder_path} already exists.")


#############################################################################################

if __name__ == "__main__":
    createGUI()