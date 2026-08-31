# โครงสร้างโปรเจกต์ — GIS OTD (PEA Transformer Network Analysis)

เว็บแอป Flask สำหรับวิเคราะห์เครือข่ายหม้อแปลง/สายจำหน่ายของ กฟภ. (PEA) — โหลดข้อมูลจาก GIS server ภายใน, คำนวณโหลด/แรงดัน/เฟส, และแสดงผลบนแผนที่ ArcGIS JS

## Entry point: สองระบบคู่ขนาน

| | `run.py` → `app/` | `app.py` |
|---|---|---|
| สถานะ | **ใช้งานจริง** (`python run.py`) | Legacy monolith เดิม |
| โครงสร้าง | Flask Blueprint package | ไฟล์เดียวรวมทุก route |
| ควรแก้ที่ไหน | ที่นี่ สำหรับงานใหม่ทั้งหมด | หลีกเลี่ยง เว้นแต่จำเป็นต้องให้ `app.py` รันแยกได้ (เช่นแก้ import ให้ตรงกับโมดูลรุ่นล่าสุด) |

`run.py` เรียก `app/create_app()` แล้ว `app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)` — `use_reloader=False` เพราะ auto-reloader เดิมจะ watch ทั้ง tree รวมถึง `feature_shareload/FEASIBLE/` ที่ shareload job เขียนไฟล์ผลลัพธ์ระหว่างรัน ทำให้แอป restart กลางงาน ต้อง restart process มือเองหลังแก้โค้ด

## `app/` — Flask Blueprint package (ใช้งานจริง)

```
app/
├── __init__.py       # create_app(), register blueprints
├── config.py         # Config, REGION_MAPPING ฯลฯ
├── database.py       # get_db_connection()
├── routes/
│   ├── auth.py              # login/logout
│   ├── projects.py          # โปรเจกต์วิเคราะห์เครือข่าย LV (pea_no_projects) — main_pipeline / run_pipeline_for_facilityid
│   ├── phase_optimizer.py   # ฟีเจอร์ Phase Optimizer
│   ├── shareload.py         # ฟีเจอร์ Shareload / Transfer Optimizer
│   └── stats.py             # หน้าสถิติ/แดชบอร์ด
├── services/
│   ├── project_service.py         # เรียก InputJsonApi / optimized_transformer_group_* / processNew_no_gui
│   ├── phase_optimizer_service.py # เรียก feature_PhaseOptimizer/run_web.py
│   ├── stats_service.py
│   └── admin_service.py
└── utils/
    ├── decorators.py   # @login_required
    └── helpers.py      # get_user_region() ฯลฯ
```

## `templates/` — Jinja templates (20 ไฟล์)

รวมหน้าแผนที่ ArcGIS JS API (`*map.html`, เช่น `phase_optimizer_map.html`, `peaNoProjectmap.html`, `testmap.html`) ที่โหลด GeoJSON layer จาก route แบบ `/…/output/<project_id>/<file>.geojson` แล้ว render ด้วย `esri/layers/GeoJSONLayer`

## โมดูล business logic ที่ root — ดิบ (dev) vs API

โปรเจกต์นี้มี pattern ที่ทำให้สคริปต์ดิบ (มักมี tkinter GUI, ตั้งชื่อไฟล์แบบมีเลขรุ่นต่อท้าย) ถูก "โปรโมท" เป็นโมดูลที่ Flask import จริงทีละรุ่น — รายละเอียดขั้นตอน + checklist ทำซ้ำอยู่ใน [`docs/API_CONVERSION_GUIDE.md`](API_CONVERSION_GUIDE.md) โมดูลที่ **active อยู่ตอนนี้**:

| ไฟล์ | บทบาท |
|---|---|
| `InputJsonApi.py` | ชื่อคงที่ — ดึง FACILITYID → BalanceLoad → join table → NetworkLV JSON ต่อ `project_id` |
| `optimized_transformer_group_310869.py` | รุ่นล่าสุด (import ตรงชื่อไฟล์) — จัดกลุ่มหม้อแปลง/optimize ตำแหน่ง/สมดุลเฟส, entry point คือ `main_pipeline(project_id, facility_id, sp_index=0)` |
| `processNew_no_gui.py` | ประมวลผลจาก shapefile ที่ผู้ใช้อัปโหลด (`run_process_from_project_folder`) |

ไฟล์รุ่นก่อนหน้าที่ถูกแทนที่แล้วอยู่ใน `notUse/` (เช่น `optimized_transformer_group_200269.py`, `optimized_transformer_group_300669.py`) — **ห้าม import จากที่นั่น**

## `feature_PhaseOptimizer/`, `feature_shareload/` — ฟีเจอร์แบบ standalone

ทั้งสองมี core script + `run_web.py` wrapper แต่เชื่อมกับ Flask คนละแบบ:

- **feature_PhaseOptimizer** — `run_web.py` ถูก **import ใน-process** เข้า `app/services/phase_optimizer_service.py` โดยตรง (`run_phase_optimizer(...)`)
- **feature_shareload** — `run_web.py` ถูกเรียกผ่าน **`subprocess.run(...)`** จาก `app/routes/shareload.py` (รันแยก process เพราะงาน CPU-heavy/ใช้เวลานาน)

ทั้งคู่เขียนผลลัพธ์เป็น GeoJSON + `results.json` ให้หน้าแผนที่ดึงไปแสดง เหมือน pattern ของ `pea_no_projects/`

## `pea_no_projects/` — runtime data directory convention

ทุกโมดูล API (Convention A/B ใน API_CONVERSION_GUIDE) ใช้โครงสร้างนี้ร่วมกัน เพื่อแยกข้อมูลของแต่ละโปรเจกต์ไม่ให้ทับกันเวลามีหลาย user ใช้พร้อมกัน:

```
pea_no_projects/
├── input/<project_id>/
│   ├── <project_id>_NetworkLV<facility_id>.json           # จาก InputJsonApi.run_once_with_facilityid
│   └── <project_id>_NetworkLV<facility_id>_with_MV.json   # จาก InputJsonApi.run_pipeline_for_facilityid (merge MV)
└── output/<project_id>/
    ├── results.json                    # ผลล่าสุด
    ├── results_cache/<sp_index>.json   # ผลลัพธ์ต่อ splitting-candidate index (เลือกจากแผนที่)
    ├── lv_lines.geojson / mv_lines.geojson / meter_groups.geojson / feature_groups.geojson
    ├── edge_diffs.json                 # สำหรับ index selector บนแผนที่
    └── downloads/                      # shapefile/CSV ให้ดาวน์โหลด
```

## โฟลเดอร์ runtime อื่นๆ

| โฟลเดอร์ | ใช้ทำอะไร |
|---|---|
| `notUse/` | คลังไฟล์รุ่นเก่าที่ถูกแทนที่แล้ว (ห้าม import) |
| `uploads/` | shapefile ที่ผู้ใช้อัปโหลด (โฟลเดอร์วิเคราะห์แบบ upload-shapefile ไม่ผ่าน GIS) |
| `output/` | ผลลัพธ์ของ flow แบบ upload-shapefile (คู่กับ `pea_no_projects/output/` ของ flow แบบ FACILITYID) |
| `db/` | schema SQL (`db.sql`, `create_phase_optimizer_projects_table.sql`) |
| `static/js/` | JS ฝั่ง client ที่ใช้ร่วมหลายหน้า |
| `logs/` | log ไฟล์รายรัน (`setup_run_file_logger`, `setup_logging`) |
| `venv/` | virtualenv — **หมายเหตุ**: `pyvenv.cfg` ปัจจุบันชี้ไป `C:\Program Files\Python310` ซึ่งไม่มีอยู่จริงในเครื่องนี้ ใช้ `py -3` (system launcher) แทนเวลารันสคริปต์เดี่ยวๆ นอก IDE |

## ดูเพิ่มเติม

- [`docs/API_CONVERSION_GUIDE.md`](API_CONVERSION_GUIDE.md) — ขั้นตอนแปลงสคริปต์ดิบเป็น API + checklist + ตารางฟีเจอร์ปัจจุบัน
