# คู่มือ: ผนวกโค้ดรุ่นใหม่เข้าระบบ (สำหรับ Claude และทีมพัฒนา)

คู่มือนี้คือ "ขั้นตอนมาตรฐาน" ที่สรุปจากการทำมาแล้วหลายรอบ (PhaseOptimizer 01/03/16/17 ก.ย. 2026, Runopendss 16/18 ก.ย. 2026, TransferOptimizer 16 ก.ย. 2026, `INPUTJSON2`/`optimized_transformer_group`)
ใช้คู่กับ [`API_CONVERSION_GUIDE.md`](API_CONVERSION_GUIDE.md) (รายละเอียด convention A/B และกรณีศึกษาแต่ละรอบ)

**หลักคิด:** โค้ดใหม่ที่ผู้ใช้ให้มาคือ business logic ล่าสุด ส่วนระบบเรามี "ของที่ใช้งานได้ดีอยู่แล้ว" (แผนที่, สถานะการแก้ไข, การเชื่อม GIS/DB, การรันหลายผู้ใช้) — งานคือ **เอา logic ใหม่ + คงของเดิมไว้ทั้งหมด** ไม่ใช่เอาไฟล์ใหม่ไปแทนที่ทั้งไฟล์

---

## 1. ผู้ใช้ทำแค่ 2 อย่าง

1. วางไฟล์โค้ดใหม่ในโฟลเดอร์ [`incoming/`](../incoming/README.md) ตั้งชื่อ `<ชื่อเดิม>_<วันเดือนปี ค.ศ.>.py` (เช่น `PhaseOptimizer_17092026.py`)
2. บอก Claude ว่า "มีโค้ดใหม่ใน incoming/" (และบอกถ้ามีเงื่อนไขพิเศษ / ต้องการ push ขึ้น git)

โฟลเดอร์ `incoming/` ไม่ถูก commit (ดู `.gitignore`) — เมื่อผนวกเสร็จ ไฟล์ดิบถูกย้ายไป `incoming/_processed/` เก็บไว้เทียบย้อนหลังในเครื่อง

## 2. ไฟล์ดิบ → ปลายทาง

| ชื่อไฟล์ดิบ (ขึ้นต้นด้วย) | ปลายทาง | Convention | จุด import ที่ต้องแก้ | ลำดับ |
|---|---|---|---|---|
| `INPUTJSON2_*`, `INPUT_FACILITY_*` | เขียนทับ `InputJsonApi.py` (ชื่อคงที่) | A | ไม่ต้องแก้ | 1 |
| `Runopendss_All*` | `feature_shareload/` | B | `PhaseOptimizer_*.py`, `TransferOptimizer_*.py` (บรรทัด `from Runopendss_All... import`) + comment ที่อ้างชื่อ | 2 |
| `TransferOptimizer_*` | `feature_shareload/` | B | `feature_shareload/run_web.py` (importlib), `PhaseOptimizer_*.py` (importlib) | 3 |
| `PhaseOptimizer_*` | `feature_PhaseOptimizer/` | B | `feature_PhaseOptimizer/run_web.py` (`from PhaseOptimizer_... import` + comment) | 4 |
| `optimized_transformer_group_*` | root | B | `app/routes/projects.py`, `app/services/project_service.py`, `app.py` | 1 |

- ถ้ามาหลายไฟล์พร้อมกัน ให้ผนวกตาม **ลำดับ dependency** (เลขลำดับด้านบน — ตัวล่างพึ่งตัวบน)
- ชื่อที่ไม่อยู่ในตาราง → **ถามผู้ใช้** ว่าเป็นของฟีเจอร์ไหน อย่าเดา
- Convention B: รุ่นเก่าที่ถูกแทนที่ต้องย้ายไป `notUse/` ทุกครั้ง

## 3. ขั้นตอนผนวก (ทำตามลำดับ)

1. **ดูว่ามีอะไรเข้ามา** — `ls incoming/` และ `git status`
   - ⚠ ผู้ใช้บางครั้ง **วางทับชื่อเดิม** (ไม่ได้เปลี่ยนเลขรุ่น): `git status` จะขึ้น `M` (modified) และอาจมีไฟล์สำรอง `*old.py` โผล่มา → diff กับ `git show HEAD:<path>` ไม่ใช่กับไฟล์ตามชื่อรุ่นก่อนหน้า
2. **Diff** ไฟล์ใหม่กับรุ่นที่ใช้งานอยู่ (normalize บรรทัดก่อน เพราะไฟล์ผู้ใช้มักเป็น CRLF):
   ```bash
   diff -u <(sed 's/\r$//' <รุ่นปัจจุบัน>) <(sed 's/\r$//' incoming/<ไฟล์ใหม่>)
   ```
3. **แบ่ง hunk เป็น 2 กลุ่ม** (สำคัญที่สุด):
   - **scaffolding/ฟีเจอร์เดิมที่หาย → คืน** (ดูข้อ 4 และ 5) — ไฟล์ดิบมัก fork มาจาก snapshot เก่า จึง**หายซ้ำแทบทุกรอบ** ให้ตั้งสมมติฐานว่าจะหายก่อนอ่าน diff
   - **logic ใหม่จริง → คงไว้ทั้งหมด** ห้ามแก้พฤติกรรมเอง; ถ้าไม่แน่ใจว่าเป็นการตัดสินใจใหม่หรือของหล่นจากรุ่นเก่า (เช่นเกณฑ์ตัวเลข) → **ถามผู้ใช้**
4. **ผนวก** ในไฟล์ปลายทางชื่อรุ่นใหม่ (คัดลอกจาก `incoming/` เข้าโฟลเดอร์ปลายทาง แล้วแก้ที่นั่น)
5. **เช็ค signature ที่เปลี่ยน** — ถ้าฟังก์ชันที่ไฟล์อื่น import เปลี่ยน return/พารามิเตอร์ (เคยเจอ `build_bfs_order` 4→5 ค่า) ให้ grep หา **ทุก caller ที่เรียกจริง** ไม่ใช่แค่ที่ import
6. **แก้ import ทุกจุด** ตามตารางข้อ 2 แล้ว `grep -rn "<ชื่อรุ่นเก่า>"` ให้ไม่เหลือ (ยกเว้น `notUse/` และประวัติใน docs)
7. **ย้ายรุ่นเก่าไป `notUse/`** และย้ายไฟล์ดิบไป `incoming/_processed/`
8. **ฟีเจอร์ใหม่ต้องโผล่บนหน้าเว็บด้วย** — ถ้าโค้ดใหม่เพิ่มขั้นตอน/ผลลัพธ์แบบใหม่ (เช่น Design Check) ต้องต่อเข้า `results.json`, `steps_applied`, geojson/แผนที่, ป้ายสถานะ (ข้อ 5.C–5.D) ไม่ใช่แค่ให้สคริปต์รันผ่าน (เคยพลาด: Design Check ทำงานแต่หน้าเว็บไม่แสดง)
9. **ตรวจสอบ** (ข้อ 6) → **อัปเดตเอกสาร** (ข้อ 8) → **commit/push** (ข้อ 7)

## 4. Scaffolding ที่ต้องมีเสมอ (หายซ้ำทุกรอบ)

ตรวจอัตโนมัติด้วย `py -3 tools/check_integration.py` (ดูข้อ 6) — รายการนี้คือสิ่งที่สคริปต์เช็ค

| ไฟล์ | ต้องมี |
|---|---|
| `PhaseOptimizer_*` | sys.path bootstrap (`_THIS_DIR/_PROJECT_ROOT/_SHARELOAD_DIR`) · `region` param + `self.region` · `self.error` (+ set ใน `run()` ทั้ง 2 จุด: หา JSON ไม่เจอ / baseline ไม่ converge) · `_ensure_json()` เรียก `InputJsonApi.run_once_with_facilityid(fac, project_id=f"phaseopt_{fac}")` + `set_gis_region` + ใช้ `out_path` จากผลลัพธ์ · `--region` + `region=args.region` · โหลด TransferOptimizer ด้วย `importlib.util.spec_from_file_location` ชี้รุ่นปัจจุบัน · `DEFAULT_MAX_IMBALANCE_PCT = 25.0` · หัวไฟล์/usage ใช้ชื่อไฟล์ตัวเอง |
| `TransferOptimizer_*` | sys.path bootstrap · `region` param (คู่กับพารามิเตอร์ใหม่ ไม่ใช่แทนที่) · `_ensure_json()` แบบเดียวกัน (`project_id=f"shareload_{fac}"`) · `--region` · import `Runopendss_All<รุ่นปัจจุบัน>` · `feasible_top` เรียงตามคุณภาพก่อนตัด top-5 (ไม่งั้น dropdown ไม่มีวิธีที่ดีที่สุด) |
| `Runopendss_All*` | `solve_with_opendss()` save/restore `os.getcwd()` รอบ `Compile` · `select_load_point_indices()` รับ `SUBTYPECODE` **1 และ 3** |
| `optimized_transformer_group_*` | `_API_PROJECT_ID`, `_API_SP_INDEX`, `main_pipeline()`, `_write_headless_geojson()` · ห้าม hardcode `candidate_index=0` · ห้ามใช้ `args.facility_id` ในโหมด headless |
| `InputJsonApi.py` | `_BASE_DIR` · `GISConfig`/`set_gis_region` · `project_id` param · timeout 120 วิ |

สิ่งที่ **ไม่มีจริงในโปรเจกต์** แต่ไฟล์ดิบชอบเรียก: โมดูล `INPUT_FACILITY` (ต้องเปลี่ยนเป็น `InputJsonApi`), `from TransferOptimizer import` (ต้องเปลี่ยนเป็น importlib ชี้รุ่นปัจจุบัน)

## 5. สัญญาที่ต้องรักษา (ฟีเจอร์พื้นฐานที่ห้ามหายเมื่อผนวกโค้ดใหม่)

### A. ข้อมูลมิเตอร์ต้องครบ
- ทุกมิเตอร์จริงต้องเข้าเครือข่าย/วงจร OpenDSS/แผนที่: point + `SUBTYPECODE ∈ {1, 3}` + ไม่ใช่หม้อแปลง (`TAG` มี "XF") + มี `PEANO` หรือ `PEAMETER`
- ตรวจ: จำนวน `meters=` ในล็อก `[Step 2]` ต้องเท่ากับจำนวน point ที่มี PEANO ใน JSON ดิบ (ลบหม้อแปลง) และเท่ากับจำนวนฟีเจอร์ใน `meter_groups.geojson`
- เฟสของมิเตอร์ใช้ **เฟสจริงที่จุดต่อบน backbone** (`_meter_current_pd`) ไม่ใช่ tag ดิบ (มิเตอร์ tag `ABC` อาจต่อจริงแค่เฟส A)

### B. ผลลัพธ์ที่ `feature_PhaseOptimizer/run_web.py` เขียน (หน้าเว็บ/ดาวน์โหลดพึ่งอยู่)
- ไฟล์: `results.json`, `lv_lines.geojson`, `meter_groups.geojson`, `feature_groups.geojson`, `upgrade_lines.geojson`, `downloads/phase_opt_<FAC>.xlsx|png`
- `meter_groups` properties: `peano, kw, phase_before, phase_after, moved, phase_move`
- `feature_groups` properties: `name, group` โดย `group ∈ transformer | low_v_fixed | low_v_remaining`
- `upgrade_lines` properties: `name, group` โดย `group ∈ conductor_upgrade | phase_addition` — วาด **ทุก segment ใน `upgraded_edges`** ไม่ใช่แค่ `.edge` (เคยพลาด: เส้นบนเว็บสั้นกว่าจริง)
- `results.json`: `baseline, steps_applied, design_check, phase_transfer, conductor_upgrade, phase_addition, final, final_has_problem, improved, moved_meters`
- ออบเจ็กต์ `LVOptimizer` ต้องยังมี: `net, raw, final_raw, final_result, baseline, error, region, phase_moves, phase_moves_cu, phase_moves2, applied_upgrade(.edge/.upgraded_edges/.from_size/.to_size/.n_affected), applied_phase_add(.upgraded_edges/.meter_moves/.result/.design_fixed), design_violations(_after), design_summary(_after)` และ `_has_problem()`
- ฟังก์ชันที่ `run_web.py` import: `LVOptimizer, SimResult, save_excel_report, draw_map, _meter_current_pd, _build_meter_inventory, PD_TO_PHASE`

### C. หน้าแผนที่ `templates/phase_optimizer_map.html` (ArcGIS JS 4.29)
- จุดมิเตอร์สีตาม `phase_after` (A แดง `#ef4444` / B เหลือง `#eab308` / C น้ำเงิน `#3b82f6`)
- วงแหวนมิเตอร์ที่ย้ายเฟส สีตาม **เฟสปลายทาง** เท่านั้น (ไม่แยกตามคู่ก่อน→หลัง)
- แถบ checkbox เปิด/ปิดเลเยอร์เหนือแผนที่ (`#layerToggles`): ระบบจำหน่าย, มิเตอร์, ช่วงเพิ่มขนาดสาย, ช่วงเพิ่มเฟสสาย, ย้ายไปเฟส A/B/C, แรงดันตกแก้แล้ว/ยังอยู่, หม้อแปลง — เลเยอร์ที่ไม่มีข้อมูลไม่แสดง; **Legend สร้างจากเลเยอร์ที่มีข้อมูลเท่านั้น** (ไม่งั้นมีกล่องขาวว่าง)
- เส้น "ช่วงเพิ่มขนาดสาย/เพิ่มเฟสสาย" ความหนา 2
- ทุก `view.goTo(...)` ต้องมี `{ animate: false }` + `.catch(() => {})` และทุก promise chain ต้องมี `.catch` — **บั๊ก ArcGIS 4.29** `goTo` แบบมี animation โยน `Cannot read properties of undefined (reading 'animation')` แล้วเลเยอร์อื่น (มิเตอร์) ไม่โหลดต่อ
- ห้ามใช้ Arcade `IIf()` (ไม่มีฟังก์ชันนี้ — จุดมิเตอร์หาย) ใช้ renderer แบบ field-based
- ห้ามใช้ `LayerList` widget ของ ArcGIS (ไอคอนสวิตช์ไม่ขึ้น และซ้อนทับ legend)

### D. สถานะการแก้ไข (ป้าย "ขั้นตอนที่ใช้" + ป้ายสถานะบนแผนที่)
- ป้าย "ย้ายเฟสมิเตอร์" **นับรวมทุกจังหวะ**: ย้ายปกติ, หลังเพิ่มขนาดสาย, กระจายเฟสใน Phase Addition, หลังเพิ่มเฟสสาย (Design Check ทำ Phase Addition ก่อนย้ายเฟส ทำให้รอบย้ายเฟสปกติเป็น 0 ได้)
- ป้าย Design Check (ก่อน → หลัง กลุ่ม), ป้ายเพิ่มขนาดสาย (จำนวนช่วง), ป้ายเพิ่มเฟสสาย
- ป้ายสถานะรวม: แก้ไขสำเร็จ / ยังพบปัญหาหลังปรับปรุง / **ปรับปรุงตามหลักออกแบบแล้ว** (ไฟฟ้าผ่านแต่มีการปรับ) / ระบบปกติตั้งแต่แรก
- รายการ "มิเตอร์ที่ย้ายเฟส" (`moved_meters`) แสดงครบ
- ขั้นตอน/ผลลัพธ์ชนิดใหม่จากโค้ดใหม่ → **เพิ่มเข้า `steps_applied` + ป้ายเสมอ**

### E. ค่าที่ผู้ใช้ตัดสินใจไว้แล้ว (ห้ามเปลี่ยนตามไฟล์ดิบ)
- Phase Imbalance ผ่านเกณฑ์ ≤ **25%** (ไฟล์ดิบมักย้อนกลับเป็น 15/20)
- ปุ่ม "แบ่งกลุ่มย่อยเพิ่ม" ใน `/peaNoMap/<id>` แบ่งเพิ่มพอดี 1 ระดับ (ไม่ถามเกณฑ์ kVA)
- หน้า `/transformer_stats`: คอลัมน์สรุปปัญหา / แนวทางแก้ไข / หม้อแปลงใกล้เคียง อยู่ถัดจาก FACILITY ID, หัวตารางตรึง, scrollbar แนวนอนลอย
- ถ้าไฟล์ดิบประกาศค่า business ใหม่ที่ดูตั้งใจ (ไม่ใช่ของหล่น) → **ถามผู้ใช้ก่อน**

## 6. ตรวจสอบก่อนถือว่าเสร็จ

```bash
# 1) syntax + import chain
py -3 -c "import ast; ast.parse(open('<ไฟล์>.py', encoding='utf-8').read())"
py -3 -c "from app.services.phase_optimizer_service import PhaseOptimizerService"
py -3 -c "from app.routes.shareload import shareload_bp"

# 2) scaffolding + สัญญา + ไม่มี reference ค้าง (exit 1 ถ้า FAIL)
py -3 tools/check_integration.py

# 3) รันจริง 1 เคส (ต้องระบุ region — default NE1; facility ทดสอบ 54-008953 และ 59-018889 อยู่ NE2)
cd feature_PhaseOptimizer
py -3 run_web.py 54-008953 D:/tmp_out NE2
```
- เทียบ `meters=` ในล็อกกับจำนวนใน JSON ดิบ (`pea_no_projects/input/phaseopt_<FAC>/...json`) และ `meter_groups.geojson` (ข้อ 5.A); ดู `steps_applied` ใน `results.json`
- ถ้าเข้า GIS ไม่ได้ (`getaddrinfo failed`): ใช้ JSON ที่แคชไว้ โดย monkeypatch `run_web.LVOptimizer._ensure_json = lambda self: Path('<path json>')` แล้วเรียก `run_web.run_phase_optimizer(...)`
- **หน้าแผนที่** (แก้ template): ตรวจ syntax JS — ดึง `<script>` ออก แทน `{{ ... }}` ด้วยค่าสมมติ แล้ว `node --check`; ทดสอบจริงด้วย Edge headless: เสิร์ฟ geojson + หน้าทดสอบด้วย `py -3 -m http.server` แล้ว `msedge --headless --disable-gpu --virtual-time-budget=60000 --dump-dom <url>` (ฝัง `setTimeout` เขียนผลตรวจลง DOM; virtual time ไม่นิ่ง ให้รอนานพอ)
- import smoke test ไม่รันผ่าน GIS/DB — ต้องรันเคสจริงเสมอก่อนสรุปว่าเสร็จ และแจ้งผู้ใช้ตรงๆ ถ้าทดสอบบางส่วนไม่ได้

## 7. Git

- commit เมื่อผู้ใช้สั่ง (ส่วนใหญ่ผู้ใช้จะขอ "push ด้วย") · ข้อความ commit เป็นภาษาไทย บอก "ทำไม" · ลงท้ายด้วย `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
- add ทีละไฟล์ (ไม่ใช้ `git add -A`); **ไม่ commit**: `incoming/*`, `.claude/`, `peano_list.txt`, `feature_shareload/FEASIBLE/**/transfer_*.(html|png|xlsx|_results.json)`, ไฟล์ดิบ `INPUTJSON*`/`INPUT_FACILITY_*`
- ย้ายไฟล์รุ่นเก่าด้วย `mv` แล้ว `git add` **ทั้งชื่อเก่า (ที่ถูกลบ) และชื่อใหม่ใน `notUse/`** เพื่อให้ git จับเป็น rename
- ไม่ใช้คำสั่งทำลายข้อมูล (force push, reset --hard) นอกจากผู้ใช้สั่ง

## 8. เอกสาร

- เพิ่ม "กรณีศึกษาจริง (วันที่)" ท้าย [`API_CONVERSION_GUIDE.md`](API_CONVERSION_GUIDE.md): อะไรหาย/อะไรคือ logic ใหม่/ผลตรวจจริง
- ถ้าเจอจุดเสี่ยงใหม่ที่ตรวจอัตโนมัติได้ → เพิ่มเข้า `tools/check_integration.py` และตารางข้อ 4–5 ในคู่มือนี้

## 9. บทเรียนจากอดีต (จุดที่พลาดมาแล้ว)

| อาการ | สาเหตุ | วิธีกัน |
|---|---|---|
| รันแล้วพังทันที `No module named INPUT_FACILITY` | ไฟล์ดิบเรียกโมดูลที่ไม่มีจริง | ข้อ 4 (`InputJsonApi`) |
| หาหม้อแปลงไม่เจอ ("ไม่พบ TR") | ไม่ส่ง region (default NE1) | ส่ง `region` ทุกทาง; ทดสอบด้วย `... NE2` |
| ผล Phase Optimizer เพี้ยนเป็น 20%/15% | ไฟล์ดิบ fork จาก snapshot เก่า | เช็ค `DEFAULT_MAX_IMBALANCE_PCT = 25.0` |
| fix เก่า (dropdown แชร์โหลด) หายไปกับรุ่นใหม่ | ไฟล์ดิบไม่มี fix นั้น | diff หา fix ประวัติศาสตร์ด้วย ไม่ใช่แค่ scaffolding |
| เส้นเพิ่มขนาดสายบนเว็บสั้นกว่าใน html | web อ่าน `.edge` แทน `.upgraded_edges` | ข้อ 5.B; เมื่อ field ของ dataclass เปลี่ยนความหมาย ไล่หาทุกจุดที่อ่านมัน |
| จุดมิเตอร์หายทั้งหมด | Arcade `IIf()` ไม่มีจริง | renderer field-based (5.C) |
| จุดมิเตอร์ไม่ขึ้น + console `reading 'animation'` | บั๊ก ArcGIS 4.29 `goTo` | `animate:false` + `.catch` (5.C) |
| มิเตอร์บางตัวไม่ขึ้น และโหลดหายจากวงจรจำลอง | filter รับแค่ `SUBTYPECODE=1` (มิเตอร์ 3 เฟสเป็น 3) | 5.A |
| ป้ายสถานะไม่ครบ/ "ระบบปกติ" ทั้งที่มีการแก้ | นับ `phase_moves` รอบแรกอย่างเดียว ขณะที่ Design Check เพิ่มเฟสก่อนย้ายเฟส | 5.D |
| Legend มีกล่องขาวว่าง / วิดเจ็ตทับกันกดไม่ได้ | รวมเลเยอร์ว่างใน legend; LayerList ซ้อนทับ | 5.C |
| `ValueError: too many values to unpack` | signature ของฟังก์ชันร่วมเปลี่ยน (`build_bfs_order`) | ข้อ 3.5 |
| `cwd` เพี้ยนหลังรันหลาย facility | OpenDSS `Compile` ทำ `chdir` ค้าง | ข้อ 4 (`Runopendss_All*`) |
| ไฟล์กำพร้าค้างใน root/`feature_*` | ลืมย้ายรุ่นเก่าไป `notUse/` | ข้อ 3.7; `check_integration.py` เตือน |

## 10. เมื่อไหร่ต้องถามผู้ใช้ (ไม่เดา)

- ชื่อไฟล์ไม่อยู่ในตารางข้อ 2
- ค่า business ใหม่ที่ดูตั้งใจ (เกณฑ์ตัวเลข ฯลฯ) ต่างจากที่เคยยืนยันไว้
- ผู้ใช้ให้ facility/region สำหรับทดสอบไม่พอ (ต้องรู้เขต)
- ต้องการ push หรือไม่ ถ้าผู้ใช้ไม่ได้บอก
