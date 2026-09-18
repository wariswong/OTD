# คู่มือ: แปลงสคริปต์ดิบ (dev script) ให้เป็นโมดูล API

โปรเจกต์นี้มี pattern ที่เกิดซ้ำหลายรอบแล้ว: มีการพัฒนา business logic (ดึงข้อมูล GIS, คำนวณโหลด/แรงดัน, จัดกลุ่มหม้อแปลง ฯลฯ) เป็นสคริปต์ดิบก่อน — มักมี tkinter GUI ติดมาด้วย และตั้งชื่อไฟล์แบบมีเลขรุ่นต่อท้าย (เช่น `_310869`, `_300669`) — แล้วค่อย "โปรโมท" (promote) logic ใหม่เข้าไปรวมกับโมดูลที่ Flask เรียกใช้จริงเป็น API ทีหลัง

คู่มือนี้สรุปวิธีทำซ้ำขั้นตอนนั้นให้ปลอดภัยและไม่พลาดจุดสำคัญ

## ทำไมต้องมีคู่มือนี้

- สคริปต์ดิบพัฒนาแยกจากแอปหลัก ไม่มี path/lifecycle ที่เหมาะกับเว็บแอปแบบ multi-user (เช่น เขียนไฟล์ทับ CWD ตรงๆ, ไม่รองรับ region/project_id หลายค่าพร้อมกัน)
- เวลาโปรโมทรุ่นใหม่ ถ้า diff ไม่ละเอียดพอ มักจะ "ลืม" เอา API scaffolding (ที่เคยผนวกไว้ในรุ่นก่อน) กลับเข้าไปด้วย ทำให้ของที่เคยใช้งานได้ใน production พังแบบเงียบๆ (import ผ่าน แต่ path ผิด/ตัวแปรไม่มีค่า)
- เกิดเหตุการณ์แบบนี้มาแล้วอย่างน้อย 2 รอบ (`INPUTJSON` → `InputJsonApi.py`, `optimized_transformer_group_200269` → `_300669` → `_310869`) และคาดว่าจะเกิดอีกระหว่างที่โปรแกรมยังอยู่ในช่วงพัฒนา

## สอง convention ที่มีอยู่จริงในโค้ด

### Convention A — stable-name (ชื่อไฟล์ API คงที่)

ไฟล์ดิบมีเลขรุ่นต่อท้าย แต่โมดูลที่ Flask import จริงมีชื่อคงที่ไม่เปลี่ยน — ทุกรอบที่โปรโมท ให้ **เขียนทับเนื้อหาไฟล์ชื่อคงที่** ด้วย logic ใหม่ + scaffolding เดิม โดยไม่ต้องแก้ import ที่ไหนเลย

ตัวอย่าง: `INPUTJSON2_<code>.py` (ไฟล์ดิบ) → `InputJsonApi.py` (โมดูล API ชื่อคงที่)

```python
from InputJsonApi import run_pipeline_for_facilityid   # ไม่เปลี่ยนตามรุ่น
```
ใช้ที่: `app/services/project_service.py`, `app/routes/projects.py`, `app.py`

### Convention B — versioned-import (ชื่อไฟล์ตามรุ่นคือสิ่งที่ import)

ไฟล์ที่ Flask import คือไฟล์ดิบรุ่นล่าสุดตรงๆ (ไม่มีชื่อกลาง) — ทุกรอบที่โปรโมทรุ่นใหม่ ต้อง **แก้ import ทุกจุดที่เรียกใช้**

ตัวอย่าง: `optimized_transformer_group_<code>.py`

```python
from optimized_transformer_group_310869 import main_pipeline   # ต้องแก้เลขรุ่นทุกครั้ง
```
ใช้ที่: `app/services/project_service.py`, `app/routes/projects.py`, `app.py` (จุดเดียวกับข้างบน แต่คนละบรรทัด import)

รุ่นที่ถูกแทนที่แล้วให้ย้ายไปเก็บที่ `notUse/` (ดู `notUse/optimized_transformer_group_200269.py`, `notUse/optimized_transformer_group_300669.py`) — **ถ้าลืมขั้นตอนนี้ จะมีไฟล์กำพร้าค้างอยู่ที่ root ไปเรื่อยๆ** (พบเคสจริง: `optimized_transformer_group_280469.py` ที่ไม่มีใครเรียกใช้แล้ว แต่ไม่เคยถูกย้าย)

## Checklist "API scaffolding" ที่ต้องมองหาทุกครั้ง

เวลา diff ไฟล์ดิบรุ่นใหม่กับโมดูล API ปัจจุบัน ให้ไล่หาสิ่งเหล่านี้ — ถ้าไฟล์ดิบไม่มี แปลว่าต้องผนวกกลับเข้าไป ไม่ใช่ปล่อยตามไฟล์ดิบ:

| สิ่งที่ต้องมี | ทำไม |
|---|---|
| `_BASE_DIR = os.path.dirname(os.path.abspath(__file__))` แล้ว anchor path ทั้งหมดด้วยตัวนี้ | ถ้ารันเป็น Windows Service, CWD จะไม่ใช่โฟลเดอร์โปรเจกต์ (มักเป็น `System32`) — path สัมพัทธ์ตรงๆ จะพังแบบเงียบๆ |
| Entry-point function ที่รับ `project_id` (เช่น `main_pipeline(project_id, facility_id, ...)`, `run_pipeline_for_facilityid(facility_id, project_id, region=None)`) | ให้ Flask route เรียกแบบมี state ต่อ request ได้ ไม่ใช่ global เดียวใช้ร่วมกันทุกคน |
| Path แบบ `pea_no_projects/input/{project_id}/...` และ `pea_no_projects/output/{project_id}/...` (ไม่ใช่เขียนไฟล์ทับ CWD ตรงๆ) | กัน 2 โปรเจกต์ทับไฟล์กัน เวลามีหลาย user ใช้พร้อมกัน |
| `IS_GUI` / headless branch แยกชัดเจน (`if IS_GUI: ... else: ...`) | โหมด API ต้องไม่ไปเปิดหน้าต่าง tkinter หรือ block รอ user คลิก |
| GUI class + `if __name__ == "__main__":` เดิม ให้ **comment ทิ้งไว้ ไม่ลบ** | ไฟล์ยังต้องรันแบบ standalone/ทดสอบเองได้ในเครื่อง dev |
| Region-awareness (ถ้า business logic เกี่ยวกับ GIS เขต) เช่น `GISConfig`/`set_gis_region()` | ผู้ใช้จากเขตต่างกันต้องยิง GIS server คนละตัว |
| `sp_index` + `results_cache/{sp_index}.json` (ถ้ามี "เลือก candidate อื่นจากแผนที่") | ให้หน้าแผนที่สลับดูผลลัพธ์ที่ cache ไว้ได้โดยไม่ต้อง reprocess |
| debug `print()`/`logging.info` ที่เยอะเกินจำเป็น | ตัดออกให้ log สะอาด (โค้ด dev มักมี debug log เยอะกว่าที่ควรอยู่ใน production) |

## ขั้นตอนทำจริง

1. **Diff ไฟล์ดิบรุ่นใหม่กับโมดูล API ปัจจุบัน**
   ```bash
   diff -u <โมดูล_API_เดิม>.py <ไฟล์ดิบรุ่นใหม่>.py
   ```
2. **ไล่จัดหมวดทุก hunk** เป็น 2 กลุ่ม:
   - **scaffolding หาย ต้องคืน** — เทียบกับ checklist ด้านบน มักมาพร้อม comment อธิบายเหตุผล/บั๊กที่เคยแก้ (สัญญาณสำคัญ: ถ้า API เดิมมี comment อธิบายว่าทำไมต้องทำแบบนี้ เช่น "ถ้าไม่ทำแบบนี้จะทำให้ facilityid=None เสมอ" — นั่นคือ fix ที่ต้อง preserve ไม่ใช่ noise ที่ตัดทิ้งได้)
   - **logic ใหม่จริงของผู้ใช้** — ต้องคงไว้ทั้งหมด ห้ามแก้พฤติกรรมโดยไม่ถามก่อน
3. **Merge มือ** ทีละจุดตาม checklist — แก้ไฟล์ดิบรุ่นใหม่ในที่เดิม (Convention B) หรือเขียนทับไฟล์ชื่อคงที่ (Convention A)
4. **อัปเดต import** ทุกจุดที่เรียกใช้ (เฉพาะ Convention B) — grep หาให้ครบ อย่าเชื่อแค่ไฟล์เดียว (ในโปรเจกต์นี้มักมี 3 จุด: `app/services/project_service.py`, `app/routes/projects.py`, `app.py`)
5. **ย้ายรุ่นก่อนหน้าไป `notUse/`** (เฉพาะ Convention B)
6. **Verify**:
   ```bash
   py -3 -c "import ast; ast.parse(open('<ไฟล์>.py', encoding='utf-8').read())"   # syntax
   py -3 -c "import <module_name>"                                                # import smoke test — จับ NameError/ตัวแปรหาย
   grep -rn "<ชื่อโมดูลรุ่นเก่า>" --include="*.py" .                              # ไม่มี reference ค้าง
   ```
   หมายเหตุ: import smoke test จับได้แค่ error ระดับ module scope (top-level code, decorator, default arg) ไม่ได้รันจริงผ่าน GIS/DB — ต้องทดสอบสร้างโปรเจกต์จริง 1 เคสในเว็บแอปด้วยเสมอก่อนถือว่าเสร็จ

## ตารางฟีเจอร์ปัจจุบัน (living reference — อัปเดตเพิ่มเมื่อมีฟีเจอร์ใหม่)

| ฟีเจอร์ | ไฟล์ดิบ (dev) | โมดูล API | ใครเรียกใช้ | Convention |
|---|---|---|---|---|
| BalanceLoad → NetworkLV JSON | `INPUTJSON2_<code>.py` | `InputJsonApi.py` (ชื่อคงที่) | `app/services/project_service.py`, `app/routes/projects.py`, `app.py` | A |
| Transformer group optimization | `optimized_transformer_group_<code>.py` | `optimized_transformer_group_<code>.py` (import ชื่อไฟล์ล่าสุดตรงๆ) | เช่นเดียวกัน | B |
| Phase Optimizer | `feature_PhaseOptimizer/PhaseOptimizer_<code>.py` (import ชื่อไฟล์ล่าสุดตรงๆ จาก `run_web.py`) | `feature_PhaseOptimizer/run_web.py` (import ใน-process) | `app/routes/phase_optimizer.py`, `app/services/phase_optimizer_service.py` | B (ซ้อนใน in-process wrapper) |
| Shareload / Transfer Optimizer | `feature_shareload/TransferOptimizer_<code>.py` (โหลดผ่าน `importlib.util.spec_from_file_location` เพราะชื่อไฟล์เปลี่ยนทุกรุ่น) | `feature_shareload/run_web.py` (เรียกผ่าน `subprocess.run`) | `app/routes/shareload.py` | B (ซ้อนใน subprocess wrapper) |
| Runopendss (shared DSS/JSON helpers) | `feature_shareload/Runopendss_All<code>.py` | import ตรงชื่อไฟล์ล่าสุดจาก TransferOptimizer และ PhaseOptimizer ทั้งคู่ | ทั้งสองฟีเจอร์ข้างบน | B (dependency ร่วม ต้องอัปเดตพร้อมทั้งคู่) |

รายละเอียดโครงสร้างโฟลเดอร์ทั้งหมด ดูที่ [`docs/PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)

## กรณีศึกษาจริง (2026-08-31): แปลง `INPUTJSON2_310869.py` + `optimized_transformer_group_310869.py`

- `INPUTJSON2_310869.py`: ไฟล์ดิบไม่มี `_BASE_DIR`, `GISConfig`/`set_gis_region`, `project_id` param ใน `run_once_with_facilityid`/`run_pipeline_for_facilityid` — ผนวกกลับเข้า `InputJsonApi.py` ทั้งหมด ส่วน logic ใหม่ (timeout 120, `_tag_ok` substring match, guard มิเตอร์ว่าง) คงไว้ตามไฟล์ดิบ
- `optimized_transformer_group_310869.py`: ไฟล์ดิบ import จาก `INPUTJSON2` (ต้องแก้เป็น `InputJsonApi`), ไม่มี `_API_PROJECT_ID`/`_API_SP_INDEX`, ไม่มี `main_pipeline()`/`_write_headless_geojson()`, `candidate_index` ถูก hardcode เป็น `0` แทนที่จะรับจาก `_API_SP_INDEX` (ทำให้ฟีเจอร์เลือก candidate อื่นจากแผนที่ใช้งานไม่ได้), และมีจุดที่ regress กลับไปใช้ `args.facility_id` ทั้งที่ comment เดิมเตือนไว้ตรงๆ ว่าตัวแปรนี้ไม่มีจริงในโหมด headless — ทุกจุดถูกผนวกกลับเข้าไป ส่วนฟีเจอร์ recursive-split และ voltage-after-balance reporting ที่เป็นของใหม่จริง คงไว้ทั้งหมด

## กรณีศึกษาจริง (2026-09-02): แปลง `PhaseOptimizer_01092026.py`

ฟีเจอร์ Phase Optimizer ไม่เคยผ่านรอบ "โปรโมทรุ่นใหม่" มาก่อน (ไฟล์เดิมชื่อ `PhaseOptimizer.py` เฉยๆ ไม่มีเลขรุ่น) — เคสนี้จึงเป็นครั้งแรกที่ฟีเจอร์นี้เข้าสู่ pattern Convention B (versioned-import) โดย `feature_PhaseOptimizer/run_web.py` (in-process wrapper เดิม ไม่เปลี่ยน) เปลี่ยนจาก `from PhaseOptimizer import (...)` เป็น `from PhaseOptimizer_01092026 import (...)`

Diff กับไฟล์ API เดิม (29 hunks) พบ scaffolding ที่หายไป 5 กลุ่ม — ทุกจุดมี comment อธิบายเหตุผลในไฟล์เดิมอยู่แล้ว (สัญญาณสำคัญตามคู่มือข้อ 2 ด้านบน):

1. **sys.path bootstrap** (`_THIS_DIR`/`_PROJECT_ROOT`/`_SHARELOAD_DIR`) — ไฟล์ดิบตัดออกทั้งบล็อก ทำให้ `from Runopendss_All05082026 import ...` (โมดูลอยู่ใน `feature_shareload/`) ใช้ไม่ได้เมื่อถูก import จาก cwd อื่น (เช่นจาก Flask)
2. **โหลด `TransferOptimizer-072026.py` ผ่าน `importlib` แทน `import` ตรงๆ** — ไฟล์ดิบเปลี่ยนเป็น `from TransferOptimizer import (...)` ซึ่งชื่อไม่มีขีดกลาง แปลว่าจะไป import `TransferOptimizer.py` (เวอร์ชันเก่ากว่าที่ shareload ไม่ได้ใช้แล้ว) แทนที่จะเป็นเวอร์ชันที่ shareload ใช้งานจริง — ต้องกลับไปโหลดผ่าน `importlib.util.spec_from_file_location` แบบเดิม
3. **`region` param หายทั้งคลาส** — `LVOptimizer.__init__` ตัด `region` ออก, `_ensure_json()` เปลี่ยนไปเรียก `import INPUT_FACILITY as inf` (**โมดูลนี้ไม่มีอยู่จริงในโปรเจกต์เลย** — grep ไม่เจอไฟล์ `INPUT_FACILITY.py` ที่ไหนเลย พังทันทีที่รัน) แทนที่จะเป็น `InputJsonApi.run_once_with_facilityid(fac, project_id=...)` พร้อม `inf.set_gis_region(self.region)` แบบเดิม — ต้องคืนทั้ง `region` param และ `_ensure_json()` เวอร์ชันเดิมทั้งหมด
4. **`self.error` attribute หายไป** — `run_web.py` อ้างถึง `optimizer.error` ตอน raise error กลับให้ผู้ใช้ ถ้าไม่มี attribute นี้จะพัง `AttributeError` ซ้อนทับ error จริงที่ควรแสดง
5. **`max_imbalance_pct` default เพี้ยนกลับเป็น 15.0** (3 จุด: docstring, class default, argparse default) — ไฟล์ดิบทำจาก snapshot เก่าก่อนที่ผู้ใช้จะขอปรับเป็น 25% ในรอบก่อนหน้า (ดู commit "ปรับเกณฑ์ Phase Imbalance เริ่มต้นจาก 15% เป็น 25%") ไม่ใช่การตั้งใจ revert — คืนเป็น 25.0 ทั้ง 3 จุด

Logic ใหม่จริงที่คงไว้ทั้งหมด: **Conductor Upgrade แบบ multi-segment** (เดิมอัปเกรดแค่ entry edge เส้นเดียวเข้า subtree แรงดันต่ำ ของใหม่อัปเกรดทุก segment บน path จากหม้อแปลงถึงทุกโหนดแรงดันต่ำใน subtree นั้น เพราะแรงดันตกสะสมตลอดสาย ไม่ใช่แค่จุดเดียว — เพิ่ม field `upgraded_edges`/`from_sizes` ใน `UpgradeResult` และปรับทุกจุดที่ใช้ผลลัพธ์นี้ทั้ง Excel/PNG/Plotly/console ให้ตรงกัน), และ **แก้บั๊กคำนวณโหลดต่อเฟสของหม้อแปลง** (เดิมใช้ `tx_secondary_power_by_node()` ซึ่งหม้อแปลง 1/2 เฟสที่ถูกแยกเป็นหลาย DSS object จะ mislabel เฟสผิด — ของใหม่อ่าน `NodeOrder`/`CktElement.Powers()` ตรงๆ ต่อ object)

**บั๊กตกค้างที่เจอทีหลัง (คนละรอบกับการแปลง แต่เกิดจาก field ใหม่ `upgraded_edges`)**: `run_web.py` (ตัวสร้าง `upgrade_lines.geojson` ให้แผนที่เว็บ) ยังวาดเส้นจาก `opt.applied_upgrade.edge` (entry edge เส้นเดียว) เหมือนก่อนมี multi-segment — ทำให้แผนที่เว็บโชว์เส้นสั้นกว่าที่อัปเกรดจริง (ซึ่ง plot ในไฟล์ `.py` วาดถูกอยู่แล้วเพราะ `draw_map()`/`draw_interactive_map()` ถูกอัปเดตไปพร้อมกับ dataclass ตอนแปลงรอบแรก) — แก้โดยให้ `run_web.py` วนวาดจาก `upgraded_edges` ทั้งลิสต์เหมือน `phase_addition` ที่ทำถูกอยู่แล้ว บทเรียน: เวลา field ของ dataclass เปลี่ยนความหมาย (`.edge` จาก "คือเส้นที่อัปเกรด" กลายเป็นแค่ "entry ตัวแทน") ต้องไล่หาทุกจุดที่ยังอ่าน field เดิมตรงๆ ไม่ใช่แค่จุดที่ diff เจอตอนแปลง

## กรณีศึกษาจริง (2026-09-03): แปลง `PhaseOptimizer_03092026.py`

**Scaffolding หายซ้ำ 5 จุดเดิมทุกจุด** เมื่อเทียบกับรอบ `PhaseOptimizer_01092026.py` (sys.path bootstrap, importlib load ของ `TransferOptimizer-072026.py`, `region` param + `_ensure_json()` แบบ `InputJsonApi`/`project_id`, `self.error`, `max_imbalance_pct` เพี้ยนกลับเป็น 15.0 ทั้ง 3 จุด รวมถึง default ของ `simulate_phase_addition(max_imbalance_pct=...)` ที่เพิ่งเพิ่มใหม่ในรอบนี้ด้วย) — แก้ไขแบบเดียวกับรอบก่อนทุกประการ

**สัญญาณสำคัญ**: นี่คือรอบที่ 2 ติดกันภายในไม่กี่วันที่ไฟล์ดิบของ Phase Optimizer หลุด scaffolding ชุดเดิมทั้งหมด — บ่งชี้ว่าไฟล์ดิบรุ่นใหม่แต่ละรุ่นน่าจะ fork มาจาก snapshot เก่า (ก่อนแปลงเป็น API) ไม่ใช่ต่อยอดจากไฟล์ที่โปรโมทแล้วล่าสุด ทุกครั้งที่มีไฟล์ `PhaseOptimizer_<code>.py` ใหม่มาให้แปลง **ให้ตั้งสมมติฐานว่าจะเจอ 5 จุดนี้หายอีกแน่ๆ** และเช็คตามลิสต์นี้ก่อนอ่าน diff เต็มด้วยซ้ำ

Logic ใหม่จริงที่คงไว้ทั้งหมด — โฟกัสที่ "phase imbalance ล้วนๆ โดยไม่มีแรงดันตก":
- **`optimize_phase_transfer` เปิด candidate pool กว้างขึ้น** ตอน imbalance-only (ไม่มี low-V): เดิมเล็งเฉพาะเฟสที่หนักสุดเฟสเดียว ของใหม่เล็งทุกเฟสที่โหลดเกินค่าเฉลี่ย และเพิ่ม limit ผู้สมัครจาก `max_candidates_per_iter` เป็นสูงสุด 12
- **`simulate_phase_addition` มีโหมด imbalance ใหม่**: เดิมทำงานเฉพาะตอนมีโหนดแรงดันตก (`low_v_set`) ของใหม่ทำงานได้แม้ไม่มีโหนดแรงดันตกเลย ถ้า imbalance เกินเกณฑ์ — เล็ง component ที่แบกโหลดมิเตอร์มากสุด (ตัวการหลักของ imbalance เพราะมิเตอร์ติดอยู่บนสายที่ไม่ครบ 3 เฟส) แทนที่จะเล็งจาก low-V node
- **`PhaseAddResult.delta_imbalance`** field ใหม่ + `_pa_score()` (ให้น้ำหนัก Vmin มากกว่า imbalance 10 เท่า) + `_pa_worth_applying()` (ยอมรับผลลัพธ์ถ้า Vmin ดีขึ้น **หรือ** imbalance ลดลง ≥3% โดย Vmin ไม่แย่ลงเกิน 1V — เดิมยอมรับเฉพาะ Vmin ดีขึ้นเท่านั้น)
- รายงาน (console/Excel sheet "Phase Addition"/Plotly summary table) อัปเดตให้โชว์ Δimbalance และ mode ที่ใช้ควบคู่กับ ΔVmin ทุกจุด

หลังผนวก diff เหลือแค่ 19 hunks (397 บรรทัด) ตรงกับ logic ใหม่ล้วนๆ ไม่มี scaffolding ค้าง — ย้าย `PhaseOptimizer.py` เดิมไป `notUse/`

## กรณีศึกษาจริง (2026-09-16): แปลง 3 ไฟล์พร้อมกัน (`Runopendss_All16092026.py` + `TransferOptimizer_16092026.py` + `PhaseOptimizer_16092026.py`)

รอบนี้ไฟล์ดิบใหม่ 3 ไฟล์เข้ามาพร้อมกันเป็น dependency chain: `Runopendss_All16092026.py` (feature_shareload/) ← `TransferOptimizer_16092026.py` (เข้ามาที่ **root**, ต้องย้ายเข้า `feature_shareload/`) ← `PhaseOptimizer_16092026.py` (feature_PhaseOptimizer/) — ต้องแปลงเรียงตามลำดับ dependency แล้วต่อ import ให้ครบทั้งสาย ไม่ใช่แปลงทีละไฟล์แบบแยกส่วน

**scaffolding ที่หายซ้ำในทั้ง `TransferOptimizer_16092026.py` และ `PhaseOptimizer_16092026.py`** (เซ็ตเดียวกับที่ PhaseOptimizer เจอมา 2 รอบก่อนหน้า — ยืนยันชัดเจนว่าไฟล์ดิบทุกไฟล์ในกลุ่มนี้ fork จาก snapshot เก่าเดียวกัน): sys.path bootstrap, `region` param, `_ensure_json()` แบบ `InputJsonApi`/`project_id` (ของใหม่กลับไปเรียก `INPUT_FACILITY` ที่ไม่มีจริงอีกแล้ว), `self.error` (เฉพาะ PhaseOptimizer), `--region` CLI arg, import `Runopendss_All05082026`/`TransferOptimizer-072026.py` แบบ hardcode ต้องเปลี่ยนเป็นชื่อไฟล์รุ่นใหม่ทั้งคู่

**เจอเพิ่มที่ไม่เคยเจอมาก่อน**: dropdown-selector fix (`feasible_top` sort-by-quality-ก่อนตัด-top-5 จาก commit `5b7dc78`) ก็หายไปด้วยในไฟล์ `TransferOptimizer_16092026.py` — ยืนยันว่า "fix ที่เคยทำไปแล้วในโค้ด" ก็เสี่ยงหายเวลาโปรโมทรุ่นใหม่เหมือนกัน ไม่ใช่แค่ scaffolding ตอนแปลงเป็น API ครั้งแรก ต้อง diff ตรวจ fix ประวัติศาสตร์ทุกตัวด้วย ไม่ใช่แค่ 5 ข้อใน checklist บนสุด

**ค่า business parameter ที่ต้องถามผู้ใช้แทนการเดา**: ไฟล์ดิบประกาศ `DEFAULT_MAX_IMBALANCE_PCT = 20.0` เป็น constant ใหม่ (ไม่ใช่ literal เปล่าแบบเดิม) พร้อม comment อธิบายเหตุผล — ต่างจากกรณี "15.0 ปนมาเฉยๆ" ก่อนหน้านี้ที่ชัดเจนว่าเป็นของตกหล่น เพราะรอบนี้ดูเหมือนการตัดสินใจใหม่โดยตั้งใจ จึงถามผู้ใช้ก่อนแทนที่จะเดาแล้วเขียนทับ — คำตอบคือให้คงค่า 25% เดิมไว้ (ค่าที่ผู้ใช้สั่งไว้ก่อนหน้า) เปลี่ยนแค่ตัวเลขในตัวแปร ไม่แตะ refactor ที่รวม literal กระจัดกระจายมาเป็น constant เดียว (เก็บ refactor นั้นไว้ เพราะเป็นของดี)

Logic ใหม่จริงที่คงไว้ทั้งหมด (ใหญ่สุดในทุกรอบที่ผ่านมา — รื้ออัลกอริทึมหลายจุดพร้อมกัน):
- **Runopendss**: ฟังก์ชันจับคู่มิเตอร์↔สายบริการ↔โหนดร่วม (`select_load_point_indices`/`select_service_line_indices`/`match_service_lines_to_meters`) ให้ `convert_json_to_dss_ordered()` กับ `TransferOptimizer.NetworkGraph` เห็นจุดต่อมิเตอร์ตรงกัน, กรอง backbone ตัด SUBTYPECODE=2 ทิ้ง, ตรวจเฟส LV feeder จริงแทนการเดา
- **TransferOptimizer**: ระบบย้ายกิ่งพร้อมจัดการเฟสทั้งชุด (`build_transfer_features` ฯลฯ — เลือกเฟส tie ให้ TR_B สมดุลที่สุด, ย้ายเฟสมิเตอร์ที่ไม่มีไฟหลังย้าย), ข้อจำกัดโหลดรายเฟสของหม้อแปลงทั้ง TR_A/TR_B ใหม่ทั้งคู่, ตรวจโหลดที่ไม่ได้รับไฟ (`count_dead_loads`)
- **PhaseOptimizer**: อัลกอริทึมย้ายเฟสแบบ bottom-up จากปลายกิ่งขึ้นมา (`BusNodeMapper`, `_descend_balance`, `_bottom_up_pass`, `optimize_phase_balance_bottom_up`) แทน/เสริม greedy เดิม, ระบบเช็ค overload รายเฟสของหม้อแปลงพร้อมคำแนะนำ, เกณฑ์ตรวจจับมิเตอร์ EV ใช้เปิด/ปิด Conductor Upgrade, Phase Addition ไม่เพิ่มเฟสเกินที่หม้อแปลงมีจริง

**ขั้นตอนต่อ chain ที่ต้องทำเพิ่มจากการแก้ scaffolding ปกติ**: อัปเดต path เป้าหมายของ `importlib.util.spec_from_file_location` ใน `PhaseOptimizer_16092026.py` ให้ชี้ไป `TransferOptimizer_16092026.py` (ไม่ใช่แค่คืนกลไก importlib เฉยๆ) และใน `feature_shareload/run_web.py` ให้ชี้ไปไฟล์เดียวกันด้วย — ทั้งสองจุดต้องอัปเดตพร้อมกันเสมอเวลา TransferOptimizer เปลี่ยนรุ่น

## กรณีศึกษาจริง (2026-09-17): แปลง `PhaseOptimizer_17092026.py`

**scaffolding หายซ้ำ 5 จุดเดิมทุกจุด** เป็นรอบที่ 4 ติดกันของฟีเจอร์นี้ (sys.path bootstrap, import `Runopendss_All05082026`/`from TransferOptimizer import (...)` ตรงๆ แทน importlib load ของ `Runopendss_All16092026`/`TransferOptimizer_16092026.py`, `region` param หาย, `self.error` หาย, `_ensure_json()` กลับไปเรียก `INPUT_FACILITY` ที่ไม่มีจริงอีกครั้ง, `--region` CLI arg หาย) รวมถึง `DEFAULT_MAX_IMBALANCE_PCT` เพี้ยนกลับเป็น 20.0 อีกครั้ง (ไม่ต้องถามผู้ใช้ซ้ำ เพราะยืนยันเป็น 25% ไปแล้วในรอบก่อน) — แก้ตามเช็คลิสต์เดิมทั้งหมดโดยไม่ต้องอ่าน diff ก่อนเลย เพราะรู้ล่วงหน้าอยู่แล้วว่าจะเจอจุดไหนบ้าง

Logic ใหม่จริงที่คงไว้ทั้งหมด — ฟีเจอร์ **"Design Check" (ตรวจมาตรฐานเฟสสาย)**: ใหญ่ที่สุดเท่าที่เคยแปลงมา เพิ่มการตรวจสอบว่าสาย LV เดินครบเฟสหม้อแปลงหรือไม่ตามหลักออกแบบ **แม้ไฟฟ้าจะผ่านเกณฑ์แล้ว** (เดิมทำงานเฉพาะตอนมีปัญหาทางไฟฟ้า — low-V หรือ imbalance เกิน):

- `DesignComp` dataclass ใหม่ — จัดกลุ่ม segment สาย LV ที่ไม่ครบเฟสหม้อแปลงเป็น connected component (`build_non_full_phase_comps`), ถือว่า "ต้องแก้" (`violates`) ถ้ายาว ≥100m (`DESIGN_MIN_COMP_LEN_M`) หรือมีมิเตอร์ ≥5 ตัว (`DESIGN_MIN_COMP_METERS`)
- `assess_design()` คำนวณสรุปภาพรวม (% ความยาวที่ครบเฟส, kW ที่อยู่บนสายไม่ครบเฟส, จำนวน component ที่ละเมิดเกณฑ์ ฯลฯ)
- `simulate_phase_addition()` มี**โหมดที่ 3** ("design mode") ต่อจากโหมด low-V และ imbalance เดิม — ถูกกระตุ้นตอนไฟฟ้าผ่านเกณฑ์แล้วแต่ยังมี design violation ค้างอยู่
- อัลกอริทึมกระจายเฟสใหม่แบบ **pool-based**: จัดกลุ่มมิเตอร์เป็น "pool" ตามโครงสร้างสาขา (BFS) แล้วเกลี่ยเข้าเป้าหมายแบบถ่วงน้ำหนักระหว่าง even-split กับสัดส่วนโหลดหม้อแปลง (`_POOL_SHARE_W`) แทนอัลกอริทึม greedy pool เดียวเดิม
- `LVOptimizer.run()` เพิ่ม **Step 3b** (ประเมิน design violation ทันทีหลัง baseline — ถ้าไม่มีทั้งปัญหาไฟฟ้าและ design violation ให้จบเลย) และ **Step 3c** (ถ้ามี design violation ให้เพิ่มเฟสสายตามหลักออกแบบ **ก่อน** ทำ phase transfer ในสั้นๆ เพื่อให้สาขาที่เพิ่งเพิ่มเฟสได้รับโหลดจริงก่อนรอบเกลี่ยเฟส) พร้อมประเมินซ้ำหลังปรับปรุงเสร็จ (`design_comps_after`/`design_summary_after`)
- รายงานผล: Excel มี worksheet ใหม่ `"Design Check"` (ws4b) แสดงทุก component พร้อมสถานะก่อน/หลัง, console report มีหัวข้อ Design Check เพิ่ม, Plotly interactive map มี trace ใหม่ (X สีม่วง) แสดงมิเตอร์ที่ถูกย้ายเฟสจาก Design Check

หลังผนวก scaffolding กลับครบ ไฟล์ผ่าน `ast.parse` และ import smoke test ทุกจุด (`region` อยู่ใน signature, `DEFAULT_MAX_IMBALANCE_PCT == 25.0`, ไม่มี `INPUT_FACILITY`/`Runopendss_All05082026`/`from TransferOptimizer import` หลงเหลือ) — ย้าย `PhaseOptimizer_16092026.py` เดิมไป `notUse/`, อัปเดต `feature_PhaseOptimizer/run_web.py` ให้ import จาก `PhaseOptimizer_17092026` แทน

## กรณีศึกษาจริง (2026-09-17 รอบที่ 2 — วันเดียวกัน): แก้ไข `PhaseOptimizer_17092026.py` ซ้ำ (**ชื่อไฟล์เดิม เนื้อหาใหม่**)

**รูปแบบใหม่ที่ไม่เคยเจอมาก่อน**: รอบนี้ไฟล์ดิบไม่ได้มาพร้อมเลขรุ่นใหม่ (`_18092026` ฯลฯ) แต่ทับเนื้อหาลงในไฟล์**ชื่อเดิม** (`PhaseOptimizer_17092026.py`) — สังเกตจาก `git status` ที่ขึ้น `M` (modified) แทนที่จะเป็นไฟล์ใหม่ พร้อมมีไฟล์สำรอง `PhaseOptimizer_17092026old.py` โผล่มาข้างๆ (ตรวจแล้วเนื้อหาเหมือนกับเวอร์ชันที่ API-fix ล่าสุด/commit ก่อนหน้าทุกตัวอักษร ยกเว้น line ending — ยืนยันว่าเป็น backup อัตโนมัติของฝั่งผู้ใช้ก่อนทับไฟล์) วิธีตรวจจับ: ถ้าชื่อไฟล์ไม่เปลี่ยนแต่ `git status` ขึ้น modified ให้ diff กับ `git show HEAD:<path>` แทนที่จะ diff กับไฟล์รุ่นก่อนหน้าตามชื่อ

**scaffolding หายซ้ำชุดเดิมอีกครั้ง** (docstring/usage, sys.path bootstrap, `Runopendss_All05082026`/`from TransferOptimizer import` ตรงๆ, `DEFAULT_MAX_IMBALANCE_PCT` กลับเป็น 20.0, `region` param, `self.error`, `_ensure_json()` กลับไปเรียก `INPUT_FACILITY`, `--region` CLI arg) — diff รอบนี้เล็กกว่าทุกรอบที่ผ่านมามาก (15 hunks/214 บรรทัด เทียบกับ 30+ hunks ปกติ) เพราะเป็นการอัปเดตจริงบนไฟล์ที่ fix แล้ว ไม่ใช่ fork จาก snapshot เก่าทั้งไฟล์

Logic ใหม่จริงที่คงไว้: ปรับการแสดงผล **"มิเตอร์กระจายเฟสใหม่หลัง Phase Addition"** ใน `draw_interactive_map()` (Plotly) จาก X สีม่วงเส้นเดียวรวมทุกเฟส เป็น**วงกลมเปิดแยกสีตามเฟสปลายทาง** (`_PHASE_MARK_COLOR = {A: แดง, B: เหลือง, C: น้ำเงิน}`) — จัดกลุ่มจุดเป็น 3 trace ตาม `new_pd` แทน 1 trace เดียว ให้สอดคล้องกับ convention สีเฟสที่ใช้ในหน้าเว็บแผนที่อยู่แล้ว (ดูกรณีศึกษาการปรับ map ตอนต้นของบทสนทนานี้)

## กรณีศึกษาจริง (2026-09-18): แปลง `Runopendss_All18092026.py` — รอบแรกที่ไม่มี scaffolding หายเลย

**ต่างจากทุกรอบก่อนหน้า**: diff กับ `Runopendss_All16092026.py` (11 hunks/227 บรรทัด) ไม่มี scaffolding หายแม้แต่จุดเดียว — เป็นไฟล์แรกในทุกรอบที่ผ่านมาที่ fork มาจากเวอร์ชันล่าสุดจริงๆ ไม่ใช่ snapshot เก่า ทุกอย่างที่ diff เจอเป็น logic ใหม่/แก้บั๊กแท้ๆ ทั้งหมด

**สิ่งที่ต้องระวังเป็นพิเศษ — signature ของ `build_bfs_order()` เปลี่ยนจาก 4-tuple เป็น 5-tuple** (`return new_bus, parent, level, tree_edge_set, parent_edge`) เพื่อให้ logic clamp เฟสด้านล่างใช้ `parent_edge[]` หา segment ต้นทางของแต่ละโหนดตรงๆ แทนการ lookup ด้วย `(u,v)` ซึ่งชี้ผิดเส้นได้เวลามีเส้นซ้ำระหว่างโหนดคู่เดียวกัน — **ต้อง grep หา caller ทุกจุดที่ unpack ค่าที่คืนมาก่อน promote เสมอ** (ไม่ใช่แค่จุดที่ import ชื่อฟังก์ชัน) เพราะถ้ามี caller unpack แบบ 4 ตัวแปรอยู่จะพังทันทีด้วย `ValueError: too many values to unpack` — ตรวจแล้วพบว่า `PhaseOptimizer_17092026.py`/`TransferOptimizer_16092026.py` import ชื่อฟังก์ชันมาเฉยๆ แต่ไม่เคยเรียกตรงๆ (ใช้ผ่าน `convert_json_to_dss_ordered()` เท่านั้น) จึงปลอดภัย แต่ไฟล์เก่าที่เลิกใช้แล้วอย่าง `TransferOptimizer.py` (ไม่มีขีดกลาง) ยังมี call แบบ 4-tuple อยู่ — ไม่กระทบเพราะไม่มีใครเรียกไฟล์นั้นแล้ว

Logic ใหม่จริงที่คงไว้ทั้งหมด — ชุดแก้บั๊กคุณภาพข้อมูล GIS ที่ทำให้ผลจำลองไฟฟ้าเพี้ยนแบบสุ่ม (สำคัญมาก เพราะกระทบผลลัพธ์ที่ผู้ใช้เห็นโดยตรง ไม่ใช่แค่ refactor):
- **Phase clamping**: พบ 110/11,538 หม้อแปลง (0.95%) ที่ segment ปลายสายถูกแท็ก `PHASEDESIGNATION` มากกว่า segment ต้นสาย (เช่น ต้นสายเฟสเดียวแต่ปลายสายถูกแท็ก 3 เฟส) ซึ่งเป็นไปไม่ได้ทางกายภาพ (เฟสลดได้อย่างเดียวเมื่อไกลจากหม้อแปลง ไม่มีจุดเพิ่มเฟสกลางสาย) — ถ้าปล่อยไว้ OpenDSS จะแก้สมการเฟสที่ "โผล่มาเอง" นี้เป็นเฟสที่ไม่มีเส้นทางไฟฟ้าจริงป้อน ทำให้แรงดันจำลองพังเหลือไม่กี่โวลต์ทั้งที่หน้างานไม่มีปัญหา — แก้โดย clamp เฟสของแต่ละ segment ให้เป็นสับเซตของเฟส segment ต้นทางเสมอ ไล่ตามลำดับ BFS จากหม้อแปลงออกไป
- **MEASURELENGTH vs SHAPE.LEN**: พบ 182/11,538 หม้อแปลง (1.6%) ที่ `MEASURELENGTH` (field แยกที่บันทึกตอนสร้าง/แก้ไขเส้น) เพี้ยนจาก `SHAPE.LEN` (คำนวณจาก geometry จริง จึงถูกเสมอ) ได้ถึง 3–1,000,000+ เท่า ทำให้ความยาวสายผิดมหาศาลและแรงดันตกจำลองเพี้ยนตาม — เปลี่ยนไปใช้ `SHAPE.LEN` เป็นหลักทั้ง backbone line และ service line
- **Meter phase mismatch**: มิเตอร์ที่แจ้งเฟสไม่ตรงกับเฟสจริงที่มี ณ จุดต่อบน backbone (หลัง clamp) จะถูกบังคับให้ใช้เฟสจริงของ backbone แทน (มิเตอร์ต้องต่อกับเฟสที่มีไฟจริงเท่านั้น)
- **`LAST_GIS_ISSUES`** global list ใหม่ — บันทึกทุกจุดที่ข้อมูล GIS ผิดปกติ (4 ประเภท: `phase_increase`, `phase_inconsistent_loop_edge`, `measurelength_mismatch`, `conductor_not_in_table`) ไม่ใช่ exception แต่เป็น diagnostic ให้ผู้เรียก (batch script/LVOptimizer) รายงานต่อว่าไฟล์ไหนควรแก้ที่ต้นตอ (ฐานข้อมูล GIS เอง) แม้จะ clamp ให้จำลองได้ถูกต้องแล้วก็ตาม — ยังไม่มีใครอ่านค่านี้ต่อ (ทั้ง `PhaseOptimizer_17092026.py` และ `run_web.py`) ถือเป็นจุดที่อาจต่อยอดได้ในอนาคตถ้าต้องการโชว์ GIS data-quality warning บนหน้าเว็บ
- **`solve_with_opendss()` cwd restore เข้มขึ้น**: ห่อ `os.chdir(orig_cwd)` ด้วย `try/except OSError` เผื่อโฟลเดอร์เดิมถูกลบไปแล้ว (พบเฉพาะ macOS/Linux ที่ลบโฟลเดอร์ซึ่งเป็น cwd อยู่ได้ — ไม่เคยเกิดบน Windows) เป็นการทำให้ fix เดิมจากรอบก่อนแข็งแรงขึ้น ไม่ใช่ regression
- **Path พกพาได้**: `dss_output_dir`/`json_dir` default เปลี่ยนจาก hardcode `D:\testpy\...` เป็น relative กับตำแหน่งไฟล์เอง (`os.path.dirname(os.path.abspath(__file__))`) — ปรับปรุงเพิ่มเติม ไม่ใช่ scaffolding ที่ต้องคืน

ตรวจสอบแล้ว: รันจริงกับ 2 facility (`54-008953`, `59-018889`, region NE2) ทั้งคู่ผ่านครบทุก step ไม่มี error, `meter_groups.geojson` ที่ได้ยังคง valid (`phase_after` เป็น A/B/C ล้วน ไม่มีค่าผิดปกติ) — ยืนยันว่า Runopendss เวอร์ชันใหม่ไม่กระทบผลลัพธ์ของ facility ที่ข้อมูลสะอาดอยู่แล้ว ย้าย `Runopendss_All16092026.py` เดิมไป `notUse/`, อัปเดต import ใน `PhaseOptimizer_17092026.py` และ `TransferOptimizer_16092026.py` ให้ชี้ไปรุ่นใหม่ทั้งคู่
