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
| Phase Optimizer | `feature_PhaseOptimizer/PhaseOptimizer.py` | `feature_PhaseOptimizer/run_web.py` (import ใน-process) | `app/routes/phase_optimizer.py`, `app/services/phase_optimizer_service.py` | in-process wrapper |
| Shareload / Transfer Optimizer | `feature_shareload/TransferOptimizer-*.py` | `feature_shareload/run_web.py` (เรียกผ่าน `subprocess.run`) | `app/routes/shareload.py` | subprocess wrapper |

รายละเอียดโครงสร้างโฟลเดอร์ทั้งหมด ดูที่ [`docs/PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)

## กรณีศึกษาจริง (2026-08-31): แปลง `INPUTJSON2_310869.py` + `optimized_transformer_group_310869.py`

- `INPUTJSON2_310869.py`: ไฟล์ดิบไม่มี `_BASE_DIR`, `GISConfig`/`set_gis_region`, `project_id` param ใน `run_once_with_facilityid`/`run_pipeline_for_facilityid` — ผนวกกลับเข้า `InputJsonApi.py` ทั้งหมด ส่วน logic ใหม่ (timeout 120, `_tag_ok` substring match, guard มิเตอร์ว่าง) คงไว้ตามไฟล์ดิบ
- `optimized_transformer_group_310869.py`: ไฟล์ดิบ import จาก `INPUTJSON2` (ต้องแก้เป็น `InputJsonApi`), ไม่มี `_API_PROJECT_ID`/`_API_SP_INDEX`, ไม่มี `main_pipeline()`/`_write_headless_geojson()`, `candidate_index` ถูก hardcode เป็น `0` แทนที่จะรับจาก `_API_SP_INDEX` (ทำให้ฟีเจอร์เลือก candidate อื่นจากแผนที่ใช้งานไม่ได้), และมีจุดที่ regress กลับไปใช้ `args.facility_id` ทั้งที่ comment เดิมเตือนไว้ตรงๆ ว่าตัวแปรนี้ไม่มีจริงในโหมด headless — ทุกจุดถูกผนวกกลับเข้าไป ส่วนฟีเจอร์ recursive-split และ voltage-after-balance reporting ที่เป็นของใหม่จริง คงไว้ทั้งหมด
