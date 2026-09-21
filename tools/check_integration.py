"""ตรวจว่าโค้ดที่ผนวกรุ่นใหม่แล้ว ยังมี API scaffolding และฟีเจอร์พื้นฐานครบตาม
docs/NEW_CODE_INTEGRATION_GUIDE.md (ส่วน "สัญญาที่ต้องรักษา") — รันก่อน commit ทุกครั้ง

    py -3 tools/check_integration.py        # exit 1 ถ้ามี FAIL
"""
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
# ไฟล์ legacy ที่ไม่มีใครเรียกใช้แล้ว (ไม่นับเป็นผู้อ้างถึงรุ่น) — ถ้าเลิกใช้จริงให้ย้ายไป notUse/ แล้วลบบรรทัดนี้
LEGACY = {"feature_shareload/TransferOptimizer.py", "feature_shareload/run_transfer.py"}
SKIP_DIRS = {"notUse", "venv", "__pycache__", "incoming", "tools", "scratch", "testpy", "pea_no_projects", "output"}

# family -> (โฟลเดอร์, regex ชื่อไฟล์, ต้องมี, ห้ามมี)
FAMILIES = {
    "PhaseOptimizer": ("feature_PhaseOptimizer", r"PhaseOptimizer_\d+",
        ["_SHARELOAD_DIR", "region: Optional[str] = None", "self.error: Optional[str] = None",
         "import InputJsonApi as inf", "inf.set_gis_region(self.region)", '"--region"', "region=args.region",
         "DEFAULT_MAX_IMBALANCE_PCT = 25.0", "spec_from_file_location"],
        ["INPUT_FACILITY", "from TransferOptimizer import", "Runopendss_All05082026"]),
    "TransferOptimizer": ("feature_shareload", r"TransferOptimizer_\d+",
        ["_PROJECT_ROOT", "InputJsonApi", "set_gis_region", '"--region"', "region: Optional[str]", "feasible_top"],
        ["INPUT_FACILITY"]),
    "Runopendss_All": ("feature_shareload", r"Runopendss_All\d+",
        ["os.chdir(orig_cwd)", "subtype not in (1, 3)"], []),
    "optimized_transformer_group": (".", r"optimized_transformer_group_\d+",
        ["main_pipeline", "_API_PROJECT_ID", "_API_SP_INDEX"], []),
}
FILE_CHECKS = {  # ไฟล์เดี่ยวที่มีสัญญาพฤติกรรม
    "templates/phase_optimizer_map.html": (
        ["animate: false", 'id="layerToggles"', "PHASE_TARGET_COLORS", "toggleJobs", ".catch(() => {})", "width: 2"], []),
    "feature_PhaseOptimizer/run_web.py": (
        ["steps_applied", "moved_meters", "phase_move", '"design_check"', "upgraded_edges", "_write_geojson_layers"], []),
}
fails, warns = [], []


def read(p):
    return (ROOT / p).read_text(encoding="utf-8", errors="replace")


def py_files():
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if not SKIP_DIRS & set(rel.parts) and rel.as_posix() not in LEGACY:
            yield p


def check_markers(label, text, need, forbid):
    for m in need:
        if m not in text:
            fails.append(f"{label}: ขาด '{m}'")
    for m in forbid:
        if m in text:
            fails.append(f"{label}: ห้ามมี '{m}'")


all_py = {p: p.read_text(encoding="utf-8", errors="replace") for p in py_files()}
for fam, (folder, pat, need, forbid) in FAMILIES.items():
    files = sorted((ROOT / folder).glob("*.py"))
    stems = [f.stem for f in files if re.fullmatch(pat, f.stem)]
    refs = {t for txt in all_py.values() for t in re.findall(pat, txt)}
    stale = sorted(refs - set(stems))
    if stale:
        fails.append(f"{fam}: อ้างถึงรุ่นที่ไม่มีไฟล์ (import/comment ค้าง): {stale}")
    active = [s for s in stems if s in refs]
    if len(active) != 1:
        fails.append(f"{fam}: ต้องมีรุ่นที่ถูกอ้างถึงเพียง 1 รุ่น แต่พบ {active}")
        continue
    orphans = [s for s in stems if s not in active]
    if orphans:
        warns.append(f"{fam}: ไฟล์รุ่นเก่าที่ไม่มีใครเรียก ควรย้ายไป notUse/: {orphans}")
    check_markers(f"{fam} ({active[0]})", read(f"{folder}/{active[0]}.py"), need, forbid)

for path, (need, forbid) in FILE_CHECKS.items():
    check_markers(path, read(path), need, forbid)

for stray in list((ROOT / "feature_PhaseOptimizer").glob("INPUT_FACILITY_*.py")) + list(ROOT.rglob("*old.py")):
    if not SKIP_DIRS & set(stray.relative_to(ROOT).parts):
        warns.append(f"ไฟล์ดิบ/สำรองที่ควรย้ายไป incoming/_processed/: {stray.relative_to(ROOT)}")

for w in warns:
    print("WARN", w)
for f in fails:
    print("FAIL", f)
print("PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
