"""
run_transfer.py
---------------
รับ FACILITYID สองตัวจาก input แล้วรัน TransferOptimizer pipeline

Usage:
    python run_transfer.py
    python run_transfer.py <FAC_A> <FAC_B>          # รับจาก CLI args ได้ด้วย
"""

import sys, re
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FAC_RE = re.compile(r"^\d{2}-\d{6}$")

def ask_facilityid(label: str) -> str:
    while True:
        val = input(f"  {label} (XX-XXXXXX) : ").strip()
        if FAC_RE.match(val):
            return val
        print(f"    [!] รูปแบบไม่ถูกต้อง — ต้องเป็น 2 หลัก - 6 หลัก เช่น 04-123456")


def main():
    print("=" * 60)
    print("  LV Load Transfer Optimizer")
    print("=" * 60)

    # รับจาก CLI args ถ้ามี (สะดวกสำหรับ web app เรียก subprocess)
    if len(sys.argv) == 3:
        fac_a, fac_b = sys.argv[1].strip(), sys.argv[2].strip()
    else:
        print("\nกรอก FACILITYID หม้อแปลงทั้ง 2 ตัว")
        fac_a = ask_facilityid("TR_A  หม้อแปลงโหลดเกิน ")
        fac_b = ask_facilityid("TR_B  หม้อแปลงรับโหลด ")

    if fac_a == fac_b:
        print("[!] FACILITYID A และ B เป็นตัวเดียวกัน — ยกเลิก")
        sys.exit(1)

    from datetime import datetime
    import json

    from TransferOptimizer import (
        TransferOptimizer,
        print_results_table,
        save_excel_report,
        draw_topology_map,
        draw_interactive_map,
        collect_optimal_node_voltages,
    )

    out_dir = Path(r"D:\TRneighborhood\output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    fa_s = re.sub(r"[^A-Za-z0-9]", "_", fac_a)
    fb_s = re.sub(r"[^A-Za-z0-9]", "_", fac_b)
    stem = out_dir / f"transfer_{fa_s}_{fb_s}_{ts}"

    optimizer = TransferOptimizer(
        fac_a=fac_a,
        fac_b=fac_b,
        force_refresh=True,
    )
    results = optimizer.run()

    if not results:
        print("\nไม่มีผลลัพธ์ — ลองปรับพารามิเตอร์แล้วรันใหม่")
        sys.exit(0)

    print_results_table(results, optimizer.net_a, optimizer.net_b)
    save_excel_report(results, optimizer.net_a, optimizer.net_b, str(stem) + ".xlsx")

    # Voltage map สำหรับ optimal scenario
    volt_a, volt_b, phase_a, phase_b = {}, {}, {}, {}
    raw_a = raw_b = None
    opt = next((r for r in results if r.get("optimal")), None)
    if opt:
        print("\n[Voltage Map] รัน simulation สำหรับ voltage coloring...")
        with open(optimizer._json_a, encoding="utf-8") as fh:
            raw_a = json.load(fh)
        with open(optimizer._json_b, encoding="utf-8") as fh:
            raw_b = json.load(fh)
        volt_a, volt_b, phase_a, phase_b = collect_optimal_node_voltages(
            optimizer.net_a, optimizer.net_b, raw_a, raw_b, opt,
            snap_tol_a=optimizer.net_a.snap_tol,
            snap_tol_b=optimizer.net_b.snap_tol,
        )
        print(f"  TR_A: {len(volt_a)} nodes  TR_B: {len(volt_b)} nodes")

    draw_topology_map(optimizer.net_a, optimizer.net_b, results,
                      str(stem) + ".png", volt_a=volt_a, volt_b=volt_b)
    draw_interactive_map(optimizer.net_a, optimizer.net_b, results,
                         str(stem) + ".html",
                         volt_a=volt_a, volt_b=volt_b,
                         phase_a=phase_a, phase_b=phase_b,
                         raw_a=raw_a, raw_b=raw_b)

    print(f"\nเสร็จสิ้น — ผลลัพธ์บันทึกที่ {out_dir}")


if __name__ == "__main__":
    main()
