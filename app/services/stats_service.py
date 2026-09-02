import io
import logging
import re
from pathlib import Path
import pandas as pd
import numpy as np
from ..database import get_db_connection

_SHARELOAD_FEASIBLE_DIR = Path(__file__).resolve().parent.parent.parent / 'feature_shareload' / 'FEASIBLE'
_SHARELOAD_PAIR_RE = re.compile(r'^(\d{2}-\d{6})_(\d{2}-\d{6})$')

class StatsService:
    @staticmethod
    def get_transformer_stats(region, search_query=None, per_page=15, offset=0, filter_problem=None, filter_fix=None, filter_name=None):
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        where_clause = "WHERE region = %s"
        params = [region]

        if search_query:
            where_clause += " AND (facility_id LIKE %s OR location LIKE %s)"
            params.extend([f"%{search_query}%", f"%{search_query}%"])

        if filter_problem:
            where_clause += " AND problem_summary = %s"
            params.append(filter_problem)

        if filter_fix:
            where_clause += " AND fix_guideline = %s"
            params.append(filter_fix)

        if filter_name:
            where_clause += " AND name = %s"
            params.append(filter_name)

        cur.execute(f"SELECT COUNT(*) as total FROM transformer_stats {where_clause}", params)
        res = cur.fetchone()
        total_rows = res['total'] if res else 0

        cur.execute(f"SELECT * FROM transformer_stats {where_clause} ORDER BY facility_id ASC LIMIT %s OFFSET %s",
                   params + [per_page, offset])
        stats = cur.fetchall()
        cur.close(); conn.close()
        return stats, total_rows

    @staticmethod
    def get_filter_options(region):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT problem_summary FROM transformer_stats "
            "WHERE region = %s AND problem_summary IS NOT NULL AND problem_summary != '' "
            "ORDER BY problem_summary",
            (region,)
        )
        problems = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT DISTINCT fix_guideline FROM transformer_stats "
            "WHERE region = %s AND fix_guideline IS NOT NULL AND fix_guideline != '' "
            "ORDER BY fix_guideline",
            (region,)
        )
        fixes = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT DISTINCT name FROM transformer_stats "
            "WHERE region = %s AND name IS NOT NULL AND name != '' "
            "ORDER BY name",
            (region,)
        )
        names = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        return problems, fixes, names

    @staticmethod
    def get_paired_transformers(facility_ids):
        """หม้อแปลงที่เคยรัน "แชร์โหลด" (Transfer Optimizer) จับคู่ด้วยแล้ว

        อ่านจากชื่อโฟลเดอร์ feature_shareload/FEASIBLE/<fac_a>_<fac_b>/ ตรงๆ
        (ข้อมูลเดียวกับที่หน้า /shareload แสดงในรายการที่เคยรันแล้ว) แล้วคืน
        {facility_id: [{"pair_key", "other"}]} เฉพาะ facility_id ที่ขอมา
        """
        wanted = set(facility_ids)
        result = {}
        if not wanted or not _SHARELOAD_FEASIBLE_DIR.exists():
            return result

        for folder in _SHARELOAD_FEASIBLE_DIR.iterdir():
            if not folder.is_dir():
                continue
            m = _SHARELOAD_PAIR_RE.match(folder.name)
            if not m:
                continue
            fac_a, fac_b = m.group(1), m.group(2)
            if fac_a in wanted:
                result.setdefault(fac_a, []).append({"pair_key": folder.name, "other": fac_b})
            if fac_b in wanted:
                result.setdefault(fac_b, []).append({"pair_key": folder.name, "other": fac_a})
        return result

    @staticmethod
    def upload_stats(file_stream, target_region, overwrite=False):
        try:
            raw_data = file_stream.read()
            decoded_data = None
            for enc in ["utf-8-sig", "windows-874", "tis-620"]:
                try:
                    decoded_data = raw_data.decode(enc)
                    logging.info(f"Successfully decoded CSV using {enc}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if not decoded_data:
                decoded_data = raw_data.decode("tis-620", errors='replace')

            data_io = io.StringIO(decoded_data)
            df = pd.read_csv(data_io, sep=None, engine='python', on_bad_lines='warn')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.where(pd.notnull(df), None)
            
            conn = get_db_connection()
            cur = conn.cursor()
            if overwrite: 
                cur.execute("DELETE FROM transformer_stats WHERE region = %s", (target_region,))
            
            insert_sql = """INSERT INTO transformer_stats (region, facility_id, location, feeder_id, rate_kva, kva, nerr, loss, aoj, name, error_msg, opsa_trsummary_len, x, y, rundate, kva_peak, pload_peak, peak_month, pun_peak, vmin_peak, ia_peak, ib_peak, ic_peak, gistag, datetime_peak, subtypecode, lat, lon, ia_rated, ib_rated, ic_rated, pct_ia, pct_ib, pct_ic, pload_kw, i_load, max_len, problem_summary, fix_guideline) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            
            rows = []
            for _, row in df.iterrows():
                rows.append((
                    target_region, 
                    str(row.get('FACILITYID', '')), 
                    str(row.get('LOCATION', '')), 
                    str(row.get('FEEDERID', '')), 
                    row.get('RATEKVA'), 
                    row.get('KVA'), 
                    row.get('NERR'), 
                    row.get('LOSS'), 
                    str(row.get('AOJ', '')), 
                    str(row.get('NAME', '')), 
                    str(row.get('ERROR', '')), 
                    row.get('OPSA.TRSUMMAY.LEN'), 
                    row.get('X'), 
                    row.get('Y'), 
                    str(row.get('RUNDATE', '')), 
                    row.get('KVA_PEAK'), 
                    row.get('PLOAD_PEAK'), 
                    str(row.get('PEAK_MONTH', '')), 
                    row.get('PUN_PEAK'), 
                    row.get('VMIN_PEAK'), 
                    row.get('IA_PEAK'), 
                    row.get('IB_PEAK'), 
                    row.get('IC_PEAK'), 
                    str(row.get('GISTAG', '')), 
                    str(row.get('DATETIME_PEAK', '')), 
                    row.get('SUBTYPECODE'), 
                    row.get('LAT'), 
                    row.get('LON'), 
                    row.get('IA_RATED'), 
                    row.get('IB_RATED'), 
                    row.get('IC_RATED'), 
                    row.get('pct_IA'), 
                    row.get('pct_IB'), 
                    row.get('pct_IC'), 
                    row.get('PLOAD_KW'), 
                    row.get('I_LOAD'), 
                    row.get('MAX_LEN'), 
                    str(row.get('\u0e2a\u0e23\u0e38\u0e1b\u0e1b\u0e31\u0e0d\u0e2b\u0e32', '')), 
                    str(row.get('\u0e41\u0e19\u0e27\u0e17\u0e32\u0e07\u0e01\u0e32\u0e23\u0e41\u0e01\u0e49\u0e44\u0e02', ''))
                ))
            
            for i in range(0, len(rows), 1000): 
                cur.executemany(insert_sql, rows[i:i+1000])
            
            conn.commit(); cur.close(); conn.close()
            return len(rows)
        except Exception as e:
            logging.error(f"StatsService upload error: {e}")
            raise e
