from ..database import get_db_connection

class AdminService:
    @staticmethod
    def get_admin_by_id(hr_employee_id):
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM admins WHERE hr_employee_id = %s", (hr_employee_id,))
        admin = cur.fetchone()
        cur.close()
        conn.close()
        return admin

    @staticmethod
    def get_all_admins():
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM admins ORDER BY role, hr_employee_id")
        admins = cur.fetchall()
        cur.close()
        conn.close()
        return admins

    @staticmethod
    def add_admin(hr_employee_id, role, region=None):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO admins (hr_employee_id, role, region) VALUES (%s, %s, %s)",
                (hr_employee_id, role, region)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding admin: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def update_admin(admin_id, role, region=None):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE admins SET role = %s, region = %s WHERE id = %s",
            (role, region, admin_id)
        )
        conn.commit()
        cur.close()
        conn.close()

    @staticmethod
    def delete_admin(admin_id):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM admins WHERE id = %s", (admin_id,))
        conn.commit()
        cur.close()
        conn.close()
