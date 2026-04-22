import mysql.connector

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'odt'
}

try:
    conn = mysql.connector.connect(**db_config)
    cur = conn.cursor()
    cur.execute("SHOW TABLES LIKE 'transformer_stats'")
    result = cur.fetchone()
    if result:
        print("TABLE_EXISTS")
    else:
        print("TABLE_MISSING")
    cur.close()
    conn.close()
except Exception as e:
    print(f"CONNECTION_ERROR: {e}")
