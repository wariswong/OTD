import mysql.connector
from .config import Config

def get_db_connection():
    """Returns a mysql.connector connection object."""
    return mysql.connector.connect(
        host=Config.DB_HOST,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        database=Config.DB_NAME
    )

def get_db_cursor(dictionary=True):
    """Context manager like helper could be added, but for now matching existing pattern."""
    conn = get_db_connection()
    cur = conn.cursor(dictionary=dictionary)
    return conn, cur
