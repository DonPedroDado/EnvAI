import sqlite3

class SQLiteStorage:
    def __init__(self, db_path="envai_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.table_created = False

    def store_data(self, data_row):
        if not self.table_created:
            columns = ", ".join([f'"{col}" REAL' for col in data_row.index])
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS sensor_data (
                timestamp TEXT,
                {columns}
            )
            """
            self.cursor.execute(create_table_query)
            self.table_created = True
            self.conn.commit()

        placeholders = ", ".join(["?"] * (len(data_row) + 1))
        insert_query = f"INSERT INTO sensor_data VALUES ({placeholders})"
        values = [str(data_row.name)] + data_row.tolist()
        self.cursor.execute(insert_query, values)
        self.conn.commit()