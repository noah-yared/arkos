import sqlite3
import os

# Connect to the SQLite database

db_path = os.path.join(
    os.path.dirname(__file__), "checkpints.sqlite"
)  # Ensure correct path
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


# Function to fetch and print all rows from a table
def read_table(table_name):
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1] for col in cursor.fetchall()]  # Extract column names

    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()

    print(f"\nData from table '{table_name}':")
    print(" | ".join(columns))  # Print column headers
    print("-" * 50)
    for row in rows:
        print(row)


def delete_last_two_entries(table_name):
    cursor.execute(f"""
        DELETE FROM {table_name}
        WHERE rowid IN (
            SELECT rowid FROM {table_name}
            ORDER BY rowid DESC
            LIMIT 100
        );
    """)
    conn.commit()
    print(f"Deleted last two entries from '{table_name}'")


if __name__ == "__main__":
    # Delete last two rows from both tables
    delete_last_two_entries("checkpoints")
    delete_last_two_entries("writes")
    # Read data from both tables
    read_table("checkpoints")
    read_table("writes")

    # Close the connection
    conn.close()
