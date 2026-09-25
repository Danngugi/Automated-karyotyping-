import sqlite3

def log_run(database: str, source: str, species: str, object_count: int, approved: bool) -> None:
    con=sqlite3.connect(database); con.execute('CREATE TABLE IF NOT EXISTS runs (source TEXT, species TEXT, object_count INTEGER, approved INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
    con.execute('INSERT INTO runs(source,species,object_count,approved) VALUES (?,?,?,?)',(source,species,object_count,int(approved))); con.commit(); con.close()
