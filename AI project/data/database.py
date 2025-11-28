import sqlite3
import hashlib

DB_NAME = "users.db"

def init_db():
    """Initialize the database and create the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT,
            gender TEXT
        )
    ''')
    
    # Check if columns exist (for migration)
    c.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in c.fetchall()]
    if 'role' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT")
    if 'gender' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN gender TEXT")
        
    conn.commit()
    conn.close()

def make_hashes(password):
    """Return a hashed password."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """Check if the password matches the hash."""
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def add_user(username, password, role, gender):
    """Add a new user to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_password = make_hashes(password)
    try:
        c.execute('INSERT INTO users(username, password, role, gender) VALUES (?,?,?,?)', (username, hashed_password, role, gender))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

def login_user(username, password):
    """Verify user credentials and return (role, gender) if successful."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =?', (username,))
    data = c.fetchall()
    conn.close()
    
    if data:
        if check_hashes(password, data[0][1]):
            # data[0] is (username, password, role, gender)
            role = data[0][2] if len(data[0]) > 2 else "Patient"
            gender = data[0][3] if len(data[0]) > 3 else "Male" # Default to Male if not found
            return role, gender
    return None, None
