import sqlite3

db_path = 'users.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

username = 'jayaselvam'
new_gender = 'Male'

# Update the gender
c.execute("UPDATE users SET gender=? WHERE username=?", (new_gender, username))
conn.commit()

print(f"Updated gender for {username} to {new_gender}")

# Verify
c.execute("SELECT * FROM users WHERE username=?", (username,))
print("Current record:", c.fetchone())

conn.close()
