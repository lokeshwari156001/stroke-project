import pandas as pd
import random
import database as db

def main():
    csv_path = 'healthcare-dataset-stroke-data.csv'
    df = pd.read_csv(csv_path)

    # Check if patient_name already exists
    if 'patient_name' in df.columns:
        print("patient_name column already exists. Overwriting/Updating...")

    # Generate Names
    # We'll create a few specific ones for easy testing
    specific_names = [
        "John Doe", "Jane Smith", "Robert Brown", "Emily Davis", "Michael Wilson",
        "Sarah Miller", "William Taylor", "Jessica Anderson", "David Thomas", "Lisa Moore"
    ]
    
    names = []
    users_to_add = []

    for i in range(len(df)):
        if i < len(specific_names):
            name = specific_names[i]
            # We will add these to the DB
            users_to_add.append(i)
        else:
            name = f"Patient_{i}"
        names.append(name)

    df['patient_name'] = names
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with 'patient_name' column.")

    # Update Database
    print("Updating users.db with test patients...")
    db.init_db()
    
    for idx in users_to_add:
        name = names[idx]
        gender = df.iloc[idx]['gender']
        # Default password '12345'
        if db.add_user(name, "12345", "Patient", gender):
            print(f"Created User -> Username: '{name}', Password: '12345', Gender: '{gender}'")
        else:
            print(f"User '{name}' already exists or could not be created.")

if __name__ == "__main__":
    main()
