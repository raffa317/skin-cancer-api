import sqlite3
import bcrypt
import datetime
import os

DB_NAME = "skin_cancer_app.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    # Create Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    # Create Scans Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                  (username, hashed.decode('utf-8')))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Username exists
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user:
        user_id, stored_hash = user
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return user_id
    return None

def log_scan(user_id, filename, prediction, confidence):
    conn = get_connection()
    c = conn.cursor()
    timestamp = datetime.datetime.now()
    c.execute('''
        INSERT INTO scans (user_id, filename, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, filename, prediction, confidence, timestamp))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT filename, prediction, confidence, timestamp 
        FROM scans 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    ''', (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows
