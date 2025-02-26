from http.client import HTTPException
import sqlite3
from datetime import datetime

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     user_id INTEGER,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def create_document_store():
    conn = get_db_connection()


    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     user_id INTEGER,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE )''')
    conn.close()

def create_users_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     email TEXT UNIQUE NOT NULL,
                     hashed_password TEXT NOT NULL)''')
    conn.close()


def insert_application_logs(session_id,user_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?, ?)',
                 (session_id,user_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

# def get_user_chat_history(user_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('SELECT session_id, user_query, gpt_response FROM application_logs WHERE user_id = ? ORDER BY created_at', (user_id,))
#     chat_history = {}
#     for row in cursor.fetchall():

#         session_id = row['session_id']
        
#         # Initialize chat_history for session_id if it doesn't exist
#         if session_id not in chat_history:
#             chat_history[session_id] = []
        
#         chat_history[session_id].extend([
#             {"role": "human", "content": row['user_query']},
#             {"role": "ai", "content": row['gpt_response']}
#         ])
#     conn.close()
#     return chat_history

def get_chat_history(session_id,user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE user_id = ? AND session_id = ? ORDER BY created_at', (user_id,session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

def get_user_chat_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get the session details (model and timestamp) for each session
    cursor.execute('SELECT session_id, model, created_at FROM application_logs WHERE user_id = ? GROUP BY session_id ORDER BY created_at', (user_id,))
    sessions = {}
    for row in cursor.fetchall():
        session_id = row['session_id']
        model = row['model']
        session_timestamp = row['created_at']
        
        # Initialize the session in the dictionary if not already initialized
        if session_id not in sessions:
            sessions[session_id] = {
                "model": model,
                "timestamp": session_timestamp,
                "queries": []
            }

    # Fetch the queries and responses for each session
    cursor.execute('SELECT session_id, user_query, gpt_response, created_at FROM application_logs WHERE user_id = ? ORDER BY created_at', (user_id,))
    for row in cursor.fetchall():
        session_id = row['session_id']
        user_query = row['user_query']
        gpt_response = row['gpt_response']
        created_at = row['created_at']

        # Add the query and response to the appropriate session's queries list
        if session_id in sessions:
            sessions[session_id]["queries"].append({
                "query": user_query,
                "response": gpt_response,
                "timestamp": created_at
            })

    conn.close()
    
    # Format the final response in the desired structure
    formatted_history = {
        session_id: {
            "model": session_data["model"],
            "timestamp": session_data["timestamp"],
            "queries": session_data["queries"]
        }
        for session_id, session_data in sessions.items()
    }
    
    return formatted_history
def delete_chat_session(user_id,session_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM application_logs WHERE user_id = ? AND session_id = ?', (user_id,session_id))
    conn.commit()
    conn.close()
    return True

def insert_document_record(filename,user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename, user_id) VALUES (?,?)', (filename,user_id))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id,user_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ? AND user_id = ?', (file_id,user_id))
    conn.commit()
    conn.close()
    return True

def get_all_documents(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, user_id, filename, upload_timestamp FROM document_store WHERE user_id = ? ORDER BY upload_timestamp DESC', (user_id,))
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]





def insert_user(email: str, hashed_password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (email, hashed_password) VALUES (?, ?)', (email, hashed_password))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    conn.close()

def get_user_by_email(email: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def reset_password_db(email: str, hashed_password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET hashed_password = ? WHERE email = ?', (hashed_password, email))
    conn.commit()
    conn.close()
    return {"message": "Password reset successfully!"}

# Initialize the database tables
create_application_logs()
create_document_store()
create_users_table()






# from neo4j import GraphDatabase, basic_auth
# from datetime import datetime

# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASSWORD = "your_password"  # change this to your actual password

# driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))


# def create_user(email, hashed_password):
#     with driver.session() as session:
#         session.run("""
#             MERGE (u:User {email: $email})
#             ON CREATE SET u.hashed_password = $hashed_password,
#                           u.created_at = datetime()
#             RETURN u
#         """, email=email, hashed_password=hashed_password)

# def insert_application_log(session_id, user_email, user_query, gpt_response, model):
#     with driver.session() as session:
#         session.run("""
#             MATCH (u:User {email: $user_email})
#             CREATE (log:ApplicationLog {
#                 session_id: $session_id,
#                 user_query: $user_query,
#                 gpt_response: $gpt_response,
#                 model: $model,
#                 created_at: datetime()
#             })
#             MERGE (u)-[:HAS_LOG]->(log)
#         """, user_email=user_email, session_id=session_id, user_query=user_query, gpt_response=gpt_response, model=model)

# def get_chat_history(session_id, user_email):
#     with driver.session() as session:
#         result = session.run("""
#             MATCH (u:User {email: $user_email})-[:HAS_LOG]->(log:ApplicationLog {session_id: $session_id})
#             RETURN log.user_query AS user_query, log.gpt_response AS gpt_response, log.created_at AS created_at
#             ORDER BY log.created_at
#         """, user_email=user_email, session_id=session_id)
#         messages = []
#         for record in result:
#             messages.extend([
#                 {"role": "human", "content": record["user_query"]},
#                 {"role": "ai", "content": record["gpt_response"]}
#             ])
#         return messages

# def delete_chat_session(user_email, session_id):
#     with driver.session() as session:
#         session.run("""
#             MATCH (u:User {email: $user_email})-[r:HAS_LOG]->(log:ApplicationLog {session_id: $session_id})
#             DELETE r, log
#         """, user_email=user_email, session_id=session_id)

# def insert_document_record(filename, user_email):
#     with driver.session() as session:
#         result = session.run("""
#             MATCH (u:User {email: $user_email})
#             CREATE (d:Document {
#                 filename: $filename,
#                 upload_timestamp: datetime()
#             })
#             MERGE (u)-[:UPLOADED]->(d)
#             RETURN id(d) AS file_id
#         """, user_email=user_email, filename=filename)
#         return result.single()["file_id"]

# def get_all_documents(user_email):
#     with driver.session() as session:
#         result = session.run("""
#             MATCH (u:User {email: $user_email})-[:UPLOADED]->(d:Document)
#             RETURN id(d) AS id, d.filename AS filename, d.upload_timestamp AS upload_timestamp
#             ORDER BY d.upload_timestamp DESC
#         """, user_email=user_email)
#         documents = [dict(record) for record in result]
#         return documents

# def reset_password_db(email, hashed_password):
#     with driver.session() as session:
#         session.run("""
#             MATCH (u:User {email: $email})
#             SET u.hashed_password = $hashed_password
#         """, email=email, hashed_password=hashed_password)
#         return {"message": "Password reset successfully!"}
