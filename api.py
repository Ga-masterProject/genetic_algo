from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from uuid import uuid4
from main import main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Database setup
DATABASE_URL = 'result.db'

def db_init():
    with sqlite3.connect(DATABASE_URL) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                slug TEXT NOT NULL,
                title TEXT NOT NULL,
                subTitle TEXT NOT NULL,
                tags TEXT NOT NULL,
                status TEXT NOT NULL,
                best_teams TEXT,
                names_combined TEXT
            )
        ''')
        conn.commit()

# Pydantic model
class Competition(BaseModel):
    slug: str
    title: str
    subTitle: str
    tags: str

def insert_competition(competition: Competition) -> str:
    task_id = str(uuid4())
    status = 'in progress'
    with sqlite3.connect(DATABASE_URL) as conn:
        conn.execute('''
            INSERT INTO tasks (id, slug, title, subTitle, tags, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_id, competition.slug, competition.title, competition.subTitle, competition.tags, status))
        conn.commit()
    return task_id

def update_data(task_id: str, best_teams: str, names_combined:str, status: str):
    with sqlite3.connect(DATABASE_URL) as conn:
        conn.execute('UPDATE tasks SET status = ?, best_teams = ?, names_combined = ? WHERE Id = ?', (status, best_teams, names_combined, task_id))
        conn.commit()

def genetic_algo(task_id: str, competition: Competition):
    best_teams, names_combined = main(competition)
    update_data(task_id, best_teams, names_combined, 'completed')
    print("GA completed, Best teams are",best_teams)

@app.on_event("startup")
async def on_startup():
    db_init()

@app.post("/tasks/")
async def create_tasks(competition: Competition, background_tasks: BackgroundTasks):
    task_id = insert_competition(competition)
    print("start collecting data")
    background_tasks.add_task(genetic_algo, task_id, competition)
    return {"status": "accepted", "task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    with sqlite3.connect(DATABASE_URL) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT status FROM tasks WHERE id = ?', (task_id,))
        result = cursor.fetchone()
        if result:
            return {"task_id": task_id, "status": result[0]}
        else:
            raise HTTPException(status_code=404, detail="Competition not found")

