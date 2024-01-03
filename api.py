import concurrent.futures
import sqlite3
from main import main
import nltk

# Database setup
DATABASE_URL = 'result.db'
MAIN_DB_URL = 'data3.db'
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

def fetch_competitions():
    with sqlite3.connect(MAIN_DB_URL) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT Id, Slug, Title, Subtitle, Tags FROM test_data')
        return cursor.fetchall()

def insert_competition(competition):
    status = 'in progress'
    with sqlite3.connect(DATABASE_URL) as conn:
        conn.execute('''
            INSERT INTO tasks (id, slug, title, subTitle, tags, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (competition['Id'], competition['Slug'], competition['Title'], competition['Subtitle'], competition['Tags'], status))
        conn.commit()
    return competition['Id']

def process_competition(competition):
    Id, Slug, Title, Subtitle, Tags = competition
    Id = insert_competition({'Id': Id, 'Slug': Slug, 'Title': Title, 'Subtitle': Subtitle, 'Tags': Tags})
    print(f"Processing competition: {Id}")
    best_teams, names_combined = main(competition)  # Your genetic algorithm logic
    update_competition_status(Id, "completed", best_teams, names_combined)
    print(f"Completed competition: {Id}")

def update_competition_status(competition_id, new_status, best_teams, names_combined):
    with sqlite3.connect(DATABASE_URL) as conn:
        conn.execute('UPDATE tasks SET status = ?, best_teams = ?, names_combined = ? WHERE Id = ?', (new_status, best_teams, names_combined, competition_id))
        conn.commit()

def main_executor():
    db_init()  # Initialize the database and tables
    max_threads = 100   # Number of competitions to run concurrently

    all_competitions = fetch_competitions()
    total_competitions = len(all_competitions)
    processed_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        while processed_count < total_competitions:
            # Get the next set of competitions to process
            next_competitions = all_competitions[processed_count:processed_count + max_threads]
            futures = [executor.submit(process_competition, comp) for comp in next_competitions]

            # Wait for the current set of competitions to be processed
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

            processed_count += len(next_competitions)

if __name__ == "__main__":
    main_executor()

# Pydantic model
# class Competition(BaseModel):
#     slug: str
#     title: str
#     subTitle: str
#     tags: str

# def insert_competition(competition: Competition) -> str:
#     task_id = str(uuid4())
#     status = 'in progress'
#     with sqlite3.connect(DATABASE_URL) as conn:
#         conn.execute('''
#             INSERT INTO tasks (id, slug, title, subTitle, tags, status)
#             VALUES (?, ?, ?, ?, ?, ?)
#         ''', (task_id, competition.slug, competition.title, competition.subTitle, competition.tags, status))
#         conn.commit()
#     return task_id

# def update_data(task_id: str, best_teams: str, names_combined:str, status: str):
#     with sqlite3.connect(DATABASE_URL) as conn:
#         conn.execute('UPDATE tasks SET status = ?, best_teams = ?, names_combined = ? WHERE Id = ?', (status, best_teams, names_combined, task_id))
#         conn.commit()

# def genetic_algo(task_id: str, competition: Competition):
#     best_teams, names_combined = main(competition)
#     update_data(task_id, best_teams, names_combined, 'completed')
#     print("GA completed, Best teams are",best_teams)

# @app.on_event("startup")
# async def on_startup():
#     db_init()

# @app.post("/tasks/")
# async def create_tasks(competition: Competition, background_tasks: BackgroundTasks):
#     task_id = insert_competition(competition)
#     print("start collecting data")
#     background_tasks.add_task(genetic_algo, task_id, competition)
#     return {"status": "accepted", "task_id": task_id}

# @app.get("/status/{task_id}")
# async def get_status(task_id: str):
#     with sqlite3.connect(DATABASE_URL) as conn:
#         cursor = conn.cursor()
#         cursor.execute('SELECT status FROM tasks WHERE id = ?', (task_id,))
#         result = cursor.fetchone()
#         if result:
#             return {"task_id": task_id, "status": result[0]}
#         else:
#             raise HTTPException(status_code=404, detail="Competition not found")

