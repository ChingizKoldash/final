from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, Request, Form, Response, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from typing import List, Optional , Dict
import uvicorn
from collections import defaultdict
from uuid import uuid4
import asyncio
import threading, time
from datetime import datetime
import random, string

SECRET_KEY = "secret-key"
ALGORITHM = "HS256"


answered_users = {} 

def clear_answered_users_periodically():
    while True:
        time.sleep(900)  # 15 минут
        answered_users.clear()

threading.Thread(target=clear_answered_users_periodically, daemon=True).start()
# DB Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./quiz.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_size=10,
    max_overflow=20,
    pool_timeout=30
)
guest_answers = {}

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    quizzes = relationship("Quiz", back_populates="owner")

class Quiz(Base):
    __tablename__ = "quizzes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="quizzes")
    questions = relationship("Question", back_populates="quiz")
    code = Column(String, unique=True, index=True)

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    answer = Column(String)  # правильный вариант
    options = Column(String)  # варианты ответа через ';'
    quiz_id = Column(Integer, ForeignKey("quizzes.id"))
    quiz = relationship("Quiz", back_populates="questions")

class UserAnswer(Base):
    __tablename__ = "user_answers"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    quiz_id = Column(Integer, ForeignKey("quizzes.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    is_correct = Column(Boolean)


class UserQuizProgress(Base):
    __tablename__ = "progress"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    quiz_id = Column(Integer, ForeignKey("quizzes.id"))
    current_index = Column(Integer, default=0)
    score = Column(Integer, default=0)
    start_time = Column(String, default=lambda: datetime.utcnow().isoformat())
    end_time = Column(String, nullable=True)

# Auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()

def get_current_user(token: Optional[str] = Cookie(None), db: Session = Depends(lambda: SessionLocal())):
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = get_user(db, username=username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# App setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Routes
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/token")
def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username})
    response = RedirectResponse(url="/quizzes", status_code=302)
    response.set_cookie(key="token", value=token, httponly=True)
    return response

@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    
    if db.query(User).filter(User.username == username).first():
        return HTMLResponse("Username already exists", status_code=400)
    user = User(username=username, hashed_password=get_password_hash(password))
    db.add(user)
    db.commit()
    return RedirectResponse("/login", status_code=302)

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/quizzes", response_class=HTMLResponse)
def list_quizzes(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    
    quizzes = db.query(Quiz).filter_by(owner_id=user.id).all()
    return templates.TemplateResponse("quizzes.html", {"request": request, "quizzes": quizzes})

@app.get("/quiz/{quiz_id}", response_class=HTMLResponse)
def view_quiz(quiz_id: int, request: Request,db: Session = Depends(get_db)):
   
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    questions = quiz.questions if quiz else []
    return templates.TemplateResponse("quiz.html", {"request": request, "quiz": quiz, "questions": questions})


@app.get("/create_quiz", response_class=HTMLResponse)
def create_quiz_form(request: Request):
    return templates.TemplateResponse("create_quiz_dynamic.html", {"request": request})

@app.post("/create_quiz")
async def create_quiz(request: Request, title: str = Form(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    form = await request.form()  
    
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    quiz = Quiz(title=title, owner_id=user.id, code=code)

    db.add(quiz)
    db.commit()
    index = 0
    while f"question_{index}" in form:
        text = form.get(f"question_{index}")
        answer_letter = form.get(f"answer_{index}")
        options = [form.get(f"option_{index}_{i}") for i in range(4)]

        if text and answer_letter and all(options):
            # определяем индекс правильного ответа
            correct_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}.get(answer_letter.upper())
            if correct_index is not None:
                correct_answer = options[correct_index]
                options_str = ";".join(options)

                question = Question(
                    text=text,
                    answer=correct_answer,
                    options=options_str,
                    quiz_id=quiz.id
                )
                db.add(question)
        index += 1


    db.commit()
    return RedirectResponse(url="/quizzes", status_code=302)

@app.get("/edit_quiz/{quiz_id}", response_class=HTMLResponse)
def edit_quiz_form(quiz_id: int, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter_by(id=quiz_id, owner_id=user.id).first()
    if not quiz:
        return HTMLResponse("Quiz not found or access denied", status_code=404)
    return templates.TemplateResponse("edit_quiz.html", {"request": request, "quiz": quiz})

@app.post("/delete_quiz/{quiz_id}")
def delete_quiz(quiz_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter_by(id=quiz_id, owner_id=user.id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found or access denied")
    db.query(Question).filter_by(quiz_id=quiz_id).delete()
    db.delete(quiz)
    db.commit()
    return RedirectResponse(url="/quizzes", status_code=302)

@app.post("/edit_quiz/{quiz_id}")
async def update_quiz(
    quiz_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    quiz = db.query(Quiz).filter_by(id=quiz_id, owner_id=user.id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found or access denied")

    form = await request.form()
    quiz.title = form.get("title")

    # Обновим существующие вопросы
    for question in quiz.questions:
        qid = str(question.id)
        if f"delete_{qid}" in form:
            db.delete(question)
        else:
            question.text = form.get(f"question_{qid}")
            question.options = form.get(f"options_{qid}")
            question.answer = form.get(f"answer_{qid}")

    # Добавим новые вопросы
    max_id = db.query(Question.id).order_by(Question.id.desc()).first()
    next_id = (max_id[0] if max_id else 0) + 1
    form = await request.form()
    index = 0
    while True:
        index += 1
        new_q = form.get(f"new_question_{index}")
        new_o = form.get(f"new_options_{index}")
        new_a = form.get(f"new_answer_{index}")
        if not new_q and not new_o and not new_a and index > len(form)/4:
            break

        if new_q and new_o and new_a:
            q = Question(
                id=next_id,
                text=new_q,
                options=new_o,
                answer=new_a,
                quiz_id=quiz.id
            )
            db.add(q)
            next_id += 1
        

    db.commit()
    return RedirectResponse(url="/quizzes", status_code=302)


@app.get("/quiz/{quiz_id}/start", response_class=HTMLResponse)
def start_quiz(quiz_id: int, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    progress = db.query(UserQuizProgress).filter_by(user_id=user.id, quiz_id=quiz_id).first()
    if not progress:
        progress = UserQuizProgress(user_id=user.id, quiz_id=quiz_id, current_index=0, score=0)
        db.add(progress)
        db.commit()
    return RedirectResponse(url=f"/quiz/{quiz_id}/question", status_code=302)

@app.get("/quiz/{quiz_id}/question", response_class=HTMLResponse)
def get_question(quiz_id: int, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    progress = db.query(UserQuizProgress).filter_by(user_id=user.id, quiz_id=quiz_id).first()
    if not progress or progress.current_index >= len(quiz.questions):
        return HTMLResponse("Quiz completed!", status_code=200)
    question = quiz.questions[progress.current_index]
    correct_count = db.query(UserAnswer).filter_by(
        quiz_id=quiz_id,
        question_id=question.id,
        is_correct=True
    ).count()

    return templates.TemplateResponse("question.html", {
        "request": request,
        "quiz": quiz,
        "question": question,
        "progress": progress,
        "correct_count": correct_count
    })

@app.post("/quiz/{quiz_id}/answer")
def submit_answer(quiz_id: int, question_id: int = Form(...), answer: str = Form(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    question = db.query(Question).filter_by(id=question_id).first()
    progress = db.query(UserQuizProgress).filter_by(user_id=user.id, quiz_id=quiz_id).first()
    if question and progress and question.answer.strip().lower() == answer.strip().lower():
        progress.score += 1
    progress.current_index += 1
    quiz = db.query(Quiz).filter_by(id=quiz_id).first()
    if progress.current_index >= len(quiz.questions):
        progress.end_time = datetime.utcnow().isoformat()
    db.commit()

    # Добавим пользователя в список ответивших
    key = (quiz_id, question_id)
    if key not in answered_users:
        answered_users[key] = set()
    answered_users[key].add(user.username)

    if progress.current_index >= len(quiz.questions):
        return RedirectResponse(url=f"/quiz/{quiz_id}/result", status_code=302)

    return RedirectResponse(url=f"/quiz/{quiz_id}/question", status_code=302)


@app.get("/quiz/{quiz_id}/result", response_class=HTMLResponse)
def quiz_result(quiz_id: int, request: Request, user: User = Depends(get_current_user) , db: Session = Depends(get_db)):
    
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    total = len(quiz.questions)
    progress = db.query(UserQuizProgress).filter_by(user_id=user.id, quiz_id=quiz_id).first()


    all_progress = db.query(UserQuizProgress).filter_by(quiz_id=quiz_id).all()
    leaderboard = []
    for p in all_progress:
        u = db.query(User).filter_by(id=p.user_id).first()
        if not p.end_time:
            continue
        start = datetime.fromisoformat(p.start_time)
        end = datetime.fromisoformat(p.end_time)
        leaderboard.append({
            "name": u.username if u else "Guest",
            "score": p.score,
            "duration": (end - start).total_seconds()
        })

    leaderboard.sort(key=lambda x: (-x["score"], x["duration"]))

    return templates.TemplateResponse("result.html", {
    "request": request,
    "score": progress.score,
    "total": total,
    "leaderboard": leaderboard,
    "is_guest": False
})

# Startup
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

active_connections: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(list))

threading.Thread(target=clear_answered_users_periodically, daemon=True).start()

@app.websocket("/ws/quiz/{quiz_id}/question/{question_id}")
async def quiz_question_ws(websocket: WebSocket, quiz_id: int, question_id: int, db: Session = Depends(get_db)):
    await websocket.accept()
    while True:
        await asyncio.sleep(1)
        correct_count = db.query(UserAnswer).filter_by(
            quiz_id=quiz_id,
            question_id=question_id,
            is_correct=True
        ).count()
        users = list(answered_users.get((quiz_id, question_id), []))
        await websocket.send_json({
            "correct_count": correct_count,
            "answered_names": users
        })

    
@app.get("/join", response_class=HTMLResponse)
def join_quiz_by_code(code: str, request: Request, db: Session = Depends(get_db)):

    quiz = db.query(Quiz).filter_by(code=code).first()
    if not quiz:
        return HTMLResponse("Invalid room code", status_code=404)
    return templates.TemplateResponse("join_quiz.html", {"request": request, "quiz": quiz})
guest_sessions = {}

@app.post("/guest/start/{quiz_id}")
def guest_start_quiz(quiz_id: int, guest_name: str = Form(...), response: Response = None):
    session_id = str(uuid4())
    guest_sessions[session_id] = {
        "name": guest_name,
        "quiz_id": quiz_id,
        "current_index": 0,
        "score": 0
    }
    resp = RedirectResponse(url=f"/guest/quiz/{quiz_id}/question", status_code=302)
    resp.set_cookie("guest_session_id", session_id)
    return resp

@app.get("/guest/quiz/{quiz_id}/question", response_class=HTMLResponse)
def guest_question(quiz_id: int, request: Request, guest_session_id: Optional[str] = Cookie(None), db: Session = Depends(get_db)):

    session = guest_sessions.get(guest_session_id)
    if not session or session["quiz_id"] != quiz_id:
        return HTMLResponse("Session expired or invalid.", status_code=400)

    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if session["current_index"] >= len(quiz.questions):
        return RedirectResponse(url=f"/guest/quiz/{quiz_id}/result", status_code=302)

    question = quiz.questions[session["current_index"]]
    correct_count = db.query(UserAnswer).filter_by(
        quiz_id=quiz_id,
        question_id=question.id,
        is_correct=True
    ).count()

    return templates.TemplateResponse("question.html", {
        "request": request,
        "quiz": quiz,
        "question": question,
        "progress": session,
        "correct_count": correct_count
    })


@app.post("/guest/quiz/{quiz_id}/answer")
def guest_answer(quiz_id: int, question_id: int = Form(...), answer: str = Form(...), guest_session_id: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    session = guest_sessions.get(guest_session_id)
    if not session or session["quiz_id"] != quiz_id:
        return HTMLResponse("Session expired or invalid.", status_code=400)

    question = db.query(Question).filter_by(id=question_id).first()
    if question and answer.strip().lower() == question.answer.strip().lower():
        session["score"] += 1
    session["current_index"] += 1

    # Добавим гостя в список ответивших
    key = (quiz_id, question_id)
    if key not in answered_users:
        answered_users[key] = set()
    answered_users[key].add(session["name"])

    return RedirectResponse(url=f"/guest/quiz/{quiz_id}/question", status_code=302)

@app.get("/guest/quiz/{quiz_id}/result", response_class=HTMLResponse)
def guest_result(quiz_id: int, request: Request, guest_session_id: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    session = guest_sessions.get(guest_session_id)
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()

    if not session:
        return HTMLResponse("Session expired.", status_code=400)

    # Получаем текущего гостя
    leaderboard = [{
        "name": session["name"],
        "score": session["score"],
        "duration": 0.0  # У гостей нет времени прохождения
    }]

    
    all_progress = db.query(UserQuizProgress).filter_by(quiz_id=quiz_id).all()
    for p in all_progress:
        if not p.end_time:
            continue
        start = datetime.fromisoformat(p.start_time)
        end = datetime.fromisoformat(p.end_time)
        user = db.query(User).filter_by(id=p.user_id).first()
        leaderboard.append({
            "name": user.username if user else "Unknown",
            "score": p.score,
            "duration": (end - start).total_seconds()
        })

    leaderboard.sort(key=lambda x: (-x["score"], x["duration"]))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "score": session["score"],
        "total": len(quiz.questions),
        "leaderboard": leaderboard,
        "is_guest": True
    })


