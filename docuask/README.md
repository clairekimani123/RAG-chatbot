# DocuAsk — AI Document Assistant

DocuAsk is a full-stack RAG (Retrieval-Augmented Generation) application that lets users upload documents — PDFs, images, text files, and Word documents — and have AI-powered conversations about their content. Every answer is grounded in the uploaded document, not the model's general knowledge, which means no hallucinated facts and no made-up information.

**Live demo:** [ducoask.vercel.app](https://ducoask.vercel.app)
**Backend API:** [rag-chatbot-3yrs.onrender.com](https://rag-chatbot-3yrs.onrender.com)
**Author:** Claire Kimani — [GitHub](https://github.com/clairekimani123) · [Portfolio](https://portfolio-nine-xi-fc7wggw8ye.vercel.app)

---

## What it does

A user signs up, uploads a file, and starts asking questions about it. The system finds the most relevant parts of the document using semantic search, then generates an answer using only that context. Conversations are saved permanently, so users can return days later and continue where they left off — exactly like ChatGPT's conversation history, but scoped to their own private documents.

---

## How it works

```
Upload phase:
  File → extract text → split into chunks → generate embeddings → store in PostgreSQL (pgvector)

Question phase:
  Question → embed → find closest chunks (cosine similarity) → build prompt with context
           → send to Groq LLM → stream answer back → save to conversation history
```

The core idea is retrieval-augmented generation: instead of relying on what the AI model was trained on, every answer is generated using content retrieved directly from the user's own document at the moment the question is asked.

---

## Features

- **Multi-format support** — PDF, PNG/JPEG images (described via vision model), TXT, and DOCX files
- **Semantic search** — finds relevant content by meaning, not just keyword matching
- **Streaming responses** — answers appear word by word in real time
- **Persistent conversation history** — every conversation is saved to the database and can be resumed
- **User authentication** — JWT-based auth, each user only sees their own documents and conversations
- **Source citations** — every answer shows which part of the document it was drawn from
- **Three-theme dark UI** — black, lavender, and purple themes, switchable instantly
- **Fully responsive** — works on desktop and mobile with a collapsible sidebar

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI (Python) |
| Database | PostgreSQL with pgvector extension |
| ORM | SQLAlchemy |
| AI inference | Groq (Llama 3.1 for chat, Llama 4 Scout for vision) |
| Authentication | JWT (python-jose) + bcrypt password hashing |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |
| Frontend framework | React 18 + TypeScript |
| Build tool | Vite |
| HTTP client | Axios |
| Backend hosting | Render |
| Frontend hosting | Vercel |

---

## Architecture

### Database schema

Four core tables:

- **users** — registered accounts with hashed passwords
- **documents** — uploaded files with metadata (type, chunk count, owner)
- **chunks** — text segments with their vector embeddings, linked to a document
- **conversations** — one per chat session, linked to a user and a document
- **conversation_messages** — every message ever sent, permanently stored

### Backend structure

```
docuask/
├── app/
│   ├── main.py                      # FastAPI entry point, CORS, router registration
│   ├── core/
│   │   └── config.py                # environment-driven settings
│   ├── db/
│   │   ├── database.py              # connection, session management, pgvector init
│   │   └── models.py                # User, Document, Chunk, Conversation, ConversationMessage
│   ├── services/
│   │   ├── auth_service.py          # password hashing, JWT creation/verification
│   │   ├── document_service.py      # ingestion pipeline for all file types
│   │   └── conversation_service.py  # RAG query pipeline + persistent history
│   └── api/
│       ├── auth.py                  # /api/auth/register, /login, /me
│       ├── documents.py             # upload, list, delete
│       └── chat.py                  # ask, stream, conversation history
├── requirements.txt
└── README.md
```

### Frontend structure

```
docuask-ui/
├── src/
│   ├── main.tsx
│   ├── App.tsx                  # auth gate, global state, responsive layout
│   ├── AuthContext.tsx          # login/register/logout, token persistence
│   ├── AuthPage.tsx             # sign in / sign up screen
│   ├── index.css                # theme system via CSS variables
│   ├── api/
│   │   └── client.ts            # all backend communication, streaming parser
│   ├── types/
│   │   └── index.ts             # shared TypeScript interfaces
│   └── components/
│       ├── Sidebar.tsx          # upload, document list, conversation history, theme switcher
│       ├── ChatArea.tsx         # message state, streaming logic
│       ├── MessageBubble.tsx    # individual message rendering
│       └── ChatInput.tsx        # auto-resizing input with keyboard shortcuts
```

---

## Local development setup

### Prerequisites

- Python 3.11
- Node.js 18+
- PostgreSQL with the pgvector extension
- A free [Groq API key](https://console.groq.com)

### Backend

```bash
git clone https://github.com/clairekimani123/RAG-chatbot
cd RAG-chatbot/docuask

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY, DATABASE_URL, SECRET_KEY

# Enable pgvector on your local Postgres
sudo -i -u postgres psql
CREATE DATABASE docuask;
\c docuask
CREATE EXTENSION IF NOT EXISTS vector;
\q

uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd ../docuask-ui
npm install

# create .env.development
echo "VITE_API_URL=http://localhost:8000" > .env.development

npm run dev
```

UI runs at `http://localhost:5173`.

---

## Environment variables

### Backend (`.env`)

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key from console.groq.com |
| `DATABASE_URL` | PostgreSQL connection string |
| `SECRET_KEY` | Random string used to sign JWT tokens |
| `ENVIRONMENT` | `development` or `production` |

### Frontend

| Variable | Description |
|---|---|
| `VITE_API_URL` | Base URL of the backend API |

---

## API reference

### Auth

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create account, returns JWT |
| POST | `/api/auth/login` | Login, returns JWT |
| GET | `/api/auth/me` | Get current user info |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload and process a file |
| GET | `/api/documents` | List the user's documents |
| GET | `/api/documents/{id}` | Get one document |
| DELETE | `/api/documents/{id}` | Delete a document and its chunks |

### Chat

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat/ask` | Ask a question, returns full answer |
| POST | `/api/chat/stream` | Ask a question, streams the answer |
| GET | `/api/chat/conversations` | List all conversations |
| GET | `/api/chat/conversations/document/{id}` | Conversations for one document |
| GET | `/api/chat/conversations/{id}/messages` | Full message history |
| DELETE | `/api/chat/conversations/{id}` | Delete a conversation |

All endpoints except `/register` and `/login` require a `Bearer` token in the `Authorization` header.

---

## Design decisions and tradeoffs

**Why Groq instead of OpenAI** — Groq offers free, fast inference, which made it possible to build and run this project without ongoing API costs during development.

**Why a hash-based embedding fallback in production** — Render's free tier provides 512MB of RAM, which isn't enough to run `sentence-transformers` with PyTorch. The production deployment uses a lightweight deterministic embedding function instead. This keeps the app running on free infrastructure; on a paid tier, swapping back to a proper sentence-transformer model would improve semantic search quality.

**Why pgvector over a dedicated vector database** — Since PostgreSQL was already the primary database, adding the pgvector extension avoided introducing a second piece of infrastructure (like Pinecone or Weaviate) for a project of this scale.

---

## Roadmap

- Swap back to a proper embedding model once infrastructure supports it
- OCR support for scanned PDFs
- Multi-document conversations (ask questions across several files at once)
- Shareable conversation links
- Usage analytics dashboard

---

## License

This project was built as a personal learning project and portfolio piece.