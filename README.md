# 🔬 Nexus Research Agent

An AI-powered multi-agent research assistant built with React 19. It combines live web search, document-based RAG, and a 5-agent debate system to deliver well-rounded, confidence-scored answers — all running directly in the browser with no backend required.

---

## ✨ Features

### 🤖 5 Specialized AI Agents
Each query is analyzed by a panel of agents with distinct personalities:


### 🌐 Live Web Search
Integrates with the **Tavily API** to fetch real-time search results and ground answers in current information.

### 📄 Document RAG (Retrieval-Augmented Generation)
Upload your own files and the agent retrieves relevant context using a custom **TF-IDF engine**:
- Supports PDF, images (JPG, PNG, WebP), plain text, Markdown, CSV, and JSON
- Text is chunked (600 words, 80-word overlap) and ranked by relevance per query

### 📊 Confidence Scoring
Every response includes a confidence score calculated from:
- Number and quality of web sources
- Uploaded document coverage
- Debate consensus strength
- Response length and completeness

### 🙋 Human-in-the-Loop (HITL)
Before running complex research, the agent can pause and ask you to confirm or refine the query — keeping you in control.

### 💬 Session Management
Multiple chat sessions with persistent history stored in `localStorage`.

### ⚙️ Fully Configurable
Control the model, number of debating agents, temperature, and toggle features (web search, RAG, debate, HITL) on the fly.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Create React App |
| LLM | [Groq API](https://console.groq.com) — `llama-3.3-70b-versatile` |
| Web Search | [Tavily API](https://tavily.com) |
| RAG Engine | Custom TF-IDF (no external vector DB) |
| Styling | CSS-in-JS (inline styles + injected `<style>`) |
| Fonts | DM Sans, Fraunces, JetBrains Mono |

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- A free [Groq API key](https://console.groq.com)
- A free [Tavily API key](https://tavily.com) *(optional, for web search)*

### Installation

```bash
# Clone the repository
git clone https://github.com/IswaryaRama/nexus-agent.git
cd nexus-agent

# Install dependencies
npm install

# Start the development server
npm start
```

The app will open at **http://localhost:3000**

### Adding API Keys

1. Click the ⚙️ **Settings** icon inside the app
2. Enter your **Groq API Key**
3. Enter your **Tavily API Key** *(optional)*
4. Save — you're ready to go!

> API keys are stored only in your browser's `localStorage` and never sent anywhere except the respective API endpoints.

---

## 📁 Project Structure

```
nexus-agent/
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── manifest.json
├── src/
│   ├── App.js          # All app logic and UI (1438 lines)
│   ├── App.css
│   ├── index.js
│   └── index.css
├── package.json
└── README.md
```

---

## 🎮 How to Use

1. **Ask a question** — type any research query in the chat box
2. **Toggle features** — use the toolbar to enable/disable Web Search, RAG, Debate, or HITL
3. **Upload documents** — drag & drop files to give the agent private context
4. **Review the debate** — expand the agent panel to see how each agent responded
5. **Rate responses** — use the star rating and correction box to give feedback

---

## 📦 Available Scripts

```bash
npm start       # Run in development mode
npm run build   # Build for production
npm test        # Run tests
