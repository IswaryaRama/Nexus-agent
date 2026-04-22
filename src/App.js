import { useState, useEffect, useRef, useCallback } from "react";

// ── Constants & Config ──────────────────────────────────────────────────────
const CONFIG = {
  APP_NAME: "Nexus Research Agent",
  DEFAULT_MODEL: "llama-3.3-70b-versatile",
  OPENAI_BASE_URL: "https://api.groq.com/openai/v1",
  TAVILY_BASE_URL: "https://api.tavily.com",
  CHUNK_SIZE: 600,
  CHUNK_OVERLAP: 80,
  TOP_K_CHUNKS: 5,
};

const AGENTS = [
  { id: "factual", name: "Atlas", role: "Factual Analyst", color: "#00d4ff", emoji: "🔍",
    personality: "You are Atlas, a rigorous factual analyst. You verify claims with precision, cite specific evidence, and clearly distinguish facts from opinions. You are direct, accurate, and methodical.", order: 1 },
  { id: "skeptical", name: "Sceptron", role: "Skeptical Reviewer", color: "#ff6b9d", emoji: "⚠",
    personality: "You are Sceptron, a critical thinker who challenges assumptions. You identify weaknesses in arguments, highlight missing context, potential biases, and alternative interpretations.", order: 2 },
  { id: "synthesizer", name: "Nexus", role: "Synthesizer", color: "#00ff88", emoji: "🧬",
    personality: "You are Nexus, an expert synthesizer. You integrate multiple perspectives, find common ground, and produce nuanced conclusions that acknowledge complexity.", order: 3 },
  { id: "devils_advocate", name: "Contra", role: "Devil's Advocate", color: "#ffb020", emoji: "🎭",
    personality: "You are Contra, the devil's advocate. You argue the opposing viewpoint to stress-test conclusions.", order: 4 },
  { id: "domain_expert", name: "Sage", role: "Domain Expert", color: "#a78bfa", emoji: "📚",
    personality: "You are Sage, a deep domain expert providing specialized knowledge and technical depth.", order: 5 },
];

const STOP_WORDS = new Set([
  "the","a","an","and","or","but","in","on","at","to","for","of","with","is","are",
  "was","were","be","been","have","has","had","do","does","did","will","would",
  "could","should","may","might","that","this","these","those","it","its","i","we",
  "you","he","she","they","my","our","your","his","her","their","not","no","nor",
  "so","yet","both","either","neither","from","by","up","about","into","through",
  "can","all","each","more","than","also","when","where","which","as","if","then",
  "because","while","although","how","what","who","any","some","such","same","other","only",
]);

// ── Utility Functions ────────────────────────────────────────────────────────
const generateId = () => Date.now().toString(36) + Math.random().toString(36).substring(2, 7);
const truncate = (str, n) => str && str.length > n ? str.substring(0, n) + "…" : str;
const formatTime = (date) => new Date(date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const formatFileSize = (bytes) => {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / (1024 * 1024)).toFixed(1) + "MB";
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const getFileIcon = (filename) => {
  const ext = filename.split(".").pop().toLowerCase();
  const icons = { pdf: "📄", png: "🖼", jpg: "🖼", jpeg: "🖼", gif: "🖼", webp: "🖼", txt: "📝", md: "📝", csv: "📊", json: "📋" };
  return icons[ext] || "📁";
};

// ── Markdown Renderer ────────────────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return "";
  const escapeHtml = (t) => String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;");
  let html = escapeHtml(text);
  html = html.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => `<pre><code${lang ? ` class="language-${lang}"` : ""}>${code.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/~~(.+?)~~/g, "<del>$1</del>");
  html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");
  html = html.replace(/^---+$/gm, "<hr>");
  html = html.replace(/^[*\-] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/\[(\d+)\]/g, '<sup><a class="source-cite" title="Source $1">[$1]</a></sup>');
  html = html.replace(/\n\n+/g, "</p><p>");
  html = `<p>${html}</p>`;
  html = html.replace(/<p>(<h[1-6]>)/g,"$1").replace(/(<\/h[1-6]>)<\/p>/g,"$1")
             .replace(/<p>(<pre>)/g,"$1").replace(/(<\/pre>)<\/p>/g,"$1")
             .replace(/<p>(<ul>)/g,"$1").replace(/(<\/ul>)<\/p>/g,"$1")
             .replace(/<p>(<hr>)/g,"$1").replace(/<p><\/p>/g,"")
             .replace(/\n/g,"<br>");
  return html;
}

// ── RAG / TF-IDF ─────────────────────────────────────────────────────────────
function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s]/g," ").split(/\s+/).filter((t) => t.length > 2 && !STOP_WORDS.has(t));
}

function buildTFIDF(texts) {
  const N = texts.length;
  const df = {};
  const tfs = [];
  texts.forEach((text) => {
    const terms = tokenize(text);
    const freq = {};
    terms.forEach((t) => { freq[t] = (freq[t] || 0) + 1; });
    tfs.push(freq);
    Object.keys(freq).forEach((t) => { df[t] = (df[t] || 0) + 1; });
  });
  return { df, tfs, N };
}

function tfidfScore(queryTerms, docFreq, totalDocs, docFreqMap) {
  let score = 0;
  queryTerms.forEach((term) => {
    if (docFreq[term]) {
      const tf = docFreq[term];
      const idf = Math.log(totalDocs / (1 + (docFreqMap[term] || 0)));
      score += tf * idf;
    }
  });
  return score;
}

function chunkText(text, chunkSize = 600, overlap = 80) {
  if (!text || text.length === 0) return [];
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunkWords = words.slice(i, i + chunkSize);
    if (chunkWords.length < 10) continue;
    chunks.push({ text: chunkWords.join(" "), startWord: i, endWord: Math.min(i + chunkSize, words.length), wordCount: chunkWords.length });
    if (i + chunkSize >= words.length) break;
  }
  return chunks;
}

async function retrieveRelevantChunks(documents, query, topK = CONFIG.TOP_K_CHUNKS) {
  if (!documents.length) return [];
  const allChunks = [];
  documents.forEach((doc) => {
    (doc.chunks || []).forEach((chunk, i) => {
      allChunks.push({ text: chunk.text, docId: doc.id, docName: doc.name, chunkIndex: i });
    });
  });
  if (!allChunks.length) return [];
  const texts = allChunks.map((c) => c.text);
  const { df, tfs, N } = buildTFIDF(texts);
  const queryTerms = tokenize(query);
  const scored = allChunks.map((chunk, i) => ({ ...chunk, score: tfidfScore(queryTerms, tfs[i], N, df) }));
  const sorted = scored.sort((a, b) => b.score - a.score);
  const result = [];
  const docCounts = {};
  for (const chunk of sorted) {
    if (result.length >= topK) break;
    const dc = docCounts[chunk.docId] || 0;
    if (dc >= 2) continue;
    result.push(chunk);
    docCounts[chunk.docId] = dc + 1;
  }
  return result.filter((r) => r.score > 0);
}

function formatRagContext(chunks) {
  if (!chunks || !chunks.length) return "";
  return chunks.map((chunk, i) => `[Excerpt ${i + 1}] Source: ${chunk.docName} (chunk ${chunk.chunkIndex + 1})\n${chunk.text}`).join("\n\n---\n\n");
}

// ── File Processing ───────────────────────────────────────────────────────────
async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = () => reject(new Error("File read failed"));
    reader.readAsText(file);
  });
}

async function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      const base64 = dataUrl.split(",")[1];
      resolve({ base64, mimeType: file.type, dataUrl });
    };
    reader.onerror = () => reject(new Error("File read failed"));
    reader.readAsDataURL(file);
  });
}

async function processFile(file, settings, onProgress) {
  const ext = file.name.split(".").pop().toLowerCase();
  const isImage = file.type.startsWith("image/");
  const isPdf = file.type === "application/pdf";
  const isCsv = ext === "csv";
  const isJson = ext === "json";

  onProgress(10, `Reading ${file.name}…`);
  let extractedContent = "";

  if (isImage) {
    onProgress(20, `OCR processing ${file.name}…`);
    const { base64, mimeType } = await readFileAsBase64(file);
    if (settings.groqApiKey) {
      try {
        extractedContent = await callGroq({ prompt: "Perform OCR on this image. Extract ALL text.", images: [{ base64, mimeType }], settings, temperature: 0.1, maxTokens: 4096, model: "meta-llama/llama-4-scout-17b-16e-instruct" });
      } catch (e) {
        extractedContent = `[Image: ${file.name}]\n[OCR failed: ${e.message}]`;
      }
    } else {
      extractedContent = `[Image: ${file.name}]\n[OCR requires API key]`;
    }
  } else if (isCsv) {
    onProgress(30, `Parsing CSV…`);
    const raw = await readFileAsText(file);
    extractedContent = `# ${file.name}\n\n${raw}`;
  } else if (isJson) {
    onProgress(30, `Parsing JSON…`);
    const raw = await readFileAsText(file);
    try {
      const parsed = JSON.parse(raw);
      extractedContent = `# ${file.name}\n\`\`\`json\n${JSON.stringify(parsed, null, 2).substring(0, 20000)}\n\`\`\``;
    } catch { extractedContent = raw; }
  } else {
    onProgress(30, `Reading ${file.name}…`);
    extractedContent = await readFileAsText(file);
  }

  onProgress(60, `Chunking ${file.name}…`);
  const chunks = chunkText(extractedContent, CONFIG.CHUNK_SIZE, CONFIG.CHUNK_OVERLAP);
  onProgress(100, `Done — ${chunks.length} chunks`);

  return {
    id: generateId(),
    name: file.name,
    size: file.size,
    content: extractedContent.substring(0, 50000),
    chunks: chunks.slice(0, 100),
    addedAt: new Date().toISOString(),
  };
}

// ── Groq API ─────────────────────────────────────────────────────────────────
async function callGroq({ prompt, systemPrompt = "", history = [], maxTokens = 2048, temperature = 0.7, images = [], model = null, settings }) {
  if (!settings.groqApiKey) throw new Error("Groq API key not configured. Get your free key at console.groq.com");
  const modelId = model || settings.model || CONFIG.DEFAULT_MODEL;
  const messages = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  for (const turn of history) messages.push({ role: turn.role, content: turn.content });

  const isVisionModel = modelId.includes("llava") || modelId.includes("scout") || modelId.includes("maverick");
  if (images.length > 0 && isVisionModel) {
    const userContent = images.map((img) => ({ type: "image_url", image_url: { url: `data:${img.mimeType};base64,${img.base64}` } }));
    userContent.push({ type: "text", text: prompt });
    messages.push({ role: "user", content: userContent });
  } else {
    messages.push({ role: "user", content: prompt });
  }

  const response = await fetch(`${CONFIG.OPENAI_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${settings.groqApiKey}` },
    body: JSON.stringify({ model: modelId, messages, max_tokens: maxTokens, temperature }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(`Groq API: ${errData?.error?.message || `error ${response.status}`}`);
  }
  const data = await response.json();
  const choice = data.choices?.[0];
  if (!choice) throw new Error("No response from Groq API");
  return choice.message?.content || "";
}

// ── Web Search ────────────────────────────────────────────────────────────────
async function performWebSearch(query, settings, maxResults = 5) {
  if (settings.tavilyApiKey) {
    try {
      const response = await fetch(`${CONFIG.TAVILY_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${settings.tavilyApiKey}` },
        body: JSON.stringify({ query, search_depth: "basic", max_results: maxResults, include_answer: true }),
      });
      if (response.ok) {
        const data = await response.json();
        const results = (data.results || []).map((r) => ({ title: r.title || "Untitled", url: r.url || "", snippet: r.content || r.snippet || "", score: r.score || 0.5, source: "tavily" }));
        if (data.answer) results.unshift({ title: "📡 Web Search Answer", url: "", snippet: data.answer, score: 1.0, source: "tavily_answer" });
        return results;
      }
    } catch (e) { console.warn("Tavily failed:", e.message); }
  }

  // Fallback: Groq knowledge simulation
  if (!settings.groqApiKey) return [];
  try {
    const response = await callGroq({
      prompt: `Act as a web search engine for: "${query}". Provide ${maxResults} realistic results. Return ONLY valid JSON array: [{"title":"...","url":"...","snippet":"..."}]`,
      temperature: 0.5, maxTokens: 1500, settings,
    });
    const match = response.match(/\[[\s\S]*\]/);
    if (!match) return [];
    return JSON.parse(match[0]).slice(0, maxResults).map((r) => ({ ...r, source: "groq_knowledge", score: 0.7 }));
  } catch { return []; }
}

// ── Multi-Agent Debate ────────────────────────────────────────────────────────
function analyzeSentiment(text) {
  const lower = text.toLowerCase();
  const agreeWords = ["agree","correct","accurate","well-stated","solid","strong","valid","sound","largely correct"];
  const disagreeWords = ["incorrect","inaccurate","wrong","disagree","flawed","misleading","missing","fails to","overlooks","insufficient"];
  const ag = agreeWords.filter((w) => lower.includes(w)).length;
  const dg = disagreeWords.filter((w) => lower.includes(w)).length;
  if (ag > dg + 1) return "agree";
  if (dg > ag) return "disagree";
  return "neutral";
}

function assessDebateStrength(votes) {
  const total = votes.agree + votes.disagree + votes.neutral;
  if (!total) return "none";
  const ar = votes.agree / total;
  const dr = votes.disagree / total;
  if (ar >= 0.7) return "strong";
  if (ar >= 0.5) return "moderate";
  if (dr >= 0.5) return "weak";
  return "mixed";
}

async function runMultiAgentDebate({ query, initialResponse, ragContext, webResults, numAgents, settings, onAgentUpdate }) {
  const selectedAgents = AGENTS.slice(0, numAgents);
  const debateTurns = [];
  const votes = { agree: 0, disagree: 0, neutral: 0 };

  const contextSummary = [
    ragContext ? `Document sources: ${truncate(ragContext, 500)}` : "",
    webResults.length > 0 ? `Web sources:\n${webResults.slice(0, 3).map((r, i) => `[${i+1}] ${r.title}: ${truncate(r.snippet || "", 150)}`).join("\n")}` : "",
  ].filter(Boolean).join("\n\n");

  for (const agent of selectedAgents) {
    if (onAgentUpdate) onAgentUpdate(agent.id, "thinking");
    try {
      const prevDebate = debateTurns.length > 0
        ? `\n\nPrevious analyses:\n${debateTurns.map((t) => `${t.agent.name}: ${truncate(t.response, 200)}`).join("\n")}`
        : "";
      const prompt = `## Query\n${query}\n\n## Initial Response\n${truncate(initialResponse, 800)}\n\n## Context\n${contextSummary || "None."}${prevDebate}\n\n## Task\nAs ${agent.name} (${agent.role}), critically analyze the initial response. Focus on: accuracy, completeness, nuance, and your verdict. Be concise (150-250 words).`;
      const agentResponse = await callGroq({ prompt, systemPrompt: agent.personality, temperature: 0.8, maxTokens: 400, settings });
      const sentiment = analyzeSentiment(agentResponse);
      votes[sentiment]++;
      const turn = { agent, response: agentResponse, sentiment, round: 1 };
      debateTurns.push(turn);
      if (onAgentUpdate) onAgentUpdate(agent.id, "done");
      await sleep(300);
    } catch (e) {
      debateTurns.push({ agent, response: `[${agent.name} unavailable: ${e.message}]`, sentiment: "neutral", round: 1, error: true });
      if (onAgentUpdate) onAgentUpdate(agent.id, "error");
    }
  }

  const synthesizer = selectedAgents.find((a) => a.id === "synthesizer") || selectedAgents[0];
  const debateSummary = debateTurns.filter((t) => !t.error).map((t) => `${t.agent.name}: ${truncate(t.response, 300)}`).join("\n\n");
  const agreeRatio = votes.agree / (votes.agree + votes.disagree + votes.neutral + 0.001);
  let consensus = "";
  try {
    consensus = await callGroq({
      prompt: `## Query\n${query}\n\n## Initial Response\n${truncate(initialResponse, 400)}\n\n## Debate\n${debateSummary}\n\nSynthesize a final consensus verdict (150-200 words). ${agreeRatio > 0.6 ? "Majority agreed." : "Agents had mixed views."}`,
      systemPrompt: synthesizer.personality, temperature: 0.6, maxTokens: 350, settings,
    });
  } catch (e) { consensus = `Consensus unavailable: ${e.message}`; }

  return { turns: debateTurns, consensus, votes, strength: assessDebateStrength(votes), numAgents };
}

// ── Confidence Score ─────────────────────────────────────────────────────────
function calculateConfidence({ hasWeb, numWeb, hasDoc, numDoc, debateStrength, responseLength }) {
  let score = 0.45;
  if (hasWeb) score += Math.min(0.20, numWeb * 0.05);
  if (hasDoc) score += Math.min(0.15, numDoc * 0.04);
  if (debateStrength === "strong") score += 0.18;
  else if (debateStrength === "moderate") score += 0.10;
  else if (debateStrength === "weak") score += 0.03;
  else if (debateStrength === "none") score -= 0.05;
  if (responseLength > 500) score += 0.05;
  if (responseLength > 1000) score += 0.03;
  return Math.min(0.97, Math.max(0.15, score));
}

// ─────────────────────────────────────────────────────────────────────────────
// REACT COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;1,9..144,300&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@300;400&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0b0f;
    --surface: #111318;
    --surface2: #181c24;
    --border: #1e2330;
    --border2: #252d3d;
    --text: #e8eaf0;
    --text-dim: #8892a4;
    --text-muted: #4a5568;
    --accent: #00d4ff;
    --accent2: #00ff88;
    --accent-red: #ff4757;
    --accent-gold: #ffb020;
    --accent-pink: #ff6b9d;
    --accent-violet: #a78bfa;
    --font-display: 'Fraunces', Georgia, serif;
    --font-body: 'DM Sans', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius: 10px;
    --sidebar-w: 260px;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--font-body); font-size: 14px; line-height: 1.6; overflow: hidden; height: 100vh; }

  /* Layout */
  .app { display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  .sidebar { width: var(--sidebar-w); min-width: var(--sidebar-w); background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; transition: transform 0.3s; }
  .sidebar-brand { display: flex; align-items: center; gap: 10px; padding: 18px 16px; border-bottom: 1px solid var(--border); }
  .brand-logo { width: 34px; height: 34px; background: linear-gradient(135deg, var(--accent), var(--accent2)); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-family: var(--font-display); font-size: 18px; color: #000; font-weight: 400; flex-shrink: 0; }
  .brand-name { font-family: var(--font-display); font-size: 17px; color: var(--text); }
  .brand-sub { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); }
  .sidebar-section { padding: 12px; border-bottom: 1px solid var(--border); }
  .sidebar-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--text-muted); font-family: var(--font-mono); margin-bottom: 8px; display: block; }
  .sidebar-label-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .doc-count-badge { background: var(--surface2); color: var(--accent); font-size: 10px; font-family: var(--font-mono); padding: 1px 6px; border-radius: 10px; border: 1px solid var(--border2); }

  .new-chat-btn { width: 100%; background: linear-gradient(135deg, var(--accent)18, var(--accent2)10); border: 1px solid var(--accent)40; color: var(--accent); padding: 8px 14px; border-radius: var(--radius); cursor: pointer; font-size: 13px; font-family: var(--font-body); display: flex; align-items: center; gap: 8px; transition: all 0.2s; }
  .new-chat-btn:hover { background: linear-gradient(135deg, var(--accent)30, var(--accent2)20); }

  .session-list { max-height: 200px; overflow-y: auto; }
  .session-item { display: flex; align-items: center; gap: 8px; padding: 7px 8px; border-radius: 7px; cursor: pointer; transition: background 0.15s; position: relative; }
  .session-item:hover { background: var(--surface2); }
  .session-item.active { background: var(--accent)12; border: 1px solid var(--accent)25; }
  .session-icon { font-size: 12px; flex-shrink: 0; }
  .session-title { font-size: 12px; color: var(--text-dim); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .session-delete { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 11px; padding: 2px 4px; opacity: 0; transition: opacity 0.15s; border-radius: 4px; }
  .session-item:hover .session-delete { opacity: 1; }
  .session-delete:hover { color: var(--accent-red); background: var(--accent-red)15; }

  .sidebar-docs-section { flex: 1; overflow-y: auto; }
  .doc-list { margin-bottom: 8px; }
  .doc-item { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 7px; background: var(--surface2); margin-bottom: 4px; }
  .doc-icon { font-size: 14px; flex-shrink: 0; }
  .doc-info { flex: 1; overflow: hidden; }
  .doc-name { font-size: 12px; color: var(--text); display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .doc-meta { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); }
  .doc-delete { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 11px; padding: 2px 4px; border-radius: 4px; }
  .doc-delete:hover { color: var(--accent-red); background: var(--accent-red)15; }

  .upload-zone { border: 1px dashed var(--border2); border-radius: var(--radius); padding: 16px 12px; text-align: center; cursor: pointer; transition: all 0.2s; }
  .upload-zone:hover, .upload-zone.drag-over { border-color: var(--accent); background: var(--accent)08; }
  .upload-icon { font-size: 20px; margin-bottom: 4px; }
  .upload-text { font-size: 12px; color: var(--text-dim); }
  .upload-hint { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); margin-top: 2px; }

  .upload-progress { margin-top: 8px; }
  .progress-info { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim); font-family: var(--font-mono); margin-bottom: 4px; }
  .progress-bar { background: var(--surface2); border-radius: 4px; height: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.3s; }

  .sidebar-footer { padding: 12px; border-top: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
  .settings-btn { background: none; border: 1px solid var(--border2); color: var(--text-dim); padding: 6px 12px; border-radius: 7px; cursor: pointer; font-size: 12px; font-family: var(--font-body); transition: all 0.2s; }
  .settings-btn:hover { border-color: var(--accent); color: var(--accent); }
  .sidebar-status { display: flex; align-items: center; gap: 6px; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent-red); }
  .status-dot.ready { background: var(--accent2); }
  .status-dot.processing { background: var(--accent-gold); animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .status-text { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); }

  /* Main */
  .main-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .topbar { display: flex; align-items: center; justify-content: space-between; padding: 0 20px; height: 52px; border-bottom: 1px solid var(--border); background: var(--surface)cc; backdrop-filter: blur(8px); flex-shrink: 0; }
  .topbar-name { font-family: var(--font-display); font-size: 15px; color: var(--text); }
  .agent-chips { display: flex; gap: 6px; }
  .agent-chip { font-size: 11px; font-family: var(--font-mono); padding: 3px 8px; border-radius: 20px; border: 1px solid; transition: opacity 0.3s; }
  .agent-chip.thinking { animation: pulse 0.8s ease-in-out infinite; }

  /* Chat */
  .chat-area { flex: 1; overflow-y: auto; padding: 20px; scroll-behavior: smooth; }
  .chat-area::-webkit-scrollbar { width: 4px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* Welcome */
  .welcome-screen { max-width: 640px; margin: 40px auto; text-align: center; }
  .welcome-logo { width: 60px; height: 60px; background: linear-gradient(135deg, var(--accent), var(--accent2)); border-radius: 16px; display: flex; align-items: center; justify-content: center; font-family: var(--font-display); font-size: 30px; color: #000; margin: 0 auto 20px; }
  .welcome-title { font-family: var(--font-display); font-size: 32px; font-weight: 300; color: var(--text); margin-bottom: 8px; }
  .welcome-title em { font-style: italic; color: var(--accent); }
  .welcome-subtitle { font-size: 14px; color: var(--text-muted); margin-bottom: 32px; }
  .welcome-features { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 32px; }
  .feature-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; display: flex; align-items: center; gap: 10px; text-align: left; }
  .feature-icon { font-size: 20px; flex-shrink: 0; }
  .feature-text { font-size: 12px; color: var(--text-dim); line-height: 1.4; }
  .starter-label { font-size: 12px; color: var(--text-muted); margin-bottom: 10px; }
  .starter-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  .starter-chip { background: var(--surface2); border: 1px solid var(--border2); color: var(--text-dim); padding: 7px 14px; border-radius: 20px; cursor: pointer; font-size: 12px; font-family: var(--font-body); transition: all 0.2s; }
  .starter-chip:hover { border-color: var(--accent); color: var(--accent); background: var(--accent)08; }

  /* Messages */
  .messages-container { max-width: 760px; margin: 0 auto; }
  .message-wrapper { margin-bottom: 24px; }
  .message-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .message-avatar { width: 28px; height: 28px; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0; }
  .avatar-user { background: var(--surface2); color: var(--text-dim); border: 1px solid var(--border2); }
  .avatar-ai { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #000; }
  .message-sender { font-size: 13px; font-weight: 500; color: var(--text); }
  .message-time { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); margin-left: 8px; }
  .message-body { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .message-wrapper.user .message-body { background: var(--surface2); }
  .message-text { font-size: 14px; line-height: 1.7; color: var(--text); }
  .message-text p { margin: 0 0 10px; }
  .message-text p:last-child { margin-bottom: 0; }
  .message-text h1,.message-text h2,.message-text h3,.message-text h4 { font-family: var(--font-display); font-weight: 400; color: var(--text); margin: 16px 0 8px; }
  .message-text h2 { font-size: 18px; } .message-text h3 { font-size: 16px; } .message-text h4 { font-size: 14px; color: var(--text-dim); }
  .message-text ul { padding-left: 20px; margin: 8px 0; }
  .message-text li { margin-bottom: 4px; }
  .message-text code { background: var(--surface2); border: 1px solid var(--border); padding: 1px 5px; border-radius: 4px; font-family: var(--font-mono); font-size: 12px; color: var(--accent); }
  .message-text pre { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 14px; overflow-x: auto; margin: 10px 0; }
  .message-text pre code { background: none; border: none; padding: 0; font-size: 13px; color: var(--text); }
  .message-text blockquote { border-left: 3px solid var(--accent); padding-left: 12px; color: var(--text-dim); margin: 8px 0; }
  .message-text a { color: var(--accent); text-decoration: none; }
  .message-text a:hover { text-decoration: underline; }
  .message-text strong { color: var(--text); font-weight: 600; }
  .message-text em { color: var(--text-dim); }
  .message-text table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
  .message-text th { background: var(--surface2); border: 1px solid var(--border2); padding: 6px 10px; text-align: left; color: var(--accent); font-family: var(--font-mono); font-size: 11px; text-transform: uppercase; }
  .message-text td { border: 1px solid var(--border); padding: 6px 10px; }
  .message-text hr { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
  .message-text del { color: var(--text-muted); }

  /* Typing indicator */
  .typing-indicator { display: flex; gap: 10px; margin-bottom: 24px; max-width: 760px; margin-left: auto; margin-right: auto; }
  .typing-avatar { width: 28px; height: 28px; border-radius: 7px; background: linear-gradient(135deg, var(--accent), var(--accent2)); display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; color: #000; flex-shrink: 0; margin-top: 2px; }
  .typing-bubble { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; flex: 1; }
  .agent-thinking-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
  .typing-dots { display: flex; gap: 4px; align-items: center; margin-bottom: 8px; }
  .typing-dots span { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: bounce 1.2s ease-in-out infinite; }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
  .processing-steps { display: flex; flex-direction: column; gap: 4px; }
  .processing-step { display: flex; align-items: center; gap: 8px; font-size: 12px; font-family: var(--font-mono); color: var(--text-muted); }
  .processing-step.active { color: var(--accent); }
  .processing-step.done { color: var(--accent2); }
  .processing-step.error { color: var(--accent-red); }
  .step-icon { font-size: 10px; }

  /* Confidence */
  .confidence-bar { display: flex; align-items: center; gap: 10px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
  .confidence-label { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); white-space: nowrap; }
  .confidence-track { flex: 1; background: var(--surface2); border-radius: 4px; height: 5px; overflow: hidden; }
  .confidence-fill { height: 100%; border-radius: 4px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }
  .confidence-value { font-size: 11px; font-family: var(--font-mono); white-space: nowrap; }

  /* Sources */
  .sources-section { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
  .sources-title { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); margin-bottom: 8px; }
  .sources-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .source-tag { display: flex; align-items: center; gap: 5px; background: var(--surface2); border: 1px solid var(--border2); border-radius: 6px; padding: 4px 8px; font-size: 11px; color: var(--text-dim); text-decoration: none; transition: all 0.15s; cursor: default; }
  a.source-tag { cursor: pointer; }
  a.source-tag:hover { border-color: var(--accent); color: var(--accent); }
  .source-tag-icon { font-size: 12px; }

  /* Debate */
  .debate-section { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }
  .debate-toggle-btn { background: var(--surface2); border: 1px solid var(--border2); color: var(--text-dim); padding: 7px 14px; border-radius: 7px; cursor: pointer; font-size: 12px; font-family: var(--font-body); width: 100%; text-align: left; transition: all 0.15s; }
  .debate-toggle-btn:hover { border-color: var(--accent); color: var(--accent); }
  .debate-panel { margin-top: 10px; border: 1px solid var(--border2); border-radius: var(--radius); overflow: hidden; }
  .debate-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; background: var(--surface2); border-bottom: 1px solid var(--border); }
  .debate-header-title { font-size: 12px; font-family: var(--font-mono); color: var(--text-dim); }
  .debate-round-badge { font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); }
  .debate-turns { padding: 10px 14px; display: flex; flex-direction: column; gap: 12px; }
  .debate-turn { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .debate-turn-header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--surface2); flex-wrap: wrap; }
  .agent-name-badge { font-size: 11px; font-family: var(--font-mono); padding: 2px 8px; border-radius: 12px; border: 1px solid; }
  .agent-role { font-size: 11px; color: var(--text-muted); }
  .debate-turn-text { padding: 10px 12px; font-size: 13px; line-height: 1.6; color: var(--text-dim); white-space: pre-wrap; }
  .debate-consensus { margin: 0 14px 14px; padding: 12px; background: var(--accent)08; border: 1px solid var(--accent)25; border-radius: 8px; }
  .debate-consensus-label { font-size: 11px; font-family: var(--font-mono); color: var(--accent); margin-bottom: 6px; }
  .debate-consensus-text { font-size: 13px; line-height: 1.6; color: var(--text-dim); }

  /* Rating */
  .rating-section { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .rating-label { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); }
  .star-rating { display: flex; gap: 3px; }
  .star-btn { background: none; border: none; font-size: 16px; cursor: pointer; color: var(--text-muted); transition: color 0.15s; padding: 0 1px; line-height: 1; }
  .star-btn.active, .star-btn:hover { color: var(--accent-gold); }
  .correction-toggle { background: none; border: 1px solid var(--border2); color: var(--text-muted); padding: 3px 10px; border-radius: 5px; cursor: pointer; font-size: 11px; font-family: var(--font-body); transition: all 0.15s; }
  .correction-toggle:hover { border-color: var(--accent); color: var(--accent); }
  .correction-box { margin-top: 8px; width: 100%; }
  .correction-textarea { width: 100%; background: var(--surface2); border: 1px solid var(--border2); color: var(--text); padding: 8px 10px; border-radius: 7px; font-size: 12px; font-family: var(--font-body); resize: vertical; min-height: 60px; }
  .correction-textarea:focus { outline: none; border-color: var(--accent); }
  .correction-submit { margin-top: 6px; background: var(--accent)15; border: 1px solid var(--accent)40; color: var(--accent); padding: 5px 14px; border-radius: 5px; cursor: pointer; font-size: 12px; font-family: var(--font-body); transition: all 0.15s; }
  .correction-submit:hover { background: var(--accent)25; }

  /* Input area */
  .input-area { padding: 12px 20px 16px; border-top: 1px solid var(--border); background: var(--surface)cc; backdrop-filter: blur(8px); flex-shrink: 0; }
  .input-options { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
  .option-tag { background: var(--surface2); border: 1px solid var(--border2); color: var(--text-muted); padding: 4px 12px; border-radius: 16px; cursor: pointer; font-size: 12px; font-family: var(--font-body); transition: all 0.15s; }
  .option-tag.active { border-color: var(--accent)50; color: var(--accent); background: var(--accent)10; }
  .option-tag:hover { border-color: var(--border2); color: var(--text-dim); }
  .option-tag.active:hover { border-color: var(--accent); }
  .input-box { background: var(--surface2); border: 1px solid var(--border2); border-radius: 12px; transition: border-color 0.2s; }
  .input-box:focus-within { border-color: var(--accent)60; }
  .input-row { display: flex; align-items: flex-end; padding: 8px 12px; gap: 10px; }
  .chat-textarea { flex: 1; background: none; border: none; color: var(--text); font-size: 14px; font-family: var(--font-body); line-height: 1.5; resize: none; outline: none; max-height: 200px; min-height: 24px; }
  .chat-textarea::placeholder { color: var(--text-muted); }
  .send-btn { width: 34px; height: 34px; background: linear-gradient(135deg, var(--accent), var(--accent2)); border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: opacity 0.2s; }
  .send-btn:hover { opacity: 0.85; }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .input-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 6px; }
  .input-hint { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); }
  .model-badge { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); background: var(--surface2); border: 1px solid var(--border); padding: 2px 8px; border-radius: 10px; }

  /* Modal overlay */
  .modal-overlay { position: fixed; inset: 0; background: #000a; backdrop-filter: blur(4px); z-index: 100; display: flex; align-items: center; justify-content: center; }

  /* Settings modal */
  .settings-modal { background: var(--surface); border: 1px solid var(--border2); border-radius: 14px; width: 480px; max-width: 95vw; max-height: 90vh; display: flex; flex-direction: column; }
  .settings-header { display: flex; align-items: center; justify-content: space-between; padding: 18px 20px; border-bottom: 1px solid var(--border); }
  .settings-header h3 { font-family: var(--font-display); font-size: 18px; font-weight: 300; }
  .close-btn { background: none; border: 1px solid var(--border2); color: var(--text-dim); width: 28px; height: 28px; border-radius: 6px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center; }
  .close-btn:hover { border-color: var(--accent-red); color: var(--accent-red); }
  .settings-body { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .settings-section { margin-bottom: 20px; }
  .settings-section h4 { font-size: 12px; color: var(--text-dim); font-family: var(--font-mono); margin-bottom: 12px; letter-spacing: 0.5px; }
  .field-group { margin-bottom: 12px; }
  .field-group label { display: block; font-size: 12px; color: var(--text-dim); margin-bottom: 5px; }
  .required { color: var(--accent-red); }
  .field-group input, .field-group select { width: 100%; background: var(--surface2); border: 1px solid var(--border2); color: var(--text); padding: 8px 12px; border-radius: 7px; font-size: 13px; font-family: var(--font-body); outline: none; transition: border-color 0.2s; }
  .field-group input:focus, .field-group select:focus { border-color: var(--accent)60; }
  .field-group select option { background: var(--surface); }
  .field-hint { font-size: 11px; color: var(--text-muted); margin-top: 3px; display: block; }
  .field-hint a { color: var(--accent); }
  .toggle-group { display: flex; align-items: center; justify-content: space-between; }
  .toggle-switch { position: relative; display: inline-block; width: 40px; height: 22px; cursor: pointer; }
  .toggle-switch input { opacity: 0; width: 0; height: 0; }
  .toggle-slider { position: absolute; inset: 0; background: var(--surface2); border: 1px solid var(--border2); border-radius: 22px; transition: 0.2s; }
  .toggle-slider::before { content: ""; position: absolute; width: 16px; height: 16px; left: 2px; top: 2px; background: var(--text-muted); border-radius: 50%; transition: 0.2s; }
  input:checked + .toggle-slider { background: var(--accent)30; border-color: var(--accent)60; }
  input:checked + .toggle-slider::before { transform: translateX(18px); background: var(--accent); }
  .settings-footer { padding: 14px 20px; border-top: 1px solid var(--border); }
  .btn-save { width: 100%; background: linear-gradient(135deg, var(--accent)25, var(--accent2)15); border: 1px solid var(--accent)40; color: var(--accent); padding: 10px; border-radius: 8px; cursor: pointer; font-size: 14px; font-family: var(--font-body); transition: all 0.2s; }
  .btn-save:hover { background: linear-gradient(135deg, var(--accent)35, var(--accent2)25); }

  /* HITL modal */
  .hitl-modal { background: var(--surface); border: 1px solid var(--accent)30; border-radius: 14px; width: 460px; max-width: 95vw; padding: 24px; }
  .hitl-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .hitl-icon { font-size: 20px; }
  .hitl-header h3 { font-family: var(--font-display); font-size: 18px; font-weight: 300; }
  .hitl-desc { font-size: 13px; color: var(--text-dim); margin-bottom: 14px; line-height: 1.5; }
  .hitl-context { margin-bottom: 14px; }
  .hitl-context label { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); display: block; margin-bottom: 5px; }
  .hitl-query-text { background: var(--surface2); border: 1px solid var(--border2); border-radius: 7px; padding: 8px 12px; font-size: 13px; color: var(--text-dim); font-family: var(--font-mono); }
  .hitl-modify-section { margin-bottom: 18px; }
  .hitl-modify-section label { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); display: block; margin-bottom: 5px; }
  .hitl-textarea { width: 100%; background: var(--surface2); border: 1px solid var(--border2); color: var(--text); padding: 8px 12px; border-radius: 7px; font-size: 13px; font-family: var(--font-mono); resize: vertical; min-height: 60px; outline: none; }
  .hitl-textarea:focus { border-color: var(--accent)60; }
  .hitl-actions { display: flex; gap: 10px; justify-content: flex-end; }
  .btn-deny { background: var(--accent-red)15; border: 1px solid var(--accent-red)40; color: var(--accent-red); padding: 8px 16px; border-radius: 7px; cursor: pointer; font-size: 13px; font-family: var(--font-body); transition: all 0.15s; }
  .btn-deny:hover { background: var(--accent-red)25; }
  .btn-modify { background: var(--accent-gold)15; border: 1px solid var(--accent-gold)40; color: var(--accent-gold); padding: 8px 16px; border-radius: 7px; cursor: pointer; font-size: 13px; font-family: var(--font-body); transition: all 0.15s; }
  .btn-modify:hover { background: var(--accent-gold)25; }
  .btn-approve { background: var(--accent2)15; border: 1px solid var(--accent2)40; color: var(--accent2); padding: 8px 16px; border-radius: 7px; cursor: pointer; font-size: 13px; font-family: var(--font-body); transition: all 0.15s; }
  .btn-approve:hover { background: var(--accent2)25; }

  /* Toast */
  .toast-container { position: fixed; bottom: 80px; right: 20px; display: flex; flex-direction: column; gap: 8px; z-index: 200; }
  .toast { display: flex; align-items: center; gap: 8px; background: var(--surface2); border: 1px solid var(--border2); border-radius: 8px; padding: 10px 14px; font-size: 13px; color: var(--text); max-width: 360px; box-shadow: 0 4px 20px #0008; animation: slideIn 0.3s ease; }
  .toast.success { border-color: var(--accent2)40; }
  .toast.error { border-color: var(--accent-red)40; }
  .toast.warning { border-color: var(--accent-gold)40; }
  .toast-icon { font-size: 14px; flex-shrink: 0; }
  @keyframes slideIn { from { transform: translateX(20px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

  @media (max-width: 768px) {
    .sidebar { position: fixed; left: 0; top: 0; bottom: 0; z-index: 50; transform: translateX(-100%); }
    .sidebar.open { transform: translateX(0); }
    .welcome-features { grid-template-columns: 1fr; }
  }
`;

// ── Toast Component ───────────────────────────────────────────────────────────
function ToastContainer({ toasts }) {
  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.type}`}>
          <span className="toast-icon">{t.icon || { success: "✓", error: "✕", info: "ℹ", warning: "⚠" }[t.type] || "ℹ"}</span>
          <span>{t.message}</span>
        </div>
      ))}
    </div>
  );
}

// ── HITL Modal ────────────────────────────────────────────────────────────────
function HITLModal({ request, onRespond }) {
  const [modifiedQuery, setModifiedQuery] = useState(request?.query || "");
  useEffect(() => { setModifiedQuery(request?.query || ""); }, [request]);
  if (!request) return null;
  return (
    <div className="modal-overlay">
      <div className="hitl-modal">
        <div className="hitl-header">
          <span className="hitl-icon">⚡</span>
          <h3>{request.title}</h3>
        </div>
        <p className="hitl-desc">{request.description}</p>
        <div className="hitl-context">
          <label>Query / Action:</label>
          <div className="hitl-query-text">{request.query}</div>
        </div>
        <div className="hitl-modify-section">
          <label>Modify before executing (optional):</label>
          <textarea className="hitl-textarea" value={modifiedQuery} onChange={(e) => setModifiedQuery(e.target.value)} />
        </div>
        <div className="hitl-actions">
          <button className="btn-deny" onClick={() => onRespond("deny", request.query)}>✕ Deny</button>
          <button className="btn-modify" onClick={() => onRespond("modify", modifiedQuery)}>✎ Modify & Approve</button>
          <button className="btn-approve" onClick={() => onRespond("approve", request.query)}>✓ Approve</button>
        </div>
      </div>
    </div>
  );
}

// ── Settings Modal ────────────────────────────────────────────────────────────
function SettingsModal({ settings, onSave, onClose }) {
  const [form, setForm] = useState({ ...settings });
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="settings-modal">
        <div className="settings-header">
          <h3>⚙ Configuration</h3>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>
        <div className="settings-body">
          <div className="settings-section">
            <h4>🔑 API KEYS</h4>
            <div className="field-group">
              <label>Groq API Key <span className="required">*</span></label>
              <input type="password" value={form.groqApiKey} onChange={(e) => set("groqApiKey", e.target.value)} placeholder="gsk_..." />
              <span className="field-hint">Required. Free at <a href="https://console.groq.com/keys" target="_blank" rel="noreferrer">console.groq.com</a></span>
            </div>
            <div className="field-group">
              <label>Tavily API Key</label>
              <input type="password" value={form.tavilyApiKey} onChange={(e) => set("tavilyApiKey", e.target.value)} placeholder="tvly-..." />
              <span className="field-hint">Optional. For real web search. Free at <a href="https://tavily.com" target="_blank" rel="noreferrer">tavily.com</a></span>
            </div>
            <div className="field-group">
              <label>n8n Webhook URL</label>
              <input type="text" value={form.n8nWebhookUrl} onChange={(e) => set("n8nWebhookUrl", e.target.value)} placeholder="https://your-n8n.cloud/webhook/..." />
              <span className="field-hint">Optional. For advanced RAG pipeline.</span>
            </div>
          </div>
          <div className="settings-section">
            <h4>🤖 AGENT SETTINGS</h4>
            <div className="field-group">
              <label>Groq Model</label>
              <select value={form.model} onChange={(e) => set("model", e.target.value)}>
                <option value="llama-3.3-70b-versatile">llama-3.3-70b (Best, Recommended)</option>
                <option value="llama-3.1-8b-instant">llama-3.1-8b (Fastest)</option>
                <option value="llama3-70b-8192">llama3-70b (Stable)</option>
                <option value="gemma2-9b-it">gemma2-9b (Lightweight)</option>
                <option value="meta-llama/llama-4-scout-17b-16e-instruct">llama-4-scout (Vision)</option>
              </select>
            </div>
            <div className="field-group">
              <label>Number of Debate Agents</label>
              <select value={form.numAgents} onChange={(e) => set("numAgents", parseInt(e.target.value))}>
                <option value={3}>3 Agents (Factual · Skeptical · Synthesizer)</option>
                <option value={4}>4 Agents (+ Devil's Advocate)</option>
                <option value={5}>5 Agents (+ Domain Expert)</option>
              </select>
            </div>
            {[["hitlEnabled", "Human-in-the-Loop (HITL)"], ["debateEnabled", "Multi-Agent Debate"], ["autoSearch", "Auto Web Search for Recent Info"]].map(([k, label]) => (
              <div className="field-group toggle-group" key={k}>
                <label>{label}</label>
                <label className="toggle-switch">
                  <input type="checkbox" checked={form[k]} onChange={(e) => set(k, e.target.checked)} />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            ))}
          </div>
        </div>
        <div className="settings-footer">
          <button className="btn-save" onClick={() => onSave(form)}>Save Settings</button>
        </div>
      </div>
    </div>
  );
}

// ── Confidence Bar ────────────────────────────────────────────────────────────
function ConfidenceBar({ score }) {
  const pct = Math.round(score * 100);
  const color = pct >= 80 ? "#00ff88" : pct >= 60 ? "#ffb020" : pct >= 40 ? "#ff9a44" : "#ff4757";
  const label = pct >= 80 ? "High" : pct >= 60 ? "Medium" : pct >= 40 ? "Low" : "Very Low";
  return (
    <div className="confidence-bar">
      <span className="confidence-label">Confidence</span>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="confidence-value" style={{ color }}>{pct}% · {label}</span>
    </div>
  );
}

// ── Sources ───────────────────────────────────────────────────────────────────
function SourceTags({ sources, ragChunks }) {
  if ((!sources || !sources.length) && (!ragChunks || !ragChunks.length)) return null;
  const icon = (s) => s.source === "tavily_answer" ? "📡" : s.source === "groq_knowledge" ? "🧠" : "🌐";
  return (
    <div className="sources-section">
      <div className="sources-title">Sources</div>
      <div className="sources-list">
        {(sources || []).slice(0, 6).map((s, i) =>
          s.url
            ? <a key={i} className="source-tag" href={s.url} target="_blank" rel="noopener noreferrer" title={s.snippet || ""}><span className="source-tag-icon">{icon(s)}</span><span>{truncate(s.title, 35)}</span></a>
            : <span key={i} className="source-tag" title={s.snippet || ""}><span className="source-tag-icon">{icon(s)}</span><span>{truncate(s.title, 35)}</span></span>
        )}
        {(ragChunks || []).map((c, i) => (
          <span key={`doc-${i}`} className="source-tag"><span className="source-tag-icon">📄</span><span>{truncate(c.docName, 30)}</span></span>
        ))}
      </div>
    </div>
  );
}

// ── Debate Panel ──────────────────────────────────────────────────────────────
function DebatePanel({ debate, messageId }) {
  const [open, setOpen] = useState(false);
  if (!debate || !debate.turns) return null;
  const sentimentColors = { agree: { color: "#00ff88", label: "✓ Agrees" }, disagree: { color: "#ff4757", label: "✗ Disagrees" }, neutral: { color: "#ffb020", label: "~ Neutral" } };
  const strengthMap = { strong: "● Strong consensus", moderate: "◑ Moderate agreement", weak: "○ Significant debate", mixed: "◐ Mixed views", none: "− No data" };
  const votes = debate.votes;
  return (
    <div className="debate-section">
      <button className="debate-toggle-btn" onClick={() => setOpen(!open)}>
        🤖 Debate Transcript — {debate.numAgents} agents · {strengthMap[debate.strength] || ""}
      </button>
      {open && (
        <div className="debate-panel">
          <div className="debate-header">
            <span className="debate-header-title">Multi-Agent Fact-Check</span>
            <span className="debate-round-badge">{votes.agree} agree · {votes.disagree} disagree · {votes.neutral} neutral</span>
          </div>
          <div className="debate-turns">
            {debate.turns.map((turn, i) => {
              const sc = sentimentColors[turn.sentiment] || sentimentColors.neutral;
              return (
                <div key={i} className="debate-turn">
                  <div className="debate-turn-header">
                    <span className="agent-name-badge" style={{ color: turn.agent.color, borderColor: turn.agent.color + "30", background: turn.agent.color + "10" }}>{turn.agent.emoji} {turn.agent.name}</span>
                    <span className="agent-role">{turn.agent.role}</span>
                    <span className="agent-name-badge" style={{ color: sc.color, borderColor: sc.color + "30", background: sc.color + "10", marginLeft: "auto" }}>{sc.label}</span>
                    {turn.round > 1 && <span className="agent-role" style={{ marginLeft: 4 }}>· Round 2</span>}
                  </div>
                  <div className="debate-turn-text">{turn.response}</div>
                </div>
              );
            })}
          </div>
          {debate.consensus && (
            <div className="debate-consensus">
              <div className="debate-consensus-label">⚖ Agent Consensus</div>
              <div className="debate-consensus-text">{debate.consensus}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Rating Section ────────────────────────────────────────────────────────────
function RatingSection({ messageId, onRate, onCorrection }) {
  const [stars, setStars] = useState(0);
  const [hover, setHover] = useState(0);
  const [showCorrection, setShowCorrection] = useState(false);
  const [correctionText, setCorrectionText] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const rate = (n) => { setStars(n); onRate(messageId, n); };
  const submitCorrection = () => {
    if (!correctionText.trim()) return;
    onCorrection(messageId, correctionText.trim());
    setShowCorrection(false);
    setCorrectionText("");
    setSubmitted(true);
  };

  return (
    <div className="rating-section">
      <span className="rating-label">Rate this response:</span>
      <div className="star-rating">
        {[1,2,3,4,5].map((n) => (
          <button key={n} className={`star-btn ${n <= (hover || stars) ? "active" : ""}`}
            onClick={() => rate(n)} onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(0)}>
            {n <= (hover || stars) ? "★" : "☆"}
          </button>
        ))}
      </div>
      {!submitted && (
        <button className="correction-toggle" onClick={() => setShowCorrection(!showCorrection)}>✎ Add correction</button>
      )}
      {submitted && <span style={{ fontSize: 11, color: "var(--accent2)", fontFamily: "var(--font-mono)" }}>✓ Feedback saved</span>}
      {showCorrection && (
        <div className="correction-box">
          <textarea className="correction-textarea" value={correctionText} onChange={(e) => setCorrectionText(e.target.value)} placeholder="What was wrong or missing? Your feedback helps improve future responses..." />
          <button className="correction-submit" onClick={submitCorrection}>Submit Feedback</button>
        </div>
      )}
    </div>
  );
}

// ── Message ───────────────────────────────────────────────────────────────────
function Message({ msg, onRate, onCorrection }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message-wrapper ${isUser ? "user" : "ai"}`}>
      <div className="message-header">
        <div className={`message-avatar ${isUser ? "avatar-user" : "avatar-ai"}`}>{isUser ? "U" : "N"}</div>
        <div>
          <span className="message-sender">{isUser ? "You" : "Nexus"}</span>
          <span className="message-time">{formatTime(msg.timestamp)}</span>
        </div>
      </div>
      <div className="message-body">
        {isUser
          ? <div className="message-text">{msg.content}</div>
          : <>
              <div className="message-text" dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
              {msg.confidence !== undefined && <ConfidenceBar score={msg.confidence} />}
              {(msg.sources || msg.ragChunks) && <SourceTags sources={msg.sources} ragChunks={msg.ragChunks} />}
              {msg.debate && <DebatePanel debate={msg.debate} messageId={msg.id} />}
              <RatingSection messageId={msg.id} onRate={onRate} onCorrection={onCorrection} />
            </>
        }
      </div>
    </div>
  );
}

// ── Processing Step ───────────────────────────────────────────────────────────
function ProcessingStep({ text, status }) {
  const icons = { active: "◌", done: "✓", error: "!" };
  return (
    <div className={`processing-step ${status}`}>
      <span className="step-icon">{icons[status] || "◌"}</span>
      <span>{text}</span>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  // Settings
  const defaultSettings = {
    groqApiKey: "", tavilyApiKey: "", n8nWebhookUrl: "",
    model: CONFIG.DEFAULT_MODEL, numAgents: 3,
    hitlEnabled: true, debateEnabled: true, autoSearch: true,
  };

  const [settings, setSettings] = useState(() => {
    try { const s = localStorage.getItem("nexus_settings_v2"); return s ? { ...defaultSettings, ...JSON.parse(s) } : defaultSettings; }
    catch { return defaultSettings; }
  });
  const [sessions, setSessions] = useState(() => {
    try { const s = localStorage.getItem("nexus_sessions_v2"); return s ? JSON.parse(s) : []; }
    catch { return []; }
  });
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [documents, setDocuments] = useState(() => {
    try { const s = localStorage.getItem("nexus_docs_v2"); return s ? JSON.parse(s) : []; }
    catch { return []; }
  });
  const [feedback, setFeedback] = useState(() => {
    try { const s = localStorage.getItem("nexus_feedback_v2"); return s ? JSON.parse(s) : []; }
    catch { return []; }
  });

  // UI state
  const [activeFeatures, setActiveFeatures] = useState({ web: true, rag: true, debate: true, hitl: true });
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [toasts, setToasts] = useState([]);

  // Processing state
  const [processingSteps, setProcessingSteps] = useState([]);
  const [agentThinking, setAgentThinking] = useState({});
  const [uploadProgress, setUploadProgress] = useState(null);

  // HITL
  const [hitlRequest, setHitlRequest] = useState(null);
  const hitlResolveRef = useRef(null);

  // Input
  const [inputValue, setInputValue] = useState("");
  const textareaRef = useRef(null);
  const chatAreaRef = useRef(null);

  // Persistence
  useEffect(() => { try { localStorage.setItem("nexus_sessions_v2", JSON.stringify(sessions.slice(-50))); } catch {} }, [sessions]);
  useEffect(() => { try { localStorage.setItem("nexus_docs_v2", JSON.stringify(documents)); } catch {} }, [documents]);
  useEffect(() => { try { localStorage.setItem("nexus_feedback_v2", JSON.stringify(feedback.slice(-200))); } catch {} }, [feedback]);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => { if (chatAreaRef.current) chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight; });
  }, []);

  const showToast = useCallback((message, type = "info", icon = null) => {
    const id = generateId();
    setToasts((t) => [...t, { id, message, type, icon }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3500);
  }, []);

  // Session helpers
  const currentSession = sessions.find((s) => s.id === currentSessionId);

  const newSession = useCallback(() => {
    const session = { id: generateId(), title: "New Research Session", messages: [], createdAt: new Date().toISOString() };
    setSessions((prev) => [...prev, session]);
    setCurrentSessionId(session.id);
    return session;
  }, []);

  const getOrCreateSession = useCallback(() => {
    if (currentSessionId) {
      const s = sessions.find((x) => x.id === currentSessionId);
      if (s) return s;
    }
    return null; // will be created on send
  }, [currentSessionId, sessions]);

  // HITL helper
  const requestHITL = useCallback((req) => {
    return new Promise((resolve) => {
      if (!activeFeatures.hitl || !settings.hitlEnabled) { resolve({ action: "approve", query: req.query }); return; }
      setHitlRequest(req);
      hitlResolveRef.current = resolve;
    });
  }, [activeFeatures.hitl, settings.hitlEnabled]);

  const handleHITLRespond = useCallback((action, query) => {
    setHitlRequest(null);
    if (hitlResolveRef.current) { hitlResolveRef.current({ action, query }); hitlResolveRef.current = null; }
  }, []);

  // Step helpers
  const addStep = useCallback((text, status = "active") => {
    const step = { id: generateId(), text, status };
    setProcessingSteps((prev) => [...prev, step]);
    return step.id;
  }, []);
  const updateStep = useCallback((id, text, status) => {
    setProcessingSteps((prev) => prev.map((s) => s.id === id ? { ...s, text, status } : s));
  }, []);

  // Send message
  const sendMessage = useCallback(async () => {
    const query = inputValue.trim();
    if (!query || isProcessing) return;
    if (!settings.groqApiKey) { setShowSettings(true); showToast("Please set your Groq API key in Settings first.", "warning", "🔑"); return; }

    setInputValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setIsProcessing(true);
    setProcessingSteps([]);
    setAgentThinking({});

    let session = sessions.find((s) => s.id === currentSessionId);
    if (!session) {
      session = { id: generateId(), title: "New Research Session", messages: [], createdAt: new Date().toISOString() };
      setSessions((prev) => [...prev, session]);
      setCurrentSessionId(session.id);
    }

    const userMsg = { role: "user", content: query, timestamp: new Date().toISOString() };

    setSessions((prev) => prev.map((s) => {
      if (s.id !== session.id) return s;
      const updated = { ...s, messages: [...s.messages, userMsg] };
      if (s.messages.length === 0) updated.title = truncate(query, 45);
      return updated;
    }));

    let ragChunks = [];
    let webResults = [];
    let finalResponse = "";
    let debate = null;

    try {
      // RAG
      if (activeFeatures.rag && documents.length > 0) {
        const stepId = addStep("Searching documents with RAG…");
        const hitl = await requestHITL({ title: "Document Access Request", description: `Agent wants to search ${documents.length} document(s).`, query: `Search query: "${truncate(query, 80)}"` });
        if (hitl.action !== "deny") {
          try {
            ragChunks = await retrieveRelevantChunks(documents, hitl.query);
            updateStep(stepId, `Found ${ragChunks.length} relevant document chunks`, "done");
          } catch (e) { updateStep(stepId, `Document search error: ${e.message}`, "error"); }
        } else { updateStep(stepId, "Document access denied", "error"); }
      }

      // Web search
      const shouldSearch = activeFeatures.web && (settings.autoSearch || activeFeatures.web);
      if (shouldSearch) {
        const stepId = addStep("Searching the web…");
        const hitl = await requestHITL({ title: "Web Search Request", description: "Agent wants to search the web for current information.", query: `Web search: "${truncate(query, 80)}"` });
        if (hitl.action !== "deny") {
          try {
            webResults = await performWebSearch(hitl.query, settings);
            updateStep(stepId, `Found ${webResults.length} web sources`, "done");
          } catch (e) { updateStep(stepId, `Web search error: ${e.message}`, "error"); }
        } else { updateStep(stepId, "Web search denied", "error"); }
      }

      // Generate response
      const genStep = addStep("Generating research response…");
      const ragContext = formatRagContext(ragChunks);
      const history = (session.messages || []).slice(-10, -1).map((m) => ({ role: m.role, content: m.content }));
      const webCtx = webResults.length > 0 ? `\n\n## Live Web Search Results\n${webResults.map((r, i) => `[${i+1}] **${r.title}**\nURL: ${r.url}\n${r.snippet || ""}`).join("\n\n")}` : "";
      const docCtx = ragContext ? `\n\n## Relevant Document Excerpts\n${ragContext}` : "";
      finalResponse = await callGroq({
        prompt: `${query}${docCtx}${webCtx}\n\nProvide a comprehensive research response with: 1) Direct answer, 2) Supporting evidence, 3) Caveats, 4) Source citations.`,
        systemPrompt: "You are Nexus, an expert AI research assistant. Use markdown formatting, cite sources as [Source N], distinguish facts from inference.",
        history,
        temperature: 0.6,
        maxTokens: 3000,
        settings,
      });
      updateStep(genStep, "Response generated", "done");

      // Debate
      if (activeFeatures.debate && settings.debateEnabled) {
        const debateStep = addStep(`Running ${settings.numAgents}-agent fact-check debate…`);
        try {
          debate = await runMultiAgentDebate({
            query, initialResponse: finalResponse, ragContext, webResults,
            numAgents: settings.numAgents, settings,
            onAgentUpdate: (agentId, status) => setAgentThinking((prev) => ({ ...prev, [agentId]: status })),
          });
          const strengthMap = { strong: "✓ Strong consensus", moderate: "~ Moderate agreement", weak: "⚠ Significant debate", mixed: "◑ Mixed views" };
          updateStep(debateStep, `Debate complete — ${strengthMap[debate.strength] || "Consensus reached"}`, "done");
        } catch (e) { updateStep(debateStep, `Debate error: ${e.message}`, "error"); debate = null; }
      }

      const confidence = calculateConfidence({ hasWeb: webResults.length > 0, numWeb: webResults.length, hasDoc: ragChunks.length > 0, numDoc: ragChunks.length, debateStrength: debate?.strength || "none", responseLength: finalResponse.length });

      const assistantMsg = {
        id: generateId(),
        role: "assistant",
        content: finalResponse,
        confidence,
        sources: webResults.map((r, i) => ({ index: i+1, ...r })),
        ragChunks: ragChunks.map((c) => ({ docName: c.docName, chunkIndex: c.chunkIndex })),
        debate,
        timestamp: new Date().toISOString(),
      };

      setSessions((prev) => prev.map((s) => s.id === session.id ? { ...s, messages: [...s.messages, assistantMsg] } : s));

    } catch (error) {
      showToast(`Error: ${truncate(error.message, 80)}`, "error");
      const errMsg = {
        id: generateId(),
        role: "assistant",
        content: `**⚠ Error:** ${error.message}\n\nCommon causes: Invalid API key, network issue, or rate limit. Check Settings to verify your Groq API key.`,
        timestamp: new Date().toISOString(),
      };
      setSessions((prev) => prev.map((s) => s.id === session.id ? { ...s, messages: [...s.messages, errMsg] } : s));
    }

    setIsProcessing(false);
    setAgentThinking({});
    setTimeout(() => textareaRef.current?.focus(), 100);
  }, [inputValue, isProcessing, settings, sessions, currentSessionId, documents, activeFeatures, addStep, updateStep, requestHITL, showToast]);

  // File upload
  const handleFiles = useCallback(async (files) => {
    const allowed = Array.from(files).filter((f) => {
      const ext = f.name.split(".").pop().toLowerCase();
      return ["pdf","png","jpg","jpeg","gif","webp","txt","md","csv","json"].includes(ext);
    });
    if (!allowed.length) { showToast("No supported files selected", "warning"); return; }

    for (const file of allowed) {
      setUploadProgress({ filename: file.name, pct: 0 });
      try {
        const doc = await processFile(file, settings, (pct, label) => setUploadProgress({ filename: label, pct }));
        setDocuments((prev) => {
          const existing = prev.findIndex((d) => d.name === doc.name);
          if (existing >= 0) { const next = [...prev]; next[existing] = doc; return next; }
          return [...prev, doc];
        });
        showToast(`Processed: ${doc.name} (${doc.chunks?.length || 0} chunks)`, "success", "📄");
      } catch (e) { showToast(`Failed: ${file.name} — ${e.message}`, "error"); }
    }
    setUploadProgress(null);
  }, [settings, showToast]);

  const handleSaveSettings = useCallback((newSettings) => {
    setSettings(newSettings);
    try { localStorage.setItem("nexus_settings_v2", JSON.stringify(newSettings)); } catch {}
    setShowSettings(false);
    showToast("Settings saved", "success", "✓");
  }, [showToast]);

  const handleRate = useCallback((messageId, stars) => {
    setFeedback((prev) => {
      const existing = prev.findIndex((f) => f.messageId === messageId);
      const item = { messageId, stars, updatedAt: new Date().toISOString() };
      if (existing >= 0) { const next = [...prev]; next[existing] = { ...next[existing], ...item }; return next; }
      return [...prev, { ...item, createdAt: new Date().toISOString() }];
    });
    showToast(`Rated ${stars} star${stars > 1 ? "s" : ""} — thank you!`, "success");
  }, [showToast]);

  const handleCorrection = useCallback((messageId, text) => {
    setFeedback((prev) => {
      const existing = prev.findIndex((f) => f.messageId === messageId);
      if (existing >= 0) { const next = [...prev]; next[existing] = { ...next[existing], correction: text }; return next; }
      return [...prev, { messageId, correction: text, createdAt: new Date().toISOString() }];
    });
    showToast("Correction saved — thank you!", "success", "🎯");
  }, [showToast]);

  const toggleFeature = (key) => setActiveFeatures((prev) => ({ ...prev, [key]: !prev[key] }));

  const statusDotClass = !settings.groqApiKey ? "" : isProcessing ? "processing" : "ready";
  const statusText = !settings.groqApiKey ? "No API key" : isProcessing ? "Processing…" : "Ready";

  const messages = currentSession?.messages || [];
  const showWelcome = messages.length === 0 && !isProcessing;

  const activeDebatingAgents = AGENTS.slice(0, settings.numAgents).filter((a) => agentThinking[a.id]);

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        {/* Sidebar */}
        <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
          <div className="sidebar-brand">
            <div className="brand-logo">N</div>
            <div>
              <div className="brand-name">Nexus</div>
              <div className="brand-sub">Research Agent</div>
            </div>
          </div>

          <div className="sidebar-section">
            <button className="new-chat-btn" onClick={newSession}><span>+</span> New Research Session</button>
          </div>

          <div className="sidebar-section">
            <span className="sidebar-label">Sessions</span>
            <div className="session-list">
              {sessions.length === 0
                ? <div style={{ fontSize: 11, color: "var(--text-muted)", padding: "6px 4px", fontFamily: "var(--font-mono)" }}>No sessions yet</div>
                : sessions.slice().reverse().slice(0, 20).map((s) => (
                    <div key={s.id} className={`session-item ${s.id === currentSessionId ? "active" : ""}`} onClick={() => setCurrentSessionId(s.id)}>
                      <span className="session-icon">💬</span>
                      <span className="session-title">{s.title || "Untitled"}</span>
                      <button className="session-delete" onClick={(e) => { e.stopPropagation(); setSessions((prev) => prev.filter((x) => x.id !== s.id)); if (s.id === currentSessionId) setCurrentSessionId(null); }}>✕</button>
                    </div>
                  ))
              }
            </div>
          </div>

          <div className="sidebar-section sidebar-docs-section">
            <div className="sidebar-label-row">
              <span className="sidebar-label">Documents</span>
              <span className="doc-count-badge">{documents.length}</span>
            </div>
            <div className="doc-list">
              {documents.length === 0
                ? <div style={{ fontSize: 11, color: "var(--text-muted)", padding: "6px 4px", fontFamily: "var(--font-mono)" }}>No documents</div>
                : documents.map((doc) => (
                    <div key={doc.id} className="doc-item">
                      <span className="doc-icon">{getFileIcon(doc.name)}</span>
                      <div className="doc-info">
                        <span className="doc-name">{truncate(doc.name, 22)}</span>
                        <span className="doc-meta">{formatFileSize(doc.size)} · {doc.chunks?.length || 0} chunks</span>
                      </div>
                      <button className="doc-delete" onClick={() => { setDocuments((prev) => prev.filter((d) => d.id !== doc.id)); showToast("Document removed", "info"); }}>✕</button>
                    </div>
                  ))
              }
            </div>

            <div className="upload-zone"
              onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("drag-over"); }}
              onDragLeave={(e) => { e.currentTarget.classList.remove("drag-over"); }}
              onDrop={(e) => { e.preventDefault(); e.currentTarget.classList.remove("drag-over"); handleFiles(e.dataTransfer.files); }}
              onClick={() => document.getElementById("file-input-hidden").click()}>
              <input id="file-input-hidden" type="file" multiple accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.txt,.md,.csv,.json" style={{ display: "none" }} onChange={(e) => { handleFiles(e.target.files); e.target.value = ""; }} />
              <div className="upload-icon">⬆</div>
              <div className="upload-text">Drop files or click to upload</div>
              <div className="upload-hint">PDF · Images · TXT · CSV · JSON</div>
            </div>

            {uploadProgress && (
              <div className="upload-progress">
                <div className="progress-info">
                  <span>{uploadProgress.filename}</span>
                  <span>{Math.round(uploadProgress.pct)}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadProgress.pct}%` }} />
                </div>
              </div>
            )}
          </div>

          <div className="sidebar-footer">
            <button className="settings-btn" onClick={() => setShowSettings(true)}>⚙ Settings</button>
            <div className="sidebar-status">
              <div className={`status-dot ${statusDotClass}`} />
              <span className="status-text">{statusText}</span>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="main-content">
          <header className="topbar">
            <div className="topbar-name">{currentSession?.title || "New Research Session"}</div>
            <div className="agent-chips">
              {activeDebatingAgents.map((agent) => (
                <div key={agent.id} className={`agent-chip ${agentThinking[agent.id]}`} style={{ color: agent.color, borderColor: agent.color + "40" }}>
                  {agent.emoji} {agent.name}
                </div>
              ))}
            </div>
          </header>

          <div className="chat-area" ref={chatAreaRef}>
            {showWelcome && (
              <div className="welcome-screen">
                <div className="welcome-logo">N</div>
                <h1 className="welcome-title">Nexus <em>Research</em> Agent</h1>
                <p className="welcome-subtitle">Multi-modal RAG · Multi-Agent Debate · Human-in-the-Loop</p>
                <div className="welcome-features">
                  {[["📄","Upload PDFs & images for OCR + vector search"],["🤖","3 AI agents debate & fact-check every answer"],["🌐","Live web search via Tavily for latest info"],["⚡","Human-in-the-loop approval before actions"]].map(([icon,text]) => (
                    <div key={text} className="feature-card"><span className="feature-icon">{icon}</span><span className="feature-text">{text}</span></div>
                  ))}
                </div>
                <div className="starter-queries">
                  <p className="starter-label">Try asking:</p>
                  <div className="starter-chips">
                    {["Summarize my uploaded documents","What are the latest AI research trends?","Compare and fact-check: quantum computing vs classical","Extract all data tables from my PDFs"].map((q) => (
                      <button key={q} className="starter-chip" onClick={() => { setInputValue(q); textareaRef.current?.focus(); }}>{q}</button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="messages-container">
              {messages.map((msg) => (
                <Message key={msg.id || msg.timestamp} msg={msg} onRate={handleRate} onCorrection={handleCorrection} />
              ))}

              {isProcessing && (
                <div className="typing-indicator">
                  <div className="typing-avatar">N</div>
                  <div className="typing-bubble">
                    <div className="agent-thinking-row">
                      {AGENTS.slice(0, settings.numAgents).filter((a) => agentThinking[a.id]).map((agent) => (
                        <div key={agent.id} className={`agent-chip thinking`} style={{ color: agent.color, borderColor: agent.color + "40", background: agent.color + "10" }}>
                          {agent.emoji} {agent.name}
                        </div>
                      ))}
                    </div>
                    <div className="typing-dots"><span /><span /><span /></div>
                    <div className="processing-steps">
                      {processingSteps.map((step) => <ProcessingStep key={step.id} text={step.text} status={step.status} />)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Input */}
          <div className="input-area">
            <div className="input-options">
              {[["web","🌐 Web Search"],["rag","📚 Doc RAG"],["debate","🤖 Debate"],["hitl","⚡ HITL"]].map(([key, label]) => (
                <button key={key} className={`option-tag ${activeFeatures[key] ? "active" : ""}`} onClick={() => toggleFeature(key)}>{label}</button>
              ))}
            </div>
            <div className="input-box">
              <div className="input-row">
                <textarea
                  ref={textareaRef}
                  className="chat-textarea"
                  value={inputValue}
                  onChange={(e) => { setInputValue(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px"; }}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                  placeholder="Ask anything — I'll search the web, your documents, and use multi-agent debate to give you the best answer..."
                  rows={1}
                />
                <button className="send-btn" onClick={sendMessage} disabled={isProcessing || !inputValue.trim()}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round"/>
                  </svg>
                </button>
              </div>
            </div>
            <div className="input-footer">
              <span className="input-hint">Shift+Enter for new line · Enter to send</span>
              <span className="model-badge">{settings.model || CONFIG.DEFAULT_MODEL}</span>
            </div>
          </div>
        </main>
      </div>

      {/* Modals */}
      {hitlRequest && <HITLModal request={hitlRequest} onRespond={handleHITLRespond} />}
      {showSettings && <SettingsModal settings={settings} onSave={handleSaveSettings} onClose={() => setShowSettings(false)} />}

      {/* Toasts */}
      <ToastContainer toasts={toasts} />
    </>
  );
}
