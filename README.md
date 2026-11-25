# YojanaSetu Voice — Multilingual Government Scheme Assistant & Analytics Platform

**YojanaSetu Voice** upgrades YojanaSetu into a voice-first, multilingual civic services assistant built for the **Scaler × Wispr Flow "India Runs on Voice" Hackathon (Civic Services Track)**.

> **Built With Voice:** This voice-first upgrade was architected and built hands-free using **Wispr Flow** inside **Antigravity IDE**.

Citizens can speak or dictate naturally—without typing in English—to discover the top 3 matching Indian central and state welfare schemes along with **Eligibility**, **Documents Needed**, and **Official Website** links, plus **Text-to-Speech (`gTTS`)** playback and a full **Analytics Dashboard**.

## Key Features

* **Voice-First Discovery (Tab 1 — Default):**
  * Supports **Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, and English**.
  * Regional mic recording via `st.audio_input` (transcribed with Gemini or OpenAI Whisper when configured) alongside a **Wispr-Flow-friendly** dictation box.
  * Extracts **State, Age, Occupation/Beneficiary Type, and Need** from natural speech.
  * Dynamically inspects SQLite tables and columns (`schemes.db` — 3,473 Q&A records across 96 schemes) and executes safe read-only `SELECT` queries via LLM (Gemini / OpenAI) with an automatic **offline multilingual keyword + fuzzy fallback**.
  * Displays clean, mobile-friendly cards with **Scheme Name, Eligibility, Documents Needed, and Official Website**, plus hands-free **TTS audio playback** (`st.audio`).
* **Analytics Dashboard (Tab 2):**
  * Interactive Plotly visualizations for top scheme categories, State vs Central breakdown, category-wise distribution, and launch year trends.
  * Sidebar filters for Category, State/Central level, and keyword search.

## Tech Stack

* **Frontend:** Streamlit (`st.tabs`, `st.audio_input`, `st.audio`, `st.pills`, `st.container(border=True)`)
* **Database:** SQLite (`schemes.db`) with dynamic schema inspection
* **Voice & AI:** Wispr Flow-compatible dictation, Gemini (`google-genai`), OpenAI (`openai`), `gTTS`
* **Analytics:** Pandas, Plotly

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment (optional for LLM & audio STT; offline fallback works out of the box):**
   ```bash
   cp .env.example .env
   # Add GEMINI_API_KEY or OPENAI_API_KEY in .env
   ```

3. **Verify the voice query pipeline:**
   ```bash
   python test_voice_query.py
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Project Structure

* `app.py` — Streamlit UI with `Voice Assistant` (default) and `Analytics Dashboard` tabs
* `voice_assistant.py` — Dynamic SQLite schema inspector, multilingual entity extractor, LLM NL-to-SQL + offline fuzzy fallback, and `gTTS` audio synthesis
* `schemes.db` — SQLite database of Indian government schemes (3,473 Q&A records across 96 schemes)
* `test_voice_query.py` — Automated verification script for Hindi, Marathi, and English voice queries
* `HACKATHON_SUBMISSION.md` — Round 1 Idea, Prompt, and Impact drafts for Track 01 (Civic Services)
* `.env.example` — Environment variable template (no hardcoded secrets)
