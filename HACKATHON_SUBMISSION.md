# Scaler × Wispr Flow — "India Runs on Voice" Hackathon (Round 1 Submission)

**Project Name:** YojanaSetu Voice  
**Track:** Track 01 — Civic Services  
**Tagline:** Keyboard-free, multilingual voice discovery for Indian government welfare schemes.

---

## 1. Idea

**Problem:** Around **89% of Indians do not speak English** (only **10.6%** are English speakers per the 2011 Census), and most citizens prefer regional languages and voice over typing. Yet discovering central and state welfare schemes requires navigating fragmented English portals, complex dropdowns, and dense PDFs—locking out farmers, women, students, street vendors, and senior citizens from benefits they are legally entitled to.

**Solution:** **YojanaSetu Voice** upgrades our existing **3,473 Q&A records across 96 schemes** (`schemes.db`, spanning 30 categories) into a **voice-first, zero-keyboard civic assistant**:
- **Speak Naturally in 8 Languages:** Citizens speak or dictate via **Wispr Flow** (or Streamlit's built-in `st.audio_input` microphone) in **Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, or English**.
- **Dynamic NL-to-SQLite Retrieval:** An LLM pipeline (Gemini / OpenAI, with a deterministic multilingual keyword + fuzzy fallback) dynamically inspects the SQLite schema, extracts **State, Age, Occupation/Beneficiary Type, and Need**, and retrieves the **top 3 matching schemes**.
- **Actionable, Hands-Free Output:** Returns strictly what a citizen needs—**Scheme Name, Eligibility, Documents Needed, and Official Website link**—in the **same language**, accompanied by **Text-to-Speech (`gTTS`)** playback so low-literacy users can listen hands-free.
- **Policy Transparency:** Retains the full **Analytics Dashboard** in a second tab for NGOs, Common Service Centres (CSCs), and civic researchers.

---

## 2. Prompt

### Build Prompt (Dictated via Wispr Flow)

This upgrade was built hands-free by dictating the architecture prompt via **Wispr Flow** inside **Antigravity IDE**:
- Inspect the existing Streamlit app and `schemes.db` schema first, then upgrade YojanaSetu into **YojanaSetu Voice** for the Civic Services track.
- Create `voice_assistant.py` with dynamic SQLite schema inspection, an LLM NL-to-SQLite pipeline (Gemini / OpenAI via `.env`, plus an offline keyword/fuzzy fallback), 8-language support (Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, English), entity extraction (state, age, occupation/beneficiary type, need), and top-3 scheme replies containing only Scheme Name, Eligibility, Documents Needed, and Official Website.
- Update `app.py` with **Voice Assistant** as the default first tab (`st.audio_input`, Wispr-Flow-friendly text input, mobile-friendly top-3 cards, and `gTTS` `st.audio` playback) and the existing **Analytics Dashboard** in the second tab, plus add `test_voice_query.py` for multilingual verification.

### In-App System Prompt (`voice_assistant.py`)

Below is the core system prompt used in `voice_assistant.py` to turn unstructured regional voice queries into grounded, 4-field civic guidance from our dynamically inspected SQLite database:

```text
You are YojanaSetu Voice, an Indian civic services assistant.

User spoken query: "{query}"
Target reply language: {language} (Hindi | Marathi | Telugu | Tamil | Kannada | Bengali | Hinglish | English)
Extracted user profile: state={state}, age={age}, occupation={occupation_or_beneficiary}, need={need}

Below are the top 3 matching schemes retrieved from SQLite via dynamically inspected schema.
For EACH scheme, return ONLY 4 fields in {language} (Hinglish in natural Romanized script; regional languages in their native script):
1. "scheme_name": Official scheme name.
2. "eligibility": 1–2 crisp sentences in {language} explaining who is eligible (age, income, state rules grounded in DB context).
3. "documents_needed": Concise comma-separated list of required documents grounded in DB context.
4. "official_website": Exact verified URL from SQLite (never alter or invent URLs).
```

**Why this prompt works for voice-first civic UX:**
- **Zero fluff:** Eliminates 10-paragraph policy essays that fail on mobile screens and TTS audio.
- **Language parity:** Answers in the citizen's exact spoken language/script (including Hinglish).
- **Grounded & safe:** Strictly read-only `SELECT` SQLite execution + URL preservation prevents hallucinated portals or broken links.

---

## 3. Impact

- **Zero-Keyboard Accessibility for Bharat:** Replaces multi-step English form-filling with a single 5-second voice query via **Wispr Flow** or regional mic recording, paired with spoken **TTS playback** for citizens who cannot easily read dense text.
- **Bridging the "Last-Mile Awareness" Gap:** Gives citizens exact **Eligibility** and **Documents Needed** upfront—preventing wasted trips to government offices or CSCs and reducing dependence on middlemen.
- **Inclusive Across Regions & Connectivity:** Supports **8 Indian languages/dialects**, runs on lightweight SQLite (`schemes.db`), and degrades gracefully with an **offline multilingual keyword + fuzzy fallback** when LLM API keys are unavailable.
- **Scale Potential:** Ready for deployment at village **Common Service Centres (CSCs)**, NGO helplines, and state citizen kiosks—turning fragmented welfare data into an instant voice bridge between citizens and governance.
