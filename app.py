import hashlib
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from voice_assistant import (
    SUPPORTED_LANGUAGES,
    get_active_provider,
    inspect_database_schema,
    query_schemes,
    synthesize_speech,
    transcribe_audio,
)

DB_PATH = "schemes.db"

SAMPLE_QUERIES: dict[str, str] = {
    "Hindi (Farmer)": "मैं उत्तर प्रदेश का 45 साल का किसान हूँ, मुझे खेती और आर्थिक मदद के लिए कौन सी सरकारी योजना मिल सकती है?",
    "Marathi (Women)": "मी महाराष्ट्रातील 28 वर्षांची महिला आहे, मला स्वतःचा व्यवसाय सुरू करण्यासाठी किंवा आर्थिक मदतीसाठी कोणती योजना आहे?",
    "Hinglish (Student)": "Main ek 20 saal ka student hoon, mujhe padhai aur skill training ke liye kaunsi yojana milegi?",
    "Telugu (Farmer)": "నేను తెలంగాణ రైతును, నాకు వ్యవసాయం మరియు పంట సహాయం కోసం ఏ పథకాలు ఉన్నాయి?",
    "Tamil (Health)": "நான் தமிழ்நாட்டைச் சேர்ந்தவர், குடும்ப மருத்துவ காப்பீடு மற்றும் சிகிச்சைக்கான அரசு திட்டம் என்ன?",
    "English (Housing)": "I am a rural citizen looking for affordable housing and clean cooking gas schemes.",
}

st.set_page_config(
    page_title="YojanaSetu Voice",
    page_icon=":material/record_voice_over:",
    layout="wide",
)


@st.cache_data(ttl="1h")
def load_schemes_dataframe(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load the schemes table from SQLite for the analytics dashboard."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql("SELECT * FROM schemes", conn)
    finally:
        conn.close()


@st.cache_data(ttl="1h")
def get_schema_summary(db_path: str = DB_PATH) -> dict:
    """Cache dynamic SQLite schema inspection."""
    return inspect_database_schema(db_path)


@st.cache_data(max_entries=50)
def cached_tts(text: str, language: str) -> tuple[bytes | None, str | None]:
    """Cache generated TTS MP3 bytes to avoid redundant synthesis across reruns."""
    return synthesize_speech(text, language)


def _init_session_state() -> None:
    if "voice_query_text" not in st.session_state:
        st.session_state["voice_query_text"] = ""
    if "voice_result" not in st.session_state:
        st.session_state["voice_result"] = None
    if "last_audio_hash" not in st.session_state:
        st.session_state["last_audio_hash"] = None


def render_voice_assistant_tab() -> None:
    """Render the default voice-first multilingual scheme discovery tab."""
    _init_session_state()
    provider = get_active_provider()

    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Speak or dictate your need in any Indian language", width="content")
        if provider == "gemini":
            st.badge("LLM: Gemini + SQLite", icon=":material/auto_awesome:", color="green")
        elif provider == "openai":
            st.badge("LLM: OpenAI + SQLite", icon=":material/auto_awesome:", color="green")
        else:
            st.badge("Mode: Smart Offline SQLite Fallback", icon=":material/offline_bolt:", color="blue")

    st.caption(
        "Supports Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, and English. "
        "Tip: Use Wispr Flow in the text box below for instant hands-free voice dictation, or record directly with your mic."
    )

    col_lang, col_sample = st.columns([1, 2], vertical_alignment="bottom")
    with col_lang:
        preferred_lang = st.selectbox(
            "Language preference",
            SUPPORTED_LANGUAGES,
            index=0,
            help="Auto-detect identifies your script or Hinglish automatically.",
        )
    with col_sample:
        selected_sample = st.pills(
            "Try a sample voice query",
            list(SAMPLE_QUERIES.keys()),
            selection_mode="single",
        )
        if selected_sample and st.session_state.get("last_sample") != selected_sample:
            st.session_state["last_sample"] = selected_sample
            st.session_state["voice_query_text"] = SAMPLE_QUERIES[selected_sample]

    with st.container(border=True):
        audio_col, text_col = st.columns([1, 2], vertical_alignment="top")

        with audio_col:
            st.markdown("**1. Record regional voice**")
            audio_recording = st.audio_input(
                "Tap the mic to speak",
                help="Record in Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, or English.",
            )
            if audio_recording is not None:
                audio_bytes = audio_recording.getvalue()
                audio_hash = hashlib.sha256(audio_bytes).hexdigest()
                if audio_hash != st.session_state["last_audio_hash"]:
                    st.session_state["last_audio_hash"] = audio_hash
                    with st.spinner("Transcribing regional voice..."):
                        stt = transcribe_audio(
                            audio_bytes,
                            mime_type=audio_recording.type or "audio/wav",
                            preferred_language=preferred_lang,
                        )
                    if stt.get("text"):
                        st.session_state["voice_query_text"] = stt["text"]
                        st.toast("Voice transcribed! Searching matching schemes...", icon=":material/mic:")
                    elif stt.get("error"):
                        st.info(stt["error"], icon=":material/mic_off:")

        with text_col:
            st.markdown("**2. Wispr Flow / Text query**")
            query_text = st.text_area(
                "Dictate with Wispr Flow or type your query",
                key="voice_query_text",
                height=110,
                placeholder="Click here and press your Wispr Flow hotkey to speak in Hindi, Marathi, Telugu, Tamil, Kannada, Bengali, Hinglish, or English...",
                label_visibility="collapsed",
            )
            with st.container(horizontal=True):
                search_clicked = st.button(
                    "Find Top 3 Schemes",
                    type="primary",
                    icon=":material/search:",
                    width="stretch",
                )
                auto_tts = st.toggle("Listen hands-free (TTS)", value=True)

    # Trigger query on button click or newly transcribed audio
    should_run = search_clicked or (
        audio_recording is not None
        and st.session_state.get("last_ran_audio_hash") != st.session_state.get("last_audio_hash")
        and bool(st.session_state["voice_query_text"].strip())
    )

    if should_run:
        clean_q = (query_text or "").strip()
        if not clean_q:
            st.warning("Please speak into the microphone or dictate/type a query first.", icon=":material/warning:")
        else:
            st.session_state["last_ran_audio_hash"] = st.session_state.get("last_audio_hash")
            with st.spinner("Matching schemes from SQLite database..."):
                st.session_state["voice_result"] = query_schemes(
                    clean_q,
                    db_path=DB_PATH,
                    preferred_language=preferred_lang,
                    top_k=3,
                )

    result = st.session_state.get("voice_result")
    if result:
        entities = result["entities"]
        labels = result["labels"]
        schemes = result["schemes"]

        st.subheader(f"Top {len(schemes)} Matching Schemes ({result['language']})")

        with st.container(horizontal=True):
            st.metric("Language", result["language"], border=True)
            st.metric("State", entities.get("state") or "All India", border=True)
            st.metric("Beneficiary", entities.get("occupation_or_beneficiary") or "Citizen", border=True)
            st.metric("Age", str(entities["age"]) if entities.get("age") else "Any", border=True)

        if auto_tts and result.get("tts_text"):
            audio_bytes, tts_err = cached_tts(result["tts_text"], result["language"])
            if audio_bytes:
                st.markdown("**Listen hands-free**")
                st.audio(audio_bytes, format="audio/mp3")
            elif tts_err:
                st.caption(f"Audio playback note: {tts_err}")

        # Clean, mobile-friendly vertical card stack with strictly the 4 required fields
        for idx, scheme in enumerate(schemes, 1):
            with st.container(border=True):
                st.markdown(f"### {idx}. {scheme['scheme_name']}")
                st.markdown(f"**{labels['eligibility']}:** {scheme['eligibility']}")
                st.markdown(f"**{labels['documents_needed']}:** {scheme['documents_needed']}")
                website = scheme.get("official_website", "")
                if website:
                    with st.container(horizontal=True, vertical_alignment="center"):
                        st.markdown(f"**{labels['official_website']}:** [{website}]({website})")
                        st.link_button("Open Portal", website, icon=":material/open_in_new:")

        tech_exp = st.expander("Pipeline details (Schema & SQLite query)", icon=":material/database:", on_change="rerun")
        if tech_exp.open:
            with tech_exp:
                schema = get_schema_summary(DB_PATH)
                st.caption(f"Primary table discovered: `{schema.get('primary_table')}` | Need extracted: `{entities.get('need')}`")
                if result.get("sql_query"):
                    st.code(result["sql_query"], language="sql")


def render_analytics_tab(df: pd.DataFrame) -> None:
    """Render the existing YojanaSetu analytics dashboard inside Tab 2."""
    with st.sidebar:
        st.header(":material/filter_alt: Analytics Filters")
        category = st.selectbox(
            "Select category",
            ["All"] + sorted(df["category"].dropna().unique().tolist()),
        )
        scheme_type = st.selectbox(
            "State or Central",
            ["All"] + sorted(df["state_or_central"].dropna().unique().tolist()),
        )
        search = st.text_input(
            "Search scheme",
            placeholder="e.g. PM Kisan, Scholarship...",
        )

    filtered_df = df.copy()
    if category != "All":
        filtered_df = filtered_df[filtered_df["category"] == category]
    if scheme_type != "All":
        filtered_df = filtered_df[filtered_df["state_or_central"] == scheme_type]
    if search:
        filtered_df = filtered_df[
            filtered_df["scheme_name"].str.contains(search, case=False, na=False)
        ]

    with st.container(horizontal=True):
        st.metric("Total Q&A records", len(df), border=True)
        st.metric("Distinct schemes", df["scheme_name"].nunique(), border=True)
        st.metric("Categories", df["category"].nunique(), border=True)
        st.metric("Filtered records", len(filtered_df), border=True)

    st.subheader("Category and governance analytics")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            category_counts = df["category"].value_counts().head(10)
            fig1 = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Top Scheme Categories",
                labels={"x": "Category", "y": "Number of Records"},
            )
            st.plotly_chart(fig1, width="stretch")

    with col2:
        with st.container(border=True):
            scheme_counts = df["state_or_central"].value_counts()
            fig2 = px.pie(
                values=scheme_counts.values,
                names=scheme_counts.index,
                title="State vs Central Schemes",
            )
            st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            category_state = (
                df.groupby(["category", "state_or_central"])
                .size()
                .reset_index(name="count")
            )
            fig3 = px.bar(
                category_state,
                x="category",
                y="count",
                color="state_or_central",
                title="Category-wise State vs Central Schemes",
            )
            st.plotly_chart(fig3, width="stretch")

    with col4:
        with st.container(border=True):
            year_counts = df["launched_year"].value_counts().sort_index()
            fig4 = px.line(
                x=year_counts.index,
                y=year_counts.values,
                title="Scheme Launch Trends Over Years",
                labels={"x": "Year", "y": "Number of Records"},
            )
            st.plotly_chart(fig4, width="stretch")

    st.subheader("Filtered scheme records")
    if len(filtered_df) > 0:
        for _, row in filtered_df.head(20).iterrows():
            with st.container(border=True):
                st.markdown(f"#### {row['scheme_name']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Type:** {row['state_or_central']}")
                with c2:
                    st.write(f"**Target beneficiary:** {row['target_beneficiary']}")
                    st.write(f"**Launch year:** {row['launched_year']}")
                st.write(f"**Question:** {row['question_english']}")
                st.write(f"**Details (Hindi):** {row['answer'][:350]}...")
                st.markdown(f"[Official Website]({row['official_website']})")
    else:
        st.warning("No schemes found for the current sidebar filters.")


# ---------------- MAIN APP ---------------- #

st.title("YojanaSetu Voice")
st.caption("Voice-First Multilingual Government Scheme Assistant & Analytics Platform")

df_schemes = load_schemes_dataframe(DB_PATH)

tab_voice, tab_analytics = st.tabs(
    [":material/mic: Voice Assistant", ":material/bar_chart: Analytics Dashboard"],
    on_change="rerun",
)

if tab_voice.open:
    with tab_voice:
        render_voice_assistant_tab()

if tab_analytics.open:
    with tab_analytics:
        render_analytics_tab(df_schemes)

st.caption("Built with Streamlit, SQLite, Wispr Flow-ready dictation, LLM NL-to-SQL, and gTTS")
