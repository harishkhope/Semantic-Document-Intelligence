import os
import html
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
USER_AVATAR_PATH = os.path.join(os.path.dirname(__file__), "user-avatar.svg")
ASSISTANT_AVATAR_PATH = os.path.join(os.path.dirname(__file__), "assistant-avatar.svg")

st.set_page_config(
    page_title="Semantic Document Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    }

    /* Keep content centered and easier to scan on recordings */
    [data-testid="stAppViewContainer"] [data-testid="block-container"] {
        max-width: 1100px;
        padding-top: 1.25rem;
        padding-bottom: 1rem;
    }

    [data-testid="stSidebar"] [data-testid="block-container"] {
        padding-top: 1.1rem;
        padding-bottom: 1rem;
    }

    [data-testid="stSidebar"] h1 {
        margin-bottom: 0.15rem;
    }

    [data-testid="stSidebar"] h3 {
        margin-top: 0.3rem;
        margin-bottom: 0.55rem;
    }

    [data-testid="stFileUploader"] {
        margin-bottom: 0.55rem;
    }

    [data-testid="stHorizontalBlock"] > div {
        align-self: end;
    }

    /* Keep document selector clearly clickable */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] input {
        cursor: pointer !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #16a34a;
        border-color: #16a34a;
        height: 2.6rem;
        border-radius: 0.6rem;
        transition: transform 160ms var(--ease-out), background-color 160ms var(--ease-out), border-color 160ms var(--ease-out);
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #15803d;
        border-color: #15803d;
    }

    div.stButton > button {
        border-radius: 0.6rem;
        transition: transform 160ms var(--ease-out);
    }

    div.stButton > button:active {
        transform: scale(0.97);
    }

    /* Consistent spacing inside each ingested document card */
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] [data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.65rem;
        padding: 0.65rem 0.8rem;
    }

    [data-testid="stMarkdownContainer"] p {
        line-height: 1.45;
    }

    .doc-title {
        margin: 0;
        font-weight: 600;
        font-size: 1.1rem;
        line-height: 1.3;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def api_get(path: str):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend. Is it running?"
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, **kwargs):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend. Is it running?"
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, detail
    except Exception as exc:
        return None, str(exc)


def api_delete(path: str):
    try:
        r = requests.delete(f"{BACKEND_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def fetch_documents() -> list[dict]:
    data, err = api_get("/documents")
    if err:
        return []
    return data or []


def fetch_collection_info() -> dict:
    data, _ = api_get("/collection/info")
    return data or {}


def _truncate_filename(filename: str, max_chars: int = 30) -> str:
    if len(filename) <= max_chars:
        return filename
    name, ext = os.path.splitext(filename)
    reserved = len(ext) + 1
    visible = max(8, max_chars - reserved)
    return f"{name[:visible]}…{ext}"


def _chat_avatar(role: str) -> str:
    if role == "assistant":
        return ASSISTANT_AVATAR_PATH
    return USER_AVATAR_PATH


# ── session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 RAG Vector Memory")
    st.caption("Powered by Qdrant + OpenAI")
    st.divider()

    # ── upload section ────────────────────────────────────────────────────────
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if st.button("Upload & Ingest", use_container_width=True, type="primary"):
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            with st.spinner("Parsing, chunking and embedding…"):
                result, err = api_post(
                    "/ingest",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                )
            if err:
                st.error(f"Upload failed: {err}")
            else:
                st.success("Document ingested!")
                st.markdown(
                    f"- **File:** {result['filename']}\n"
                    f"- **Chunks created:** {result['total_chunks']}\n"
                    f"- **Vectors stored:** {result['total_vectors_stored']}"
                )
                st.rerun()

    st.divider()

    # ── document list ─────────────────────────────────────────────────────────
    st.subheader("📚 Ingested Documents")
    documents = fetch_documents()

    if not documents:
        st.info("No documents ingested yet.")
    else:
        for doc in documents:
            with st.container(border=True):
                name_col, action_col = st.columns(
                    [3, 1], gap="small", vertical_alignment="center"
                )
                with name_col:
                    safe_name = html.escape(doc["filename"])
                    display_name = html.escape(_truncate_filename(doc["filename"]))
                    st.markdown(
                        f"<p class='doc-title' title='{safe_name}'>{display_name}</p>",
                        unsafe_allow_html=True,
                    )
                with action_col:
                    if st.button(
                        "🗑️",
                        key=f"del_{doc['filename']}",
                        help=f"Delete {doc['filename']}",
                        use_container_width=False,
                    ):
                        with st.spinner("Deleting…"):
                            result, err = api_delete(
                                f"/documents/{requests.utils.quote(doc['filename'])}"
                            )
                        if err:
                            st.error(f"Delete failed: {err}")
                        else:
                            st.success(f"Deleted {doc['filename']}")
                            st.rerun()
                st.caption(f"{doc['chunk_count']} chunks")

    st.divider()

    # ── collection stats ──────────────────────────────────────────────────────
    st.subheader("📊 Collection Stats")
    info = fetch_collection_info()
    if info:
        st.metric("Total Vectors", info.get("total_vectors", "—"))
        st.caption(f"Status: {info.get('status', '—')}")
    else:
        st.info("Could not fetch stats.")

# ── main area ─────────────────────────────────────────────────────────────────

st.title("💬 Chat with your Documents")
st.caption(
    "Ask questions about your ingested files and review retrieval sources with confidence."
)

col_left, col_right = st.columns([4, 1], vertical_alignment="bottom")
with col_left:
    documents = fetch_documents()
    doc_names = [d["filename"] for d in documents]
    filter_options = ["All documents"] + doc_names
    selected_filter = st.selectbox(
        "Filter by document (optional)",
        options=filter_options,
        index=0,
    )
    filename_filter = None if selected_filter == "All documents" else selected_filter

with col_right:
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=5)

st.divider()

# ── chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=_chat_avatar(msg["role"])):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            sources = msg["sources"]
            total = msg.get("total_sources_found", len(sources))
            with st.expander(f"📎 Sources Used ({total} found)", expanded=False):
                for idx, source in enumerate(sources, 1):
                    score_pct = source["similarity_score"]
                    st.markdown(
                        f"**{idx}. {source['filename']}** — chunk `{source['chunk_index']}`"
                    )
                    st.caption(f"Similarity Score: {score_pct * 100:.1f}%")
                    st.progress(float(source["similarity_score"]))
                    st.markdown(f"> {source['text'][:300]}…")
                    if idx < len(sources):
                        st.markdown("---")

# ── chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR_PATH):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR_PATH):
        with st.spinner("Searching and generating answer…"):
            payload = {
                "question": prompt,
                "top_k": top_k,
                "filename_filter": filename_filter,
            }
            result, err = api_post("/query", json=payload)

        if err:
            answer = f"Error: {err}"
            st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            answer = result["answer"]
            sources = result.get("sources", [])
            total = result.get("total_sources_found", 0)

            st.markdown(answer)

            with st.expander(f"📎 Sources Used ({total} found)", expanded=True):
                if sources:
                    for idx, source in enumerate(sources, 1):
                        score_pct = source["similarity_score"]
                        st.markdown(
                            f"**{idx}. {source['filename']}** — chunk `{source['chunk_index']}`"
                        )
                        st.caption(f"Similarity Score: {score_pct * 100:.1f}%")
                        st.progress(float(source["similarity_score"]))
                        st.markdown(f"> {source['text'][:300]}…")
                        if idx < len(sources):
                            st.markdown("---")
                else:
                    st.info("No sources retrieved.")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "total_sources_found": total,
                }
            )
