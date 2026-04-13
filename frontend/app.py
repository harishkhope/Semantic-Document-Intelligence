import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Semantic Document Intelligence",
    page_icon="🧠",
    layout="wide",
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

# ── session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 RAG Vector Memory")
    st.caption("Powered by Qdrant + Gemini")
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
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
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
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{doc['filename']}**  \n`{doc['chunk_count']} chunks`")
            with col2:
                if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Delete {doc['filename']}"):
                    with st.spinner("Deleting…"):
                        result, err = api_delete(f"/documents/{requests.utils.quote(doc['filename'])}")
                    if err:
                        st.error(f"Delete failed: {err}")
                    else:
                        st.success(f"Deleted {doc['filename']}")
                        st.rerun()

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

col_left, col_right = st.columns([3, 1])
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
    with st.chat_message(msg["role"]):
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
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
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
