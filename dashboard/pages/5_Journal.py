"""Journal search screen (doc 11 §3.6)."""

from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from streamlit_helpers import get_engine

from dynamo.domain.models.learning import JournalKind
from dynamo.persistence.repositories.learning import JournalRepository

st.set_page_config(page_title="Dynamo — Journal", layout="wide")
st.title("Journal")

c1, c2, c3 = st.columns(3)
query = c1.text_input("Search text")
kind = c2.selectbox("Kind", ["(any)"] + [k.value for k in JournalKind])
symbol = c3.text_input("Symbol").upper().strip()

try:
    results = JournalRepository(get_engine()).search(
        text_query=query or None,
        kinds=(JournalKind(kind),) if kind != "(any)" else (),
        symbols=(symbol,) if symbol else (),
        top_k=50,
    )
except SQLAlchemyError:
    st.warning("Database unavailable — no journal to search.")
    results = []

if not results:
    st.info("No journal entries match — the journal fills as the platform operates.")
for entry, _score in results:
    with st.container(border=True):
        st.markdown(f"**{entry.kind.value}** · {entry.created_at:%Y-%m-%d} · {entry.id}")
        if entry.symbols:
            st.caption("symbols: " + ", ".join(entry.symbols))
        st.markdown(entry.text)
        if entry.refs:
            st.caption(f"refs: {entry.refs}")
