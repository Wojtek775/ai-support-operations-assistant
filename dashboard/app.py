"""
dashboard/app.py — Streamlit dashboard for the AI Support & Operations Assistant.

Connects to the FastAPI backend at http://127.0.0.1:8000 (configurable via sidebar).
Displays system health, AI operations metrics, and recent ticket activity.
"""

import requests
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Support & Ops Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")

    backend_url = st.text_input(
        "Backend URL",
        value="http://127.0.0.1:8000",
        help="Base URL of the FastAPI backend.",
    )
    backend_url = backend_url.rstrip("/")

    st.caption(f"🔗 Currently connected to: `{backend_url}`")

    st.divider()

    refresh = st.button("🔄 Refresh Data", use_container_width=True)

    st.divider()

    st.markdown("### ℹ️ Project Info")
    st.markdown(
        """
**AI Support & Operations Assistant**

An AI-powered operational workflow engine
that processes support tickets through a
structured classification pipeline.

**API Endpoints**
- `GET /health` — system status
- `GET /metrics` — aggregated stats
- `GET /tickets` — all processed tickets
- `POST /tickets` — submit new ticket
- `GET /tickets/{id}` — single ticket
        """
    )

# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

TIMEOUT = 5  # seconds


def fetch_health(base_url: str) -> dict | None:
    """Call GET /health and return the JSON payload, or None on error."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def fetch_metrics(base_url: str) -> dict | None:
    """Call GET /metrics and return the JSON payload, or None on error."""
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_tickets(base_url: str) -> list | None:
    """Call GET /tickets and return the JSON list, or None on error."""
    try:
        resp = requests.get(f"{base_url}/tickets", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trigger a rerun when the refresh button is clicked
# (cache is busted automatically because Streamlit reruns the entire script)
# ---------------------------------------------------------------------------

if refresh:
    st.rerun()

# ---------------------------------------------------------------------------
# Fetch all data
# ---------------------------------------------------------------------------

health_data = fetch_health(backend_url)
metrics_data = fetch_metrics(backend_url)
tickets_data = fetch_tickets(backend_url)

backend_online = health_data is not None

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------

st.title("🤖 AI Support & Operations Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------------
# Connection error banner (shown when backend is completely unreachable)
# ---------------------------------------------------------------------------

if not backend_online:
    st.error(
        f"⚠️ **Cannot connect to backend at `{backend_url}`.**\n\n"
        "Make sure the FastAPI server is running:\n"
        "```\n"
        "cd ai-support-operations-assistant/backend\n"
        "uvicorn app.main:app --reload --port 8000\n"
        "```"
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 1: System Health
# ---------------------------------------------------------------------------

st.header("🏥 System Health")

if not backend_online:
    col_status, col_url = st.columns([1, 3])
    with col_status:
        st.metric("Backend Status", "🔴 Offline")
    with col_url:
        st.metric("Backend URL", backend_url)
else:
    col_status, col_version, col_url = st.columns([1, 1, 2])
    with col_status:
        st.metric("Backend Status", "🟢 Online")
    with col_version:
        version = health_data.get("version", "N/A") if isinstance(health_data, dict) else "N/A"
        st.metric("Version", version)
    with col_url:
        st.metric("Backend URL", backend_url)

    if isinstance(health_data, dict):
        with st.expander("Raw health response"):
            st.json(health_data)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: AI Operations Overview (Metrics)
# ---------------------------------------------------------------------------

st.header("📊 AI Operations Overview")

if metrics_data is None:
    st.warning("⚠️ Metrics unavailable — backend may be offline or `/metrics` returned an error.")
else:
    m = metrics_data

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="🎫 Total Tickets",
            value=m.get("total_tickets", 0),
        )
    with col2:
        st.metric(
            label="🚨 Urgent Tickets",
            value=m.get("urgent_tickets", 0),
        )
    with col3:
        st.metric(
            label="⬆️ High Priority",
            value=m.get("high_priority_tickets", 0),
        )
    with col4:
        st.metric(
            label="😠 Negative Sentiment",
            value=m.get("negative_sentiment_tickets", 0),
        )
    with col5:
        top_cat = m.get("top_category") or "—"
        st.metric(
            label="🏆 Top Category",
            value=top_cat.capitalize() if top_cat != "—" else "—",
        )

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Recent Tickets
# ---------------------------------------------------------------------------

st.header("🗂️ Recent Tickets")

if tickets_data is None:
    st.warning("⚠️ Tickets unavailable — backend may be offline or `/tickets` returned an error.")
elif len(tickets_data) == 0:
    st.info(
        "📭 **No tickets processed yet.**\n\n"
        "Submit a ticket via `/docs` or `POST /tickets` to populate the dashboard."
    )
else:
    # Build a display-friendly list (most recent first, cap at 50)
    display_tickets = []
    for t in tickets_data:
        processed_at = t.get("processed_at", "")
        # Normalise ISO timestamp for display
        try:
            dt = datetime.fromisoformat(processed_at.replace("Z", "+00:00"))
            processed_at_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            processed_at_str = processed_at

        display_tickets.append(
            {
                "Ticket ID": t.get("ticket_id", ""),
                "Customer": t.get("customer_name", ""),
                "Email": t.get("email", ""),
                "Classification": t.get("classification", "").capitalize(),
                "Priority": t.get("priority", "").capitalize(),
                "Sentiment": t.get("sentiment", "").capitalize(),
                "Processed At": processed_at_str,
            }
        )

    # Sort by processed_at descending (string sort works for ISO dates)
    display_tickets.sort(key=lambda x: x["Processed At"], reverse=True)
    display_tickets = display_tickets[:50]

    st.caption(f"Showing {len(display_tickets)} most recent ticket(s).")

    # Priority colour helper for the table
    priority_colours = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🟢",
    }

    for ticket in display_tickets:
        p = ticket["Priority"]
        icon = priority_colours.get(p, "⚪")
        ticket["Priority"] = f"{icon} {p}"

    st.dataframe(
        display_tickets,
        use_container_width=True,
        hide_index=True,
    )
