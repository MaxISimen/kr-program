"""
Kirpenko Firewall Analyzer — Streamlit Edition

Coursework topic:
Analysis of firewall logs for detecting port scanning attacks.

Discipline: Algorithmization and Programming

Run:
    streamlit run kirpenko_firewall_analyzer_streamlit.py
"""

import io
import json
import random
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Kirpenko Firewall Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .metric-card {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 12px 16px;
            border-left: 4px solid #7f5af0;
        }
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

WELL_KNOWN_PORTS: dict[int, str] = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    443: "HTTPS",
    445: "SMB",
    3306: "MySQL",
    3389: "RDP",
    8080: "HTTP-Alt",
    8443: "HTTPS-Alt",
}

RISK_ICONS: dict[str, str] = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
}


# ============================================================
# DATA GENERATION
# ============================================================

@st.cache_data(show_spinner=False)
def generate_firewall_logs(
    n_normal: int = 800,
    n_attack: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic firewall logs (normal traffic + port-scan attacks)."""

    random.seed(seed)
    np.random.seed(seed)

    base_time = datetime(2025, 1, 15, 8, 0, 0)
    records: list[dict] = []

    # ── Normal traffic ─────────────────────────────────────
    normal_src_ips = [f"192.168.1.{i}" for i in range(10, 50)]
    normal_dst_ips = [
        "10.0.0.1", "10.0.0.2", "10.0.0.5",
        "8.8.8.8", "172.16.0.1", "172.16.0.2",
    ]
    normal_ports = [80, 443, 22, 53, 8080, 8443, 3306, 3389]

    for _ in range(n_normal):
        ts = base_time + timedelta(seconds=random.randint(0, 7200))
        proto = random.choice(["TCP", "UDP"])
        records.append(
            {
                "timestamp": ts,
                "src_ip": random.choice(normal_src_ips),
                "dst_ip": random.choice(normal_dst_ips),
                "src_port": random.randint(1024, 65535),
                "dst_port": random.choice(normal_ports),
                "protocol": proto,
                "action": random.choices(["ALLOW", "DENY"], weights=[0.9, 0.1])[0],
                "flags": "SYN,ACK" if proto == "TCP" else "",
                "bytes": random.randint(64, 1500),
                "label": "normal",
            }
        )

    # ── Attack traffic ─────────────────────────────────────
    attack_ips = ["203.0.113.10", "198.51.100.5", "192.0.2.100"]

    for _ in range(n_attack):
        attacker = random.choice(attack_ips)
        target = random.choice(["10.0.0.1", "10.0.0.2"])
        attack_start = base_time + timedelta(
            seconds=random.choice([300, 1200, 2100, 3600])
        )
        ts = attack_start + timedelta(seconds=random.randint(0, 30))
        records.append(
            {
                "timestamp": ts,
                "src_ip": attacker,
                "dst_ip": target,
                "src_port": random.randint(40000, 65535),
                "dst_port": random.randint(1, 1024),
                "protocol": "TCP",
                "action": "DENY",
                "flags": random.choice(["SYN", "SYN", "SYN", "FIN", "NULL", "XMAS"]),
                "bytes": random.randint(40, 60),
                "label": "scan",
            }
        )

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# CSV I/O
# ============================================================

def load_logs_from_csv(file) -> pd.DataFrame:
    """Loads firewall logs from an uploaded CSV file."""

    df = pd.read_csv(file, parse_dates=["timestamp"])

    required_cols = {"timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "action"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["src_port", "dst_port", "bytes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to CSV bytes (for download button)."""
    return df.drop(columns=["label"], errors="ignore").to_csv(index=False).encode("utf-8")


# ============================================================
# DETECTION ALGORITHMS
# ============================================================

def detect_horizontal_scan(
    df: pd.DataFrame,
    window_sec: int,
    threshold: int,
) -> pd.DataFrame:
    """Detects horizontal port scanning: one source scans many ports on one host."""

    data = df.copy()
    data["window"] = data["timestamp"].dt.floor(f"{window_sec}s")

    grouped = (
        data.groupby(["src_ip", "dst_ip", "window"])
        .agg(
            unique_ports=("dst_port", "nunique"),
            total_packets=("dst_port", "count"),
            denied_packets=("action", lambda x: (x == "DENY").sum()),
        )
        .reset_index()
    )

    suspects = grouped[grouped["unique_ports"] >= threshold].copy()
    suspects["scan_type"] = "Horizontal Scan"
    suspects["risk_level"] = suspects["unique_ports"].apply(
        lambda v: "CRITICAL" if v > 50 else ("HIGH" if v > 30 else "MEDIUM")
    )
    return suspects


def detect_syn_scan(df: pd.DataFrame, ratio_threshold: float) -> pd.DataFrame:
    """Detects TCP SYN scanning by analysing the SYN-only packet ratio."""

    tcp_df = df[df["protocol"] == "TCP"].copy()
    if tcp_df.empty:
        return pd.DataFrame()

    tcp_df["is_syn_only"] = tcp_df["flags"].apply(
        lambda v: str(v).strip().upper() == "SYN"
    )

    grouped = (
        tcp_df.groupby("src_ip")
        .agg(
            total_tcp=("flags", "count"),
            syn_only_count=("is_syn_only", "sum"),
        )
        .reset_index()
    )
    grouped["syn_ratio"] = grouped["syn_only_count"] / grouped["total_tcp"]

    suspects = grouped[
        (grouped["syn_ratio"] >= ratio_threshold) & (grouped["total_tcp"] >= 10)
    ].copy()
    suspects["scan_type"] = "TCP SYN Scan"
    suspects["risk_level"] = "HIGH"
    return suspects


def detect_vertical_scan(
    df: pd.DataFrame,
    window_sec: int,
    threshold: int = 5,
) -> pd.DataFrame:
    """Detects vertical scanning: one source checks one port on many hosts."""

    data = df.copy()
    data["window"] = data["timestamp"].dt.floor(f"{window_sec}s")

    grouped = (
        data.groupby(["src_ip", "dst_port", "window"])
        .agg(
            unique_hosts=("dst_ip", "nunique"),
            total_packets=("dst_ip", "count"),
        )
        .reset_index()
    )

    suspects = grouped[grouped["unique_hosts"] >= threshold].copy()
    suspects["scan_type"] = "Vertical Scan"
    suspects["risk_level"] = suspects["unique_hosts"].apply(
        lambda v: "HIGH" if v > 20 else "MEDIUM"
    )
    return suspects


def detect_slow_scan(df: pd.DataFrame, port_threshold: int) -> pd.DataFrame:
    """Detects slow scanning: many ports scanned over an extended period."""

    data = df.copy()
    data["hour"] = data["timestamp"].dt.floor("1h")

    grouped = (
        data.groupby(["src_ip", "hour"])
        .agg(
            unique_ports=("dst_port", "nunique"),
            total_packets=("dst_port", "count"),
            time_span_sec=(
                "timestamp",
                lambda x: (x.max() - x.min()).total_seconds(),
            ),
        )
        .reset_index()
    )

    suspects = grouped[
        (grouped["unique_ports"] >= port_threshold) & (grouped["time_span_sec"] > 300)
    ].copy()
    suspects["scan_type"] = "Slow Scan"
    suspects["risk_level"] = "MEDIUM"
    return suspects


def run_all_detectors(
    df: pd.DataFrame,
    window_sec: int,
    threshold_ports: int,
    syn_ratio: float,
    slow_threshold: int,
) -> dict:
    """Runs all four detection algorithms and collects suspicious IPs."""

    results = {
        "horizontal": detect_horizontal_scan(df, window_sec, threshold_ports),
        "syn_scan":   detect_syn_scan(df, syn_ratio),
        "vertical":   detect_vertical_scan(df, window_sec),
        "slow_scan":  detect_slow_scan(df, slow_threshold),
    }

    suspect_ips: set[str] = set()
    for result in results.values():
        if isinstance(result, pd.DataFrame) and not result.empty and "src_ip" in result.columns:
            suspect_ips.update(result["src_ip"].unique())

    results["suspect_ips"] = suspect_ips
    return results


# ============================================================
# STATISTICS
# ============================================================

def compute_statistics(df: pd.DataFrame) -> dict:
    """Computes general firewall log statistics."""

    return {
        "total_records":    len(df),
        "unique_src_ips":   df["src_ip"].nunique(),
        "unique_dst_ips":   df["dst_ip"].nunique(),
        "unique_dst_ports": df["dst_port"].nunique(),
        "denied_pct":       round((df["action"] == "DENY").sum() / len(df) * 100, 2),
        "protocols":        df["protocol"].value_counts().to_dict(),
        "top_ports":        df["dst_port"].value_counts().head(10).to_dict(),
        "top_src_ips":      df["src_ip"].value_counts().head(5).to_dict(),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end":   str(df["timestamp"].max()),
        },
    }


def classify_port(port: int) -> str:
    """Classifies a destination port number."""

    if port in WELL_KNOWN_PORTS:
        return WELL_KNOWN_PORTS[port]
    if port < 1024:
        return "Well-Known"
    if port < 49152:
        return "Registered"
    return "Dynamic/Private"


# ============================================================
# VISUALISATIONS  (return plt.Figure, no file I/O)
# ============================================================

def plot_traffic_timeline(df: pd.DataFrame) -> plt.Figure:
    """Traffic timeline split by normal / scan traffic and DENY counts."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Firewall Traffic Timeline", fontsize=14, fontweight="bold")

    if "label" in df.columns:
        normal = df[df["label"] == "normal"]
        scan   = df[df["label"] == "scan"]

        normal_s = normal.set_index("timestamp").resample("5min").size()
        scan_s   = scan.set_index("timestamp").resample("5min").size()

        ax1.fill_between(normal_s.index, normal_s.values, alpha=0.7, label="Normal traffic")
        ax1.fill_between(scan_s.index,   scan_s.values,   alpha=0.8, label="Scanning attacks",
                         color="tomato")
    else:
        total_s = df.set_index("timestamp").resample("5min").size()
        ax1.fill_between(total_s.index, total_s.values, alpha=0.7, label="Traffic")

    ax1.set_ylabel("Packets")
    ax1.legend()
    ax1.grid(alpha=0.3)

    deny_df = df[df["action"] == "DENY"]
    if not deny_df.empty:
        deny_s = deny_df.set_index("timestamp").resample("5min").size()
        ax2.bar(deny_s.index, deny_s.values, width=0.003, alpha=0.9,
                color="crimson", label="DENY packets")

    ax2.set_ylabel("DENY packets")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig


def plot_port_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of the top-15 destination ports."""

    fig, ax = plt.subplots(figsize=(12, 5))
    top_ports = df["dst_port"].value_counts().head(15)
    bars = ax.bar([str(p) for p in top_ports.index], top_ports.values, color="steelblue")
    ax.bar_label(bars, padding=2)
    ax.set_title("Top Destination Ports")
    ax.set_xlabel("Destination port")
    ax.set_ylabel("Number of requests")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scan_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Activity heatmap: top-10 source IPs by hour."""

    top_ips = df["src_ip"].value_counts().head(10).index
    data = df[df["src_ip"].isin(top_ips)].copy()
    data["hour"] = data["timestamp"].dt.hour

    pivot = data.pivot_table(
        index="src_ip", columns="hour",
        values="dst_port", aggfunc="count", fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    img = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{h}:00" for h in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Source IP Activity Heatmap")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Source IP")

    plt.colorbar(img, ax=ax, label="Packets")
    plt.tight_layout()
    return fig


def plot_detection_summary(results: dict) -> plt.Figure:
    """Bar chart summarising all detector hit counts."""

    names = {
        "horizontal": "Horizontal Scan",
        "syn_scan":   "TCP SYN Scan",
        "vertical":   "Vertical Scan",
        "slow_scan":  "Slow Scan",
    }
    counts = {
        key: len(val)
        for key, val in results.items()
        if key in names and isinstance(val, pd.DataFrame)
    }
    colors = ["#ef4444", "#f97316", "#eab308", "#3b82f6"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([names[k] for k in counts], list(counts.values()), color=colors[:len(counts)])
    ax.bar_label(bars, padding=3)
    ax.set_title("Detection Results Summary")
    ax.set_ylabel("Detected events")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_protocol_pie(df: pd.DataFrame) -> plt.Figure:
    """Pie chart of protocol distribution."""

    proto_counts = df["protocol"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(proto_counts.values, labels=proto_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Protocol Distribution")
    plt.tight_layout()
    return fig


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report_text(df: pd.DataFrame, results: dict, stats: dict) -> str:
    """Builds a plain-text analysis report string."""

    detector_names = {
        "horizontal": "Horizontal Scan",
        "syn_scan":   "TCP SYN Scan",
        "vertical":   "Vertical Scan",
        "slow_scan":  "Slow Scan",
    }

    lines = [
        "=" * 70,
        " FIREWALL LOG ANALYSIS REPORT",
        " Port Scanning Detection System",
        "=" * 70,
        "",
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "GENERAL STATISTICS",
        "-" * 70,
        f"Total records:              {stats['total_records']}",
        f"Unique source IPs:          {stats['unique_src_ips']}",
        f"Unique destination IPs:     {stats['unique_dst_ips']}",
        f"Unique destination ports:   {stats['unique_dst_ports']}",
        f"DENY packets percentage:    {stats['denied_pct']}%",
        f"Log start:                  {stats['date_range']['start']}",
        f"Log end:                    {stats['date_range']['end']}",
        "",
        "DETECTION RESULTS",
        "-" * 70,
    ]

    total_events = 0
    for key, name in detector_names.items():
        result = results.get(key, pd.DataFrame())
        count = len(result) if isinstance(result, pd.DataFrame) else 0
        total_events += count
        lines.append(f"{name:<30} {count:>5} events")

    lines.extend(
        [
            "",
            f"Total suspicious events:    {total_events}",
            f"Suspicious IP addresses:    {len(results.get('suspect_ips', set()))}",
            "",
            "SUSPICIOUS IP ADDRESSES",
            "-" * 70,
        ]
    )

    for ip in sorted(results.get("suspect_ips", set())):
        ip_recs = df[df["src_ip"] == ip]
        lines.append(
            f"{ip:<18} packets: {len(ip_recs):<5} unique ports: {ip_recs['dst_port'].nunique():<5}"
        )

    lines.extend(
        [
            "",
            "TOP ACTIVE SOURCE IP ADDRESSES",
            "-" * 70,
        ]
    )
    for ip, cnt in list(stats["top_src_ips"].items())[:5]:
        lines.append(f"{ip:<18} {cnt:>5} packets")

    lines.extend(["", "=" * 70, " End of report", "=" * 70])
    return "\n".join(lines)


def generate_json_summary(stats: dict, results: dict) -> str:
    """Serialises key findings to a JSON string."""

    summary = {
        "statistics": stats,
        "suspect_ips": sorted(list(results.get("suspect_ips", set()))),
        "detected_events": {
            key: len(val)
            for key, val in results.items()
            if isinstance(val, pd.DataFrame)
        },
    }
    return json.dumps(summary, indent=4, ensure_ascii=False)


# ============================================================
# UI HELPERS
# ============================================================

def risk_badge(level: str) -> str:
    icon = RISK_ICONS.get(level, "⚪")
    return f"{icon} {level}"


def show_detector_results(results: dict, key: str) -> None:
    """Renders a detection result DataFrame inside a Streamlit expander."""

    result = results.get(key, pd.DataFrame())
    if isinstance(result, pd.DataFrame) and not result.empty:
        display = result.copy()
        if "risk_level" in display.columns:
            display["risk_level"] = display["risk_level"].apply(risk_badge)
        if "window" in display.columns:
            display["window"] = display["window"].astype(str)
        if "hour" in display.columns:
            display["hour"] = display["hour"].astype(str)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(f"Total events detected: **{len(result)}**")
    else:
        st.success("✅ No suspicious events detected by this algorithm.")


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main() -> None:

    # ── Header ────────────────────────────────────────────────
    st.title("🔥 Kirpenko Firewall Analyzer")
    st.caption(
        "Analysis of firewall logs for detecting port scanning attacks · "
        "Discipline: Algorithmization and Programming"
    )

    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.header("⚙️ Detection Thresholds")

    threshold_ports = st.sidebar.slider(
        "Unique ports threshold (Horizontal Scan)",
        min_value=5, max_value=100, value=15, step=1,
        help="Minimum number of unique destination ports in one window to flag as scanning.",
    )
    session_window = st.sidebar.slider(
        "Session window (seconds)",
        min_value=10, max_value=300, value=60, step=10,
        help="Time window used for grouping traffic into sessions.",
    )
    syn_ratio = st.sidebar.slider(
        "SYN ratio threshold",
        min_value=0.50, max_value=1.00, value=0.85, step=0.05,
        help="Minimum SYN-only packet ratio to flag as a SYN scan.",
    )
    slow_threshold = st.sidebar.slider(
        "Slow scan port threshold",
        min_value=10, max_value=100, value=30, step=5,
        help="Minimum unique ports over 1 hour to flag as a slow scan.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("📂 Data Source")

    data_source = st.sidebar.radio(
        "Select source",
        ["🔧 Generate test data", "📤 Upload CSV file"],
        label_visibility="collapsed",
    )

    # ── Data loading / generation ─────────────────────────────
    if data_source == "🔧 Generate test data":
        col_a, col_b = st.sidebar.columns(2)
        n_normal = col_a.number_input("Normal", min_value=100, max_value=5000, value=800, step=100)
        n_attack = col_b.number_input("Attack",  min_value=10,  max_value=1000, value=200, step=50)
        seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)

        run_btn = st.sidebar.button(
            "🚀 Generate & Analyze", use_container_width=True, type="primary"
        )
        if run_btn:
            with st.spinner("Generating logs and running detectors …"):
                df_gen = generate_firewall_logs(n_normal=int(n_normal), n_attack=int(n_attack), seed=int(seed))
                st.session_state["df"]      = df_gen
                st.session_state["results"] = run_all_detectors(
                    df_gen, session_window, threshold_ports, syn_ratio, slow_threshold
                )
                st.session_state["stats"] = compute_statistics(df_gen)
            st.sidebar.success(f"✅ {len(df_gen)} records generated")

    else:
        uploaded = st.sidebar.file_uploader(
            "Upload firewall CSV", type=["csv"],
            help="CSV must contain: timestamp, src_ip, dst_ip, src_port, dst_port, protocol, action",
        )
        if uploaded:
            try:
                df_up = load_logs_from_csv(uploaded)
                run_btn2 = st.sidebar.button(
                    "🔍 Run Analysis", use_container_width=True, type="primary"
                )
                if run_btn2:
                    with st.spinner("Running detectors …"):
                        st.session_state["df"]      = df_up
                        st.session_state["results"] = run_all_detectors(
                            df_up, session_window, threshold_ports, syn_ratio, slow_threshold
                        )
                        st.session_state["stats"] = compute_statistics(df_up)
                    st.sidebar.success(f"✅ {len(df_up)} records loaded")
            except ValueError as exc:
                st.sidebar.error(str(exc))

    # ── Guard: nothing to show yet ─────────────────────────────
    if "df" not in st.session_state:
        st.info(
            "👈 **Configure** the data source in the sidebar, then click "
            "**Generate & Analyze** (or **Run Analysis**) to start."
        )
        return

    df      = st.session_state["df"]
    results = st.session_state["results"]
    stats   = st.session_state["stats"]

    # ── Top KPI row ────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("📦 Records",        stats["total_records"])
    k2.metric("🌐 Source IPs",     stats["unique_src_ips"])
    k3.metric("🎯 Dest IPs",       stats["unique_dst_ips"])
    k4.metric("🚪 Unique ports",   stats["unique_dst_ports"])
    k5.metric("🚫 DENY %",         f"{stats['denied_pct']}%")
    k6.metric("🚨 Suspect IPs",    len(results.get("suspect_ips", set())))

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────
    tab_overview, tab_detect, tab_charts, tab_data, tab_report = st.tabs(
        ["📊 Overview", "🚨 Detections", "📈 Charts", "🗒️ Raw Data", "📄 Report"]
    )

    # ──────────────────────────────────────────────────────────
    # TAB 1 · OVERVIEW
    # ──────────────────────────────────────────────────────────
    with tab_overview:
        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            st.subheader("Detection Summary")
            st.pyplot(plot_detection_summary(results), use_container_width=True)

        with col_right:
            st.subheader("🚨 Suspicious IP Addresses")
            suspect_ips = results.get("suspect_ips", set())
            if suspect_ips:
                for ip in sorted(suspect_ips):
                    ip_data = df[df["src_ip"] == ip]
                    st.error(
                        f"**{ip}**   packets: `{len(ip_data)}`   "
                        f"unique ports: `{ip_data['dst_port'].nunique()}`"
                    )
            else:
                st.success("✅ No suspicious IPs detected.")

        st.divider()

        col_p, col_q = st.columns(2)
        with col_p:
            st.subheader("Protocol Distribution")
            st.pyplot(plot_protocol_pie(df), use_container_width=True)

        with col_q:
            st.subheader("Top 5 Source IPs (by packet count)")
            top_df = (
                pd.DataFrame.from_dict(stats["top_src_ips"], orient="index", columns=["packets"])
                .rename_axis("ip")
                .reset_index()
            )
            st.bar_chart(top_df.set_index("ip")["packets"])

        st.subheader("Log Time Range")
        st.markdown(
            f"**From:** `{stats['date_range']['start']}`  &nbsp;&nbsp;→&nbsp;&nbsp;  "
            f"**To:** `{stats['date_range']['end']}`"
        )

    # ──────────────────────────────────────────────────────────
    # TAB 2 · DETECTIONS
    # ──────────────────────────────────────────────────────────
    with tab_detect:
        st.subheader("Detection Algorithm Results")

        detector_meta = {
            "horizontal": (
                "🔵 Horizontal Scan",
                "One source IP probes many ports on a single destination host.",
            ),
            "syn_scan": (
                "🔴 TCP SYN Scan",
                "Source sends a high ratio of TCP SYN-only packets (half-open scan).",
            ),
            "vertical": (
                "🟠 Vertical Scan",
                "One source IP probes the same port across many destination hosts.",
            ),
            "slow_scan": (
                "🟡 Slow Scan",
                "Many destination ports scanned slowly over an extended time window.",
            ),
        }

        for key, (title, description) in detector_meta.items():
            result = results.get(key, pd.DataFrame())
            count = len(result) if isinstance(result, pd.DataFrame) and not result.empty else 0
            badge = f"**{count} events**" if count else "clean"
            with st.expander(f"{title} — {description} · {badge}", expanded=(count > 0)):
                show_detector_results(results, key)

    # ──────────────────────────────────────────────────────────
    # TAB 3 · CHARTS
    # ──────────────────────────────────────────────────────────
    with tab_charts:
        st.subheader("Traffic Timeline")
        st.pyplot(plot_traffic_timeline(df), use_container_width=True)

        st.subheader("Top Destination Ports")
        st.pyplot(plot_port_distribution(df), use_container_width=True)

        st.subheader("Source IP Activity Heatmap (top 10 IPs × hour)")
        st.pyplot(plot_scan_heatmap(df), use_container_width=True)

    # ──────────────────────────────────────────────────────────
    # TAB 4 · RAW DATA
    # ──────────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Firewall Log Records")

        fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
        action_opts  = df["action"].unique().tolist()
        proto_opts   = df["protocol"].unique().tolist()
        action_filt  = fc1.multiselect("Action",   action_opts, default=action_opts)
        proto_filt   = fc2.multiselect("Protocol", proto_opts,  default=proto_opts)
        port_filt    = fc3.number_input("Dst port (0 = all)", min_value=0, max_value=65535, value=0)
        ip_filt      = fc4.text_input("Source IP contains")

        filtered = df[df["action"].isin(action_filt) & df["protocol"].isin(proto_filt)]
        if port_filt:
            filtered = filtered[filtered["dst_port"] == port_filt]
        if ip_filt:
            filtered = filtered[filtered["src_ip"].str.contains(ip_filt, na=False)]

        st.info(f"Showing **{len(filtered)}** of **{len(df)}** records")
        st.dataframe(filtered, use_container_width=True, height=420, hide_index=True)

        st.download_button(
            label="⬇️ Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="firewall_logs_filtered.csv",
            mime="text/csv",
        )

    # ──────────────────────────────────────────────────────────
    # TAB 5 · REPORT
    # ──────────────────────────────────────────────────────────
    with tab_report:
        st.subheader("Plain-Text Analysis Report")

        report_text = generate_report_text(df, results, stats)
        st.code(report_text, language="text")

        json_text = generate_json_summary(stats, results)

        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            label="⬇️ Download TXT Report",
            data=report_text.encode("utf-8"),
            file_name="analysis_report.txt",
            mime="text/plain",
        )
        dl2.download_button(
            label="⬇️ Download JSON Summary",
            data=json_text.encode("utf-8"),
            file_name="summary.json",
            mime="application/json",
        )
        dl3.download_button(
            label="⬇️ Download Full CSV",
            data=df_to_csv_bytes(df),
            file_name="firewall_logs.csv",
            mime="text/csv",
        )

        with st.expander("🔍 JSON Summary Preview"):
            st.json(json.loads(json_text))


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
