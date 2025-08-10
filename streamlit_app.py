"""
Smart Green Commute Tracker - Streamlit App
Author: Professional-style implementation (for Boyina Sankar)
Purpose: Track commutes, compute emissions, show dashboards, leaderboard, and badges.
Single-file for easy start; modular functions to extend into multi-file structure later.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from dateutil.parser import parse as dtparse
from haversine import haversine, Unit
import plotly.express as px
import os
import io

# ----------------------------
# CONFIG / EMISSION FACTORS
# ----------------------------
DB_PATH = "sgct_db.sqlite3"

# Emission factors in kg CO2 per km (typical reference ranges; change if you prefer other sources)
EMISSION_FACTORS = {
    "car_petrol": 0.192,   # kg CO2 / km
    "car_diesel": 0.171,
    "bus": 0.089,
    "train": 0.041,
    "motorbike": 0.103,
    "taxi": 0.192,
    "ev_small": 0.05,      # includes upstream electricity; adjust per country
    "bike": 0.0,
    "walk": 0.0,
    "e-bike": 0.012
}

# Default baseline mode for avoided CO2 calculation (the mode user would have used otherwise)
BASELINE_MODE = "car_petrol"

# Gamification thresholds (kg CO2 avoided)
BADGE_THRESHOLDS = [
    ("Green Starter", 1),
    ("Eco Walker", 5),
    ("Sustainable Commuter", 20),
    ("Climate Champion", 100),
]

# ----------------------------
# DB HELPERS
# ----------------------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            display_name TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            start_time TEXT,
            end_time TEXT,
            mode TEXT,
            distance_km REAL,
            duration_min REAL,
            emission_kg REAL,
            baseline_emission_kg REAL,
            start_lat REAL,
            start_lon REAL,
            end_lat REAL,
            end_lon REAL,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    return conn

def get_user(conn, username):
    c = conn.cursor()
    c.execute("SELECT id, username, display_name FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if row:
        return {"id": row[0], "username": row[1], "display_name": row[2]}
    return None

def create_user(conn, username, display_name=None):
    display_name = display_name or username
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    try:
        c.execute("INSERT INTO users (username, display_name, created_at) VALUES (?, ?, ?)", (username, display_name, now))
        conn.commit()
        return get_user(conn, username)
    except sqlite3.IntegrityError:
        return get_user(conn, username)

def add_trip(conn, user_id, date, start_time, end_time, mode, distance_km, duration_min,
             emission_kg, baseline_emission_kg, start_lat, start_lon, end_lat, end_lon, notes):
    c = conn.cursor()
    c.execute("""
        INSERT INTO trips (user_id, date, start_time, end_time, mode, distance_km, duration_min,
                           emission_kg, baseline_emission_kg, start_lat, start_lon, end_lat, end_lon, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, date, start_time, end_time, mode, distance_km, duration_min,
        emission_kg, baseline_emission_kg, start_lat, start_lon, end_lat, end_lon, notes
    ))
    conn.commit()

def fetch_trips_df(conn):
    df = pd.read_sql_query("SELECT t.*, u.username, u.display_name FROM trips t JOIN users u ON t.user_id = u.id", conn)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    return df

# ----------------------------
# UTILS
# ----------------------------
def compute_distance_km(start_lat, start_lon, end_lat, end_lon):
    if any(v is None for v in [start_lat, start_lon, end_lat, end_lon]):
        return 0.0
    return haversine((start_lat, start_lon), (end_lat, end_lon), unit=Unit.KILOMETERS)

def compute_duration_minutes(start_time_iso, end_time_iso):
    try:
        start = dtparse(start_time_iso)
        end = dtparse(end_time_iso)
        delta = end - start
        return max(delta.total_seconds() / 60.0, 0.0)
    except Exception:
        return 0.0

def emission_for_mode(mode_key, distance_km):
    factor = EMISSION_FACTORS.get(mode_key, 0.0)
    # step-by-step arithmetic: emission = distance_km * factor
    return float(distance_km) * float(factor)

def baseline_emission(distance_km, baseline_mode=BASELINE_MODE):
    return emission_for_mode(baseline_mode, distance_km)

def detect_mode_from_trace(avg_speed_kmph):
    """
    Heuristic mode detection based on average speed.
    This is simplistic but useful for auto-detection when we have GPS traces.
    """
    if avg_speed_kmph < 7:
        return "walk"
    elif avg_speed_kmph < 20:
        return "bike"
    elif avg_speed_kmph < 40:
        return "motorbike"
    else:
        # could be bus/train/car depending on context; default to car_petrol
        return "car_petrol"

def assign_badges(total_avoided_kg):
    badges = []
    for name, thresh in BADGE_THRESHOLDS:
        if total_avoided_kg >= thresh:
            badges.append(name)
    return badges

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Smart Green Commute Tracker", layout="wide", initial_sidebar_state="expanded")
st.title("🌿 Smart Green Commute Tracker (Streamlit Edition)")

# small CSS for nicer look
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #e6eef8; }
    .block-container { padding-top: 1rem; padding-left: 2rem; padding-right: 2rem;}
    h1, h2, h3 { color: #e6eef8; }
    .metric { background: linear-gradient(90deg, rgba(34,197,94,0.1), rgba(16,185,129,0.05)); padding: 8px; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Initialize DB connection
conn = init_db(DB_PATH)

# Sidebar: Login / Create user
with st.sidebar:
    st.header("User")
    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        col1, col2 = st.columns([3,1])
        with col1:
            username = st.text_input("Username (unique)", key="login_username")
        with col2:
            display = st.text_input("Display name", key="login_display")
        if st.button("Create / Login"):
            if not username:
                st.warning("Enter a username.")
            else:
                user = get_user(conn, username)
                if not user:
                    user = create_user(conn, username, display_name=display or username)
                st.session_state.user = user
                st.success(f"Logged in as {user['display_name']}")
    else:
        st.markdown(f"**Logged in:** {st.session_state.user['display_name']}  \n@{st.session_state.user['username']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.header("Quick Metrics")
df_all = fetch_trips_df(conn)
if not df_all.empty:
    total_co2 = float(df_all['emission_kg'].sum())
    total_baseline = float(df_all['baseline_emission_kg'].sum())
    avoided = max(total_baseline - total_co2, 0.0)
    st.metric("Total Emissions (kg CO₂)", f"{total_co2:.2f}")
    st.metric("Baseline Emissions (kg CO₂)", f"{total_baseline:.2f}")
    st.metric("Estimated Avoided CO₂ (kg)", f"{avoided:.2f}")
else:
    st.write("No trips logged yet.")

# Main tabs
tabs = st.tabs(["Log Commute", "Dashboard", "Leaderboard", "Data / Export", "Settings / Integrations"])

# -------- Tab: Log Commute -----------
with tabs[0]:
    st.header("Log a Commute")
    if st.session_state.user is None:
        st.info("Please create/login a user from the sidebar to log trips.")
    else:
        with st.form("commute_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                date_input = st.date_input("Trip date", value=datetime.utcnow().date())
                start_time = st.time_input("Start time", value=datetime.utcnow().time().replace(second=0, microsecond=0))
                end_time = st.time_input("End time", value=(datetime.utcnow() + timedelta(minutes=20)).time().replace(second=0, microsecond=0))
                mode_choice = st.selectbox("Mode of transport", ["walk","bike","e-bike","car_petrol","car_diesel","motorbike","bus","train","ev_small","taxi"], index=3)
                distance_choice = st.radio("Distance input", ["manual (km)", "start/end coords", "upload GPS CSV (timestamp,lat,lon)"], index=1)
            with col2:
                notes = st.text_area("Notes (optional)", height=110)
                # manual distance
                manual_km = st.number_input("Distance (km) — manual", min_value=0.0, value=0.0, step=0.1)
                # coords
                start_lat = st.number_input("Start lat", value=0.0, format="%.6f")
                start_lon = st.number_input("Start lon", value=0.0, format="%.6f")
                end_lat = st.number_input("End lat", value=0.0, format="%.6f")
                end_lon = st.number_input("End lon", value=0.0, format="%.6f")
                uploaded = st.file_uploader("Upload CSV trace (timestamp,lat,lon)", type=["csv"])
            submitted = st.form_submit_button("Add Trip")

        # Process submission
        if submitted:
            # determine distance and duration
            distance_km = 0.0
            duration_min = 0.0
            detected_mode = None

            # Option: GPS CSV
            if distance_choice == "upload GPS CSV (timestamp,lat,lon)" and uploaded is not None:
                try:
                    trace_df = pd.read_csv(uploaded)
                    # expect columns: timestamp, lat, lon (case-insensitive)
                    cols = {c.lower(): c for c in trace_df.columns}
                    if 'timestamp' not in cols or 'lat' not in cols or 'lon' not in cols:
                        st.error("CSV must contain 'timestamp', 'lat', 'lon' columns (case-insensitive).")
                    else:
                        trace_df = trace_df.rename(columns={cols['timestamp']:'timestamp', cols['lat']:'lat', cols['lon']:'lon'})
                        trace_df = trace_df.sort_values('timestamp')
                        # compute total distance
                        total_dist = 0.0
                        prev = None
                        for _, row in trace_df.iterrows():
                            if prev is not None:
                                a = (prev['lat'], prev['lon'])
                                b = (row['lat'], row['lon'])
                                total_dist += haversine(a, b, unit=Unit.KILOMETERS)
                            prev = row
                        distance_km = float(total_dist)
                        # compute duration
                        try:
                            start_ts = pd.to_datetime(trace_df['timestamp'].iloc[0])
                            end_ts = pd.to_datetime(trace_df['timestamp'].iloc[-1])
                            duration_min = max((end_ts - start_ts).total_seconds() / 60.0, 0.0)
                        except Exception:
                            duration_min = 0.0
                        # average speed
                        avg_speed = (distance_km / max(duration_min/60.0, 1e-6))
                        detected_mode = detect_mode_from_trace(avg_speed)
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")
                    distance_km = 0.0

            # Option: start/end coords
            elif distance_choice == "start/end coords":
                distance_km = compute_distance_km(start_lat, start_lon, end_lat, end_lon)
                # duration from times
                date_str = date_input.isoformat()
                start_iso = f"{date_str}T{start_time}"
                end_iso = f"{date_str}T{end_time}"
                duration_min = compute_duration_minutes(str(start_iso), str(end_iso))
                # if duration positive, infer avg speed
                if duration_min > 0 and distance_km > 0:
                    avg_speed = distance_km / (duration_min/60.0)
                    detected_mode = detect_mode_from_trace(avg_speed)

            # Option: manual
            else:
                distance_km = float(manual_km)
                # duration we approximate: assume avg speed by mode if duration not given
                # default durations: walking speed 5 km/h, bike 15km/h, car 40km/h etc.
                speed_lookup = {
                    "walk":5, "bike":15, "e-bike":18, "car_petrol":40, "car_diesel":40,
                    "bus":30, "train":60, "motorbike":45, "ev_small":40, "taxi":40
                }
                spd = speed_lookup.get(mode_choice, 30)
                duration_min = (distance_km / max(spd, 0.0001)) * 60.0

            # final chosen mode: if detected_mode exists, allow user choice override
            final_mode = detected_mode or mode_choice

            # compute emissions
            emission_kg = emission_for_mode(final_mode, distance_km)
            baseline_kg = baseline_emission(distance_km, baseline_mode=BASELINE_MODE)

            # save to DB
            # times stored as ISO strings
            date_iso = date_input.isoformat()
            start_iso = f"{date_iso}T{start_time}"
            end_iso = f"{date_iso}T{end_time}"
            add_trip(
                conn=conn,
                user_id=st.session_state.user['id'],
                date=date_iso,
                start_time=str(start_iso),
                end_time=str(end_iso),
                mode=final_mode,
                distance_km=distance_km,
                duration_min=duration_min,
                emission_kg=emission_kg,
                baseline_emission_kg=baseline_kg,
                start_lat=(start_lat if distance_choice=="start/end coords" else None),
                start_lon=(start_lon if distance_choice=="start/end coords" else None),
                end_lat=(end_lat if distance_choice=="start/end coords" else None),
                end_lon=(end_lon if distance_choice=="start/end coords" else None),
                notes=notes
            )
            st.success(f"Trip logged — {distance_km:.3f} km, mode: {final_mode}, emissions: {emission_kg:.3f} kg CO₂")
            if detected_mode:
                st.info(f"Mode auto-detected as: {detected_mode} based on average speed.")

# -------- Tab: Dashboard -----------
with tabs[1]:
    st.header("Dashboard")
    df = fetch_trips_df(conn)
    if df.empty:
        st.info("No data yet. Log a commute to see charts and analytics.")
    else:
        # filter by user or all
        user_filter = st.selectbox("Show data for", options=["All users", st.session_state.user['username']] if st.session_state.user else ["All users"])
        if user_filter != "All users":
            df = df[df['username'] == user_filter]

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_trips = len(df)
        total_distance = df['distance_km'].sum()
        total_emissions = df['emission_kg'].sum()
        baseline_emissions = df['baseline_emission_kg'].sum()
        avoided = max(baseline_emissions - total_emissions, 0.0)

        col1.metric("Trips", f"{total_trips}")
        col2.metric("Total distance (km)", f"{total_distance:.2f}")
        col3.metric("CO₂ emitted (kg)", f"{total_emissions:.2f}")
        col4.metric("Estimated CO₂ avoided (kg)", f"{avoided:.2f}")

        st.markdown("### Emissions by mode")
        by_mode = df.groupby("mode").agg({"distance_km":"sum","emission_kg":"sum","id":"count"}).rename(columns={"id":"trips"}).reset_index()
        fig1 = px.bar(by_mode, x="mode", y="emission_kg", hover_data=["distance_km","trips"], title="Emissions (kg) by Mode")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### Distance / Emissions over time")
        df_time = df.groupby(pd.Grouper(key='date', freq='D')).agg({'distance_km':'sum','emission_kg':'sum'}).reset_index()
        fig2 = px.line(df_time, x='date', y=['distance_km','emission_kg'], labels={'value':'Amount', 'variable':'Metric'}, title="Daily distance and emissions")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Recent trips")
        st.dataframe(df.sort_values("date", ascending=False).reset_index(drop=True))

        # Gamification / badges for current user (if logged in)
        if st.session_state.user:
            user_df = df[df['username'] == st.session_state.user['username']]
            if not user_df.empty:
                user_baseline = user_df['baseline_emission_kg'].sum()
                user_emitted = user_df['emission_kg'].sum()
                user_avoided = max(user_baseline - user_emitted, 0.0)
                st.markdown(f"**Your cumulative avoided CO₂:** {user_avoided:.2f} kg")
                badges = assign_badges(user_avoided)
                if badges:
                    st.markdown("**Badges earned:** " + ", ".join(badges))
                else:
                    st.markdown("Earn badges by avoiding CO₂. Try a bike or walk for short trips!")

# -------- Tab: Leaderboard -----------
with tabs[2]:
    st.header("Leaderboard")
    df = fetch_trips_df(conn)
    if df.empty:
        st.info("No trips logged yet.")
    else:
        leaderboard = df.groupby("username").agg({
            "distance_km":"sum",
            "emission_kg":"sum",
            "baseline_emission_kg":"sum",
            "id":"count"
        }).rename(columns={"id":"trips"})
        leaderboard['avoided_kg'] = leaderboard['baseline_emission_kg'] - leaderboard['emission_kg']
        leaderboard = leaderboard.reset_index().sort_values("avoided_kg", ascending=False)
        st.dataframe(leaderboard[['username','trips','distance_km','emission_kg','baseline_emission_kg','avoided_kg']])
        top = leaderboard.iloc[0]
        st.markdown(f"**Top:** {top['username']} — avoided {top['avoided_kg']:.2f} kg CO₂")

# -------- Tab: Data / Export -----------
with tabs[3]:
    st.header("Data & Export")
    df = fetch_trips_df(conn)
    if df.empty:
        st.info("No trips found.")
    else:
        st.download_button("Download all trips CSV", df.to_csv(index=False).encode('utf-8'), file_name="smrt_commute_trips.csv")
        st.markdown("### Raw data")
        st.dataframe(df)

# -------- Tab: Settings / Integrations -----------
with tabs[4]:
    st.header("Settings & Integrations")
    st.markdown("""
    **Notes & next steps for integrations**
    - Google Maps / Directions: add interactive maps and route distance verification via Google Maps Directions API (requires API key).
    - Geofencing: use Geoapify or Google Maps.
    - Auth: production-ready apps should use Firebase Auth / OAuth providers rather than local usernames.
    - Export: schedule weekly summary emails via an SMTP or transactional email provider.
    - Mobile PWA: wrap Streamlit using Streamlit-Share or host with container and create a simple manifest for PWA behavior.
    - OpenAI GPT: optional reminders, challenge generation. Provide `OPENAI_API_KEY` in env to enable.
    """)
    st.markdown("### Developer options")
    st.checkbox("Enable debug logs", key="dbg")
    # quick reset DB (dev only)
    if st.button("Reset DB (delete all data)"):
        conn.close()
        os.remove(DB_PATH)
        st.success("Database deleted — reload the page.")
