# Smart-Green-Commute-Tracker
Smart Green Commute Tracker is a cross-platform, AI-assisted Progressive Web App (PWA) implemented as a Streamlit prototype to help students and communities track commute behavior, estimate CO₂ emissions, and encourage sustainable travel through gamification and community challenges. This repository contains the Streamlit implementation, data model, emission engine, dashboard, and export functionality.  
(Original project resources: Smart Green Commute Tracker Resources). 

## Project Owner
**Boyina Sankar** — Start date: 19 July 2025

## Vision & SMART Goal
Enable 150+ students across departments to adopt eco-commute practices and collectively reduce over **50 kg CO₂** within 6 weeks by providing AI-powered suggestions, gamified incentives, and an easy-to-use mobile-friendly platform.

## Key Features
- Smart commute logging (manual, start/end coords, or GPS trace CSV)
- Auto-detection heuristics for transport mode (based on speed)
- Emission estimation engine (kg CO₂ per km by mode) and baseline comparison
- Personal and group dashboards (daily/weekly charts, mode breakdowns)
- Leaderboard and badges (gamification for CO₂ avoided)
- CSV export of trip logs and admin analytics export
- Offline-first PWA-friendly design considerations and accessibility (WCAG 2.1 AA)
- Integration-ready: Google Maps Directions, Geoapify, OpenAI reminders, Firebase Auth/Firestore

## Technology Stack
- Frontend / UI: Streamlit (prototype), React + Tailwind (production suggestion)
- Backend / Data: SQLite for prototype; Firestore / PostgreSQL recommended for scale
- Maps & Geo: Google Maps API, Geoapify
- Visualization: Plotly / Recharts / Chart.js
- AI: OpenAI GPT API (for optional reminders & challenge generation)
- Deployment: Vercel / Netlify / Container (Docker) + CI/CD

## How emissions are computed (brief)
1. Distance (km) derived from:
   - Haversine between start/end coords, or
   - Summed GPS trace segments from uploaded CSV, or
   - Manual distance input
2. Emissions = `distance_km * emission_factor(mode)` (kg CO₂/km)
3. Baseline emission uses a default mode (car_petrol) to compute avoided CO₂
4. Avoided CO₂ = `baseline_emission - actual_emission` (bounded ≥ 0)

## Getting started (run locally)
1. Clone the repo:
```bash
git clone https://github.com/sankar069/Smart-Green-Commute-Tracker.git
cd Smart-Green-Commute-Tracker
(Optional) Create and activate a virtualenv:

bash
Copy
Edit
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run streamlit_app.py
Open the URL shown in the terminal (usually http://localhost:8501).

Project structure (suggested)
streamlit_app.py — single-file Streamlit prototype (logging, dashboard, leaderboard)

requirements.txt — Python dependencies

.streamlit/ — local Streamlit configs (ignore in repo)

docs/ — design mocks, API notes, deployment guides

data/ — example CSV traces and sample datasets

Production roadmap / next steps
Replace SQLite with Firebase Auth + Firestore for secure auth & scale

Use Google Maps Directions API to compute route distance (road-aware)

Implement background mobile tracking (mobile client / PWA + service worker)

Train a small ML classifier for accurate mode detection using speed/accel features

Add weekly personalized suggestions via OpenAI (opt-in)

Add optional NFT/badge issuance or integration with Badgr/Credly

Contribution
Contributions are welcome — open an issue or create a pull request. For major changes, please open an issue first to discuss the design.

License
Add a license (e.g., MIT) in LICENSE if you want to permit reuse.

Contact
Project owner: Boyina Sankar — GitHub: sankar069
