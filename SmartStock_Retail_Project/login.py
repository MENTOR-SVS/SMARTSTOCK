# login.py — Fixed for integration with SmartStock app.py
import streamlit as st
import hashlib
import json
import time
from pathlib import Path

# --- Utility: Hashing passwords ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# --- File to store password permanently ---
CRED_FILE = Path("owner_credentials.json")

# --- Default credentials ---
DEFAULT_CREDS = {
    "username": "owner",
    "password_hash": make_hashes("admin123"),
    "secret_code": "smartstock2025"
}

# --- Create credentials file if not exists ---
if not CRED_FILE.exists():
    with open(CRED_FILE, "w") as f:
        json.dump(DEFAULT_CREDS, f)

# --- Load credentials ---
with open(CRED_FILE, "r") as f:
    creds = json.load(f)

# --- CSS Styling ---
def add_login_style():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #b3d9ff 0%, #e6f0ff 100%);
        }
        .login-box {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(12px);
            padding: 2.8rem 2.5rem;
            border-radius: 22px;
            box-shadow: 0 6px 25px rgba(0,0,0,0.15);
            text-align: center;
            width: 380px;
            margin: 10vh auto;
            animation: fadeIn 0.6s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .login-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #003d99;
            margin-bottom: 0.5rem;
        }
        .project-title {
            font-size: 1.1rem;
            color: #0059b3;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #aac6ff;
            padding: 8px;
        }
        .stButton>button {
            background-color: #004aad;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #003080;
            transform: translateY(-2px);
        }
        header, footer, [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stHeader"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

# --- Login Page ---
def login_page():
    add_login_style()
    st.markdown("<div>", unsafe_allow_html=True)

    st.markdown("""
        <div class='login-title'>SmartStock Login</div>
        <div class='project-title'>Retail Inventory Optimization Dashboard</div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == creds["username"] and check_hashes(password, creds["password_hash"]):
            st.success("✅ Login successful! Redirecting...")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Incorrect username or password.")

    if st.button("Forgot Password"):
        st.session_state["forgot_mode"] = True
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --- Forgot Password Page ---
def forgot_password_page():
    add_login_style()
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("""
        <div class='login-title'>🔁 Reset Password</div>
        <div class='project-title'>SmartStock Admin Security</div>
    """, unsafe_allow_html=True)

    secret = st.text_input("Enter Secret Code")
    new_pass = st.text_input("Enter New Password", type="password")
    confirm_pass = st.text_input("Confirm New Password", type="password")

    if st.button("Reset Password"):
        if secret == creds["secret_code"]:
            if new_pass == confirm_pass and new_pass != "":
                creds["password_hash"] = make_hashes(new_pass)
                with open(CRED_FILE, "w") as f:
                    json.dump(creds, f)
                st.success("✅ Password successfully reset! Please log in.")
                time.sleep(1.2)
                st.session_state["forgot_mode"] = False
                st.rerun()
            else:
                st.error("⚠️ Passwords do not match or are empty.")
        else:
            st.error("❌ Invalid Secret Code.")

    if st.button("🔙 Back to Login"):
        st.session_state["forgot_mode"] = False
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --- Logout Function ---
def logout(place="sidebar"):
    if place == "sidebar":
        st.sidebar.write("---")
        if st.sidebar.button("🚪 Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.success("✅ Logged out successfully!")
            time.sleep(1)
            st.rerun()
    else:
        if st.button("🚪 Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.success("✅ Logged out successfully!")
            time.sleep(1)
            st.rerun()
