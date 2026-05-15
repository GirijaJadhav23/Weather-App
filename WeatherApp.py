# app.py
import streamlit as st
import os
import requests
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv

# Load .env from the same folder as app.py
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Weather App",
    page_icon="🌤️",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .main { background-color: #f0f4f8; }
        .weather-card {
            background: linear-gradient(135deg, #1e3c72, #2a69ac);
            border-radius: 20px;
            padding: 30px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        .city-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .temp-display {
            font-size: 4rem;
            font-weight: 800;
        }
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid #2a69ac;
            padding: 10px 20px;
            font-size: 1.1rem;
        }
        .stButton > button {
            border-radius: 25px;
            background: linear-gradient(135deg, #1e3c72, #2a69ac);
            color: white;
            border: none;
            padding: 10px 30px;
            font-size: 1rem;
            font-weight: 600;
            width: 100%;
        }
        .stButton > button:hover {
            opacity: 0.9;
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# --- LangChain Weather Tool ---
@tool
def weather_tool(city: str) -> str:
    """Get the current weather for a given city using OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        return (
            f"Weather in {city}: {description}, "
            f"Temp: {temp}°C, Feels like: {feels_like}°C, "
            f"Humidity: {humidity}%, Wind: {wind_speed} m/s"
        )
    elif response.status_code == 404:
        return f"City '{city}' not found."
    else:
        return f"Failed to fetch weather. Status code: {response.status_code}"

# --- Build Agent (cached to avoid re-init on every Streamlit rerun) ---
@st.cache_resource
def load_agent():
    model = ChatGroq(model="llama-3.3-70b-versatile")
    agent = create_agent(model, tools=[weather_tool], system_prompt="Use tools when needed")
    return agent

# --- Weather Icons ---
def get_weather_icon(condition: str) -> str:
    condition = condition.lower()
    if "clear" in condition or "sunny" in condition:
        return "☀️"
    elif "cloud" in condition:
        return "☁️"
    elif "rain" in condition or "drizzle" in condition:
        return "🌧️"
    elif "storm" in condition or "thunder" in condition:
        return "⛈️"
    elif "snow" in condition:
        return "❄️"
    elif "fog" in condition or "mist" in condition:
        return "🌫️"
    elif "wind" in condition:
        return "💨"
    else:
        return "🌤️"

# --- Call Agent and Parse Response ---
def get_weather(city: str) -> dict:
    agent = load_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"What is the current weather in {city}?"}]
    })

    raw_response = ""
    temp = humidity = feels_like = wind_speed = condition = "N/A"

    # Look for the tool output message which contains structured weather data
    for msg in result["messages"]:
        content = msg.content if hasattr(msg, "content") else ""
        if isinstance(content, str) and "Temp:" in content:
            raw_response = content
            try:
                parts = content.split(",")
                for part in parts:
                    part = part.strip()
                    if "Temp:" in part:
                        temp = part.split("Temp:")[1].strip().split()[0]
                    if "Feels like:" in part:
                        feels_like = part.split("Feels like:")[1].strip().split()[0]
                    if "Humidity:" in part:
                        humidity = part.split("Humidity:")[1].strip().split()[0]
                    if "Wind:" in part:
                        wind_speed = part.split("Wind:")[1].strip()
                if ":" in content:
                    after_city = content.split(":", 1)[1].strip()
                    condition = after_city.split(",")[0].strip()
            except Exception:
                pass
            break

    # Fallback: grab last non-empty message as raw response
    if not raw_response:
        for msg in reversed(result["messages"]):
            content = msg.content
            if content and content != []:
                raw_response = content
                break

    return {
        "city": city.title(),
        "temperature": f"{temp}°C" if temp != "N/A" and "°" not in str(temp) else temp,
        "humidity": f"{humidity}%" if humidity != "N/A" and "%" not in str(humidity) else humidity,
        "condition": condition,
        "wind_speed": wind_speed,
        "feels_like": f"{feels_like}°C" if feels_like != "N/A" and "°" not in str(feels_like) else feels_like,
        "raw_response": raw_response
    }

# --- Header ---
st.markdown("## 🌤️ Weather Assistant")
st.markdown("Powered by LangChain + Groq AI Agent")
st.divider()

# --- Input Section ---
col1, col2 = st.columns([3, 1])
with col1:
    city = st.text_input(
        "Enter City Name",
        placeholder="e.g. New York, London, Pune...",
        label_visibility="collapsed"
    )
with col2:
    search_btn = st.button("🔍 Search", use_container_width=True)

# --- Display Weather ---
if search_btn or (city and st.session_state.get("last_city") != city):
    if city.strip():
        st.session_state["last_city"] = city

        with st.spinner(f"Fetching weather for **{city}**..."):
            try:
                data = get_weather(city.strip())
                icon = get_weather_icon(data.get("condition", ""))

                # Main weather card
                st.markdown(f"""
                    <div class="weather-card">
                        <div class="city-title">{icon} {data['city']}</div>
                        <div style="font-size:1rem; opacity:0.8; margin-bottom:15px;">{data.get('condition', 'N/A')}</div>
                        <div class="temp-display">{data.get('temperature', 'N/A')}</div>
                        <div style="opacity:0.7; margin-top:5px;">Feels like {data.get('feels_like', 'N/A')}</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("💧 Humidity", data.get("humidity", "N/A"))
                with m2:
                    st.metric("🌡️ Temperature", data.get("temperature", "N/A"))
                with m3:
                    st.metric("💨 Wind Speed", data.get("wind_speed", "N/A"))

                # Full agent response
                with st.expander("🤖 Full Agent Response"):
                    st.write(data.get("raw_response", "No response available."))

            except Exception as e:
                st.error(f"❌ Could not fetch weather: {str(e)}")
    else:
        st.warning("⚠️ Please enter a city name.")

# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85rem;'>Built with LangChain + Streamlit</div>",
    unsafe_allow_html=True
)
