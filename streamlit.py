import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import google.generativeai as genai
import os
import base64


load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Load Gemini API key
if not GEMINI_API_KEY:
    st.error("""
        Gemini API key not found. Please create a .env file with:
        GEMINI_API_KEY=your_api_key_here
    """)
else:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_weather_report(city, date, temp_min, temp_max, rainfall):
    if GEMINI_API_KEY is None:
        return "Gemini API key not loaded. Cannot generate report."
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = f"""
        Generate a detailed weather report for {city} on {date}.
        Temperature range: {temp_min}°C to {temp_max}°C
        Rainfall: {rainfall} cm
        
        Include information about:
        1. General weather conditions
        2. Temperature analysis
        3. Rainfall expectations
        4. Wind conditions (make an educated guess)
        5. Recommendations for outdoor activities
        6. Any potential weather warnings or advisories
        
        Format the report in markdown.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Set page config
st.set_page_config(
    page_title="UrbanCity Weather Report",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Encode the image to base64
background_image_base64 = encode_image("back.jpg")  # Replace with your image path

# Create the background image CSS style with the base64-encoded string
background_image = f"url(data:image/jpeg;base64,{background_image_base64})"

st.markdown(f"""
    <style>
        .stApp {{
            background-image: {background_image};
            background-size: cover;  /* Ensure the image covers the full screen */
            background-position: center center;  /* Center the image */
            background-attachment: fixed;  /* Keep the background fixed during scroll */
        }}
    </style>
""", unsafe_allow_html=True)

# Load the trained model and historical data
@st.cache_resource
def load_model_and_data():
    with open("multivariate_model_seasons_final.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load historical data with proper date parsing
    try:
        historical_df = pd.read_csv("cleaned_data.csv", parse_dates=['Date'], dayfirst=True)
        historical_df['Date'] = pd.to_datetime(historical_df['Date'], errors='coerce')
        historical_df = historical_df.dropna(subset=['Date'])
        return model, historical_df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return model, None

model, historical_df = load_model_and_data()

# City season mappings
city_season_mapping = {
    "Mumbai": {
        "Summer": [2, 3, 4, 5],
        "Monsoon": [6, 7, 8],
        "Post-Monsoon": [9, 10],
        "Winter": [11, 12, 1]
    },
    "Delhi": {
        "Summer": [4, 5, 6],
        "Monsoon": [7, 8],
        "Post-Monsoon": [9, 10, 11],
        "Winter": [12, 1, 2, 3]
    },
    "Chennai": {
        "Summer": [3, 4, 5],
        "Monsoon": [6, 7, 8],
        "Post-Monsoon": [9, 10],
        "Winter": [11, 12, 1, 2]
    },
    "Kolkata": {
        "Summer": [5, 6, 7],
        "Monsoon": [6, 7, 8, 9],
        "Post-Monsoon": [10, 11],
        "Winter": [12, 1, 2, 3, 4]
    },
    "Bengaluru": {
        "Summer": [3, 4, 5, 6],
        "Monsoon": [7, 8, 9],
        "Post-Monsoon": [10],
        "Winter": [11, 12, 1, 2]
    },
    "Hyderabad": {
        "Summer": [3, 4, 5, 6],
        "Monsoon": [7, 8, 9],
        "Post-Monsoon": [10],
        "Winter": [11, 12, 1, 2]
    },
    "Ahmedabad": {
        "Summer": [3, 4, 5, 6],
        "Monsoon": [7, 8, 9],
        "Post-Monsoon": [9, 10],
        "Winter": [11, 12, 1, 2]
    },
    "Pune": {
        "Summer": [2, 3, 4, 5, 6],
        "Monsoon": [7, 8, 10],
        "Post-Monsoon": [10, 11],
        "Winter": [12, 1]
    }
}

season_name_mapping = {
    "Mumbai": {
        "Summer": 2,
        "Monsoon": 6,
        "Post-Monsoon": 9,
        "Winter": 11
    },
    "Delhi": {
        "Summer": 3,
        "Monsoon": 7,
        "Post-Monsoon": 9,
        "Winter": 1
    },
     "Chennai": {
        "Summer": 3,
        "Monsoon": 6,
        "Post-Monsoon": 9,
        "Winter": 11
    },
    "Kolkata": {
        "Summer": 5,
        "Monsoon": 6,
        "Post-Monsoon": 10,
        "Winter": 12
    },
    "Bengaluru": {
        "Summer": 3,
        "Monsoon": 7,
        "Post-Monsoon": 10,
        "Winter": 11
    },
    "Hyderabad": {
        "Summer": 3,
        "Monsoon": 7,
        "Post-Monsoon": 10,
        "Winter": 11
    },
    "Ahmedabad": {
        "Summer": 3,
        "Monsoon": 7,
        "Post-Monsoon": 10,
        "Winter": 11
    },
    "Pune": {
        "Summer": 2,
        "Monsoon": 7,
        "Post-Monsoon": 10,
        "Winter": 12
    }
}

# Label encoders
cities = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bengaluru", "Hyderabad", "Ahmedabad", "Pune"]
seasons = ["Summer", "Monsoon", "Post-Monsoon", "Winter"]

le = LabelEncoder()
le.fit(cities)
season_encoder = LabelEncoder()
season_encoder.fit(seasons)

# Helper functions
def get_city_season(month, city):
    if city not in city_season_mapping:
        raise ValueError(f"City '{city}' not found in season mapping.")
    
    for season, months in city_season_mapping[city].items():
        if month in months:
            return season
    return None

def get_month_from_season(season, city):
    if city not in season_name_mapping:
        raise ValueError(f"City '{city}' not found in season name mapping.")
    if season not in season_name_mapping[city]:
        raise ValueError(f"Season '{season}' not found for city '{city}'.")
    return season_name_mapping[city][season]

def predict_weather_by_date(date_str: str, city: str):
    dt = pd.to_datetime(date_str)
    
    year = dt.year
    month = dt.month
    day = dt.day
    weekday = dt.weekday()
    dayofyear = dt.dayofyear

    season = get_city_season(month, city)
    season_encoded = season_encoder.transform([season])[0]
    city_encoded = le.transform([city])[0]

    input_data = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Weekday': [weekday],
        'Season_Encoded': [season_encoded],
        'City_Encoded': [city_encoded],
        'Month_sin': [np.sin(2 * np.pi * month / 12)],
        'Month_cos': [np.cos(2 * np.pi * month / 12)],
        'DayOfYear_sin': [np.sin(2 * np.pi * dayofyear / 365)],
        'DayOfYear_cos': [np.cos(2 * np.pi * dayofyear / 365)],
        'Year_sin': [np.sin(2 * np.pi * year / 365)],
        'Year_cos': [np.cos(2 * np.pi * year / 365)]
    })

    features = ['Year', 'Month', 'Day', 'Weekday', 'Season_Encoded', 'City_Encoded',
               'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos','Year_sin', 'Year_cos']
    input_data = input_data[features]

    predictions = model.predict(input_data)[0]
    
    # Ensure rain is not negative
    rain = max(0, predictions[2])

    return {
        "Temp Min": round(predictions[0], 4),
        "Temp Max": round(predictions[1], 4),
        "Rain": round(rain, 4)
    }

def predict_weather_by_season(year: int, season: str, city: str):
    try:
        month = get_month_from_season(season, city)
    except ValueError as e:
        return {"Error": str(e)}

    season_encoded = season_encoder.transform([season])[0]

    input_data = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Day': [1],
        'Weekday': [0],
        'Season_Encoded': [season_encoded],
        'City_Encoded': [le.transform([city])[0]],
        'Month_sin': [0],
        'Month_cos': [0],
        'DayOfYear_sin': [0],
        'DayOfYear_cos': [0],
        'Year_sin': [0],
        'Year_cos': [0]
    })

    features = ['Year', 'Month', 'Day', 'Weekday', 'Season_Encoded', 'City_Encoded',
               'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos','Year_sin', 'Year_cos']
    input_data = input_data[features]

    predictions = model.predict(input_data)[0]
    
    # Ensure rain is not negative
    rain = max(0, predictions[2])

    return {
        "Temp Min": round(predictions[0], 4),
        "Temp Max": round(predictions[1], 4),
        "Rain": round(rain, 4)
    }

def plot_rainfall_distribution(city, predicted_month=None, predicted_rain=None):
    if historical_df is None:
        return None
    
    city_data = historical_df[historical_df['city'] == city].copy()
    city_data['Month'] = city_data['Date'].dt.month
    monthly_rain = city_data.groupby('Month')['Rain'].mean().reset_index()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_rain['Month'] = monthly_rain['Month'].apply(lambda x: month_names[x-1])
    
    # Convert rain from mm to cm for consistency
    monthly_rain['Rain_cm'] = monthly_rain['Rain'] / 10
    
    # Create bar chart
    fig = px.bar(monthly_rain, x='Month', y='Rain_cm',
                 title=f'Average Monthly Rainfall in {city} (Historical)',
                 labels={'Rain_cm': 'Rainfall (cm)', 'Month': 'Month'},
                 color='Rain_cm',
                 color_continuous_scale='Blues')
    
    # Highlight predicted month if available
    if predicted_month is not None and predicted_rain is not None:
        predicted_month_name = month_names[predicted_month-1]
        fig.add_trace(go.Scatter(
            x=[predicted_month_name],
            y=[predicted_rain],
            mode='markers',
            marker=dict(color='#F7374F', size=12),
            name='Predicted Rainfall',
            hovertemplate=f'<b>Predicted:</b> {predicted_rain:.1f}cm<extra></extra>'
        ))
    
    fig.update_layout(
        hovermode='x unified',
        coloraxis_showscale=False,
        yaxis_title='Rainfall (cm)',
        xaxis_title='Month'
    )
    
    return fig

def plot_temperature_trends(city, predicted_date=None, predicted_min=None, predicted_max=None):
    if historical_df is None:
        return None
    
    city_data = historical_df[historical_df['city'] == city].copy()
    city_data['Year'] = city_data['Date'].dt.year
    yearly_avg = city_data.groupby('Year').agg({'Temp Min':'mean', 'Temp Max':'mean'}).reset_index()
    
    fig = px.line(yearly_avg, x='Year', y=['Temp Min', 'Temp Max'],
                  title=f'Temperature Trends in {city} (Historical)',
                  labels={'value': 'Temperature (°C)', 'variable': 'Metric'})
    
    if predicted_date and predicted_min and predicted_max:
        # Add vertical line for prediction year
        fig.add_vline(x=predicted_date.year, line_dash="dash", line_color="#e6bb10", line_width=2)
        
        # Add prediction markers with bright colors
        fig.add_trace(go.Scatter(
            x=[predicted_date.year],
            y=[predicted_max],
            mode='markers+text',
            marker=dict(
                color='#e61010',  # Bright gold
                size=14,
                line=dict(color='black', width=2)
            ),
            text=[f"Max: {predicted_max:.1f}°C"],
            textposition="top center",
            textfont=dict(color='white', size=12),
            name='Prediction',
            hoverinfo='y+name'
        ))
        
        fig.add_trace(go.Scatter(
            x=[predicted_date.year],
            y=[predicted_min],
            mode='markers+text',
            marker=dict(
                color='#10cde6',  # Bright cyan
                size=14,
                line=dict(color='black', width=2)
            ),
            text=[f"Min: {predicted_min:.1f}°C"],
            textposition="bottom center",
            textfont=dict(color='white', size=12),
            name='Prediction',
            hoverinfo='y+name',
            showlegend=False
        ))
    
    # Improve overall layout
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_season_comparison_stacked(city, season, predicted_values):
    all_seasons = list(city_season_mapping[city].keys())
    historical_means = {s: {'Temp Max': 0, 'Temp Min': 0, 'Rain': 0} for s in all_seasons}
    
    for s in all_seasons:
        months = city_season_mapping[city][s]
        season_data = historical_df[(historical_df['city'] == city) & 
                                    (historical_df['Date'].dt.month.isin(months))]
        historical_means[s]['Temp Max'] = season_data['Temp Max'].mean()
        historical_means[s]['Temp Min'] = season_data['Temp Min'].mean()
        historical_means[s]['Rain'] = season_data['Rain'].mean()

    fig = go.Figure()

    # Stacked bars for historical data
    fig.add_trace(go.Bar(
        x=all_seasons,
        y=[historical_means[s]['Temp Min'] for s in all_seasons],
        name='Historical Min Temp',
        marker_color='#DDA853'
    ))

    fig.add_trace(go.Bar(
        x=all_seasons,
        y=[historical_means[s]['Temp Max'] - historical_means[s]['Temp Min'] for s in all_seasons],
        name='Historical Max Temp',
        marker_color='#183B4E'
    ))

    fig.add_trace(go.Bar(
        x=all_seasons,
        y=[historical_means[s]['Rain'] for s in all_seasons],
        name='Historical Rainfall',
        marker_color='#27548A'
    ))

    # Prediction points
    fig.add_trace(go.Scatter(
        x=[season],
        y=[predicted_values['Temp Min']],
        mode='markers',
        marker=dict(size=15, color='#48A6A7', symbol='diamond'),
        name='Predicted Min Temp'
    ))

    fig.add_trace(go.Scatter(
        x=[season],
        y=[predicted_values['Temp Max']],
        mode='markers',
        marker=dict(size=15, color='#818C78', symbol='diamond'),
        name='Predicted Max Temp'
    ))

    fig.add_trace(go.Scatter(
        x=[season],
        y=[predicted_values['Rain']],
        mode='markers',
        marker=dict(size=15, color='#F5EEDC', symbol='diamond'),
        name='Predicted Rainfall'
    ))

    fig.update_layout(
        title=f'{city} Seasonal Comparison',
        yaxis_title='Temperature (°C) / Rainfall (cm)',
        barmode='stack',
        legend=dict(x=1, y=1)
    )

    return fig

# UI Components
def main():
    encode_image("back.jpg")
    # Custom CSS
    st.markdown("""
        <style>
            .big-font {
                font-size:24px !important;
                font-weight: bold;
            }
            .weather-card {
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                margin: 10px 0;
            }
            .temp-min {
                color: #1E90FF;
            }
            .temp-max {
                color: #FF6347;
            }
            .rain {
                color: #4682B4;
            }
            .stSelectbox, .stDateInput, .stNumberInput {
                margin-bottom: 20px;
            }
            .result-value {
                font-size: 28px;
                font-weight: bold;
                margin-top: 10px;
            }
            .comparison-header {
                color: #2E86C1;
                margin-top: 30px;
                margin-bottom: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3222/3222807.png", width=100)
    with col2:
        st.title("UrbanCity Weather Report")
        st.markdown("Predict temperature and rainfall for major Indian cities")

    # Navigation
    tab1, tab2 = st.tabs(["📅 Date Prediction", "🌱 Season Prediction"])

    with tab1:
        st.header("Predict Weather by Specific Date")
        st.markdown("Select a date and city to get weather predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input(
                "Select Date",
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 12, 31),
                value=datetime.now()
            )
        with col2:
            selected_city = st.selectbox(
                "Select City",
                cities,
                index=0
            )
        
        if st.button("Predict Weather", key="predict_date"):
            with st.spinner("Predicting weather..."):
                try:
                    result = predict_weather_by_date(str(selected_date), selected_city)
                    if result["Rain"] > 30:
                        result["Rain"] = 1+(result['Rain']/100)
                    else:
                        result["Rain"] = (result['Rain']/100)
                    
                    st.markdown("### Prediction Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="weather-card">
                                <div class="big-font temp-min">Min Temperature</div>
                                <div class="result-value">{result['Temp Min']:.4f}°C</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="weather-card">
                                <div class="big-font temp-max">Max Temperature</div>
                                <div class="result-value">{result['Temp Max']:.4f}°C</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                            <div class="weather-card">
                                <div class="big-font rain">Rainfall</div>
                                <div class="result-value">{result['Rain']:.4f} cm</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Historical visualizations
                    if historical_df is not None:
                        st.markdown("### Temperature Trends")
                        temp_fig = plot_temperature_trends(
                            selected_city,
                            selected_date,
                            result['Temp Min'],
                            result['Temp Max']
                        )
                        st.plotly_chart(temp_fig, use_container_width=True)

                        st.markdown("### Rainfall Analysis")
                        # In your prediction section after getting results:
                        rain_fig = plot_rainfall_distribution(
                            selected_city,
                            selected_date.month,  # or month from season prediction
                            result['Rain']       # your predicted rain value
                        )
                        st.plotly_chart(rain_fig, use_container_width=True)
                    
                    # Generate and display detailed weather report
                    st.markdown("### Detailed Weather Report")
                    with st.spinner("Generating detailed report..."):
                        report = generate_weather_report(
                            selected_city,
                            str(selected_date),
                            result['Temp Min'],
                            result['Temp Max'],
                            result['Rain']
                        )
                        st.markdown(report)
                    
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

    with tab2:
        st.header("Predict Weather by Season")
        st.markdown("Select a year, season and city to get seasonal weather predictions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_year = st.number_input(
                "Year",
                min_value=2020,
                max_value=2030,
                value=datetime.now().year,
                step=1
            )
        with col2:
            selected_season = st.selectbox(
                "Select Season",
                seasons,
                index=0
            )
        with col3:
            selected_city_season = st.selectbox(
                "Select City",
                cities,
                index=0,
                key="city_season"
            )
        
        if st.button("Predict Weather", key="predict_season"):
            with st.spinner("Predicting seasonal weather..."):
                try:
                    result = predict_weather_by_season(selected_year, selected_season, selected_city_season)
                    if result["Rain"] > 4:
                        result["Rain"] = 1.2+(result['Rain']/100)
                    else:
                        result["Rain"] = (result['Rain']/100)
                    
                    st.markdown("### Prediction Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="weather-card">
                                <div class="big-font temp-min">Min Temperature</div>
                                <div class="result-value">{result['Temp Min']:.4f}°C</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="weather-card">
                                <div class="big-font temp-max">Max Temperature</div>
                                <div class="result-value">{result['Temp Max']:.4f}°C</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                            <div class="weather-card">
                                <div class="big-font rain">Rainfall</div>
                                <div class="result-value">{result['Rain']:.4f} cm/Day</div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Seasonal visualization
                    if historical_df is not None:
                        st.markdown("### Seasonal Temperature Distribution")
                        result = predict_weather_by_season(selected_year, selected_season, selected_city)
                        season_fig = plot_season_comparison_stacked(
                            selected_city_season,
                            selected_season,
                            result
                        )
                        st.plotly_chart(season_fig, use_container_width=True)
                    
                    #Generate and display detailed weather report
                    st.markdown("### Detailed Weather Report")
                    with st.spinner("Generating detailed report..."):
                        report = generate_weather_report(
                            selected_city_season,
                            f"{selected_season} {selected_year}",
                            result['Temp Min'],
                            result['Temp Max'],
                            result['Rain']
                        )
                        st.markdown(report)

                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

    # About section
    st.markdown("---")
    st.markdown("""
        ### About This App
        This weather prediction app uses machine learning to forecast:
        - Minimum and maximum temperatures (°C)
        - Rainfall (cm)
        
        The model was trained on historical weather data from major Indian cities using XGBoost.
        
        Features include:
        - Date-specific weather predictions
        - Seasonal weather patterns
        - Historical data comparison
        - AI-generated detailed weather reports
    """)

if __name__ == "__main__":
    main()