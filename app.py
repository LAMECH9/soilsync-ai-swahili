# app.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import logging

# === CONFIGURE ===
API_TOKEN = "KLRDg3ElBVveVghcN61aScAJevKMgofJF7CWcsVwG2mYt0mUQF63DdB0n6OHqOo9WYCilH7bjJ6s9sIc4zT9zzeCyPXhvytRL4wMAtbV5fRxnAmLFtEI9KXO5tvnu0Pm3rwhAfx5tXGiQOKEm98U2lGTZOIVav2hRtGwsU8SrzUPpZA6CNSNCGkCNp3sndYsrAqeme9xsqFGNEla2PBgjZ0ertc6j8nzCVzUQ8gX2T9hFnR8SoKRA7eyRMHRMDrn"
SOIL_API_URL = "https://farmerdb.kalro.org/api/SoilData/legacy/county"
AGRODEALER_API_URL = "https://farmerdb.kalro.org/api/SoilData/agrodealers"
WEATHER_API_KEY = "your_openweather_api_key"  # Replace with your OpenWeatherMap API key
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === TRANSLATIONS DICTIONARY ===
translations = {
    "en": {
        "title": "SoilSync AI: Precision Fertilizer Recommendations for Maize",
        "select_user_type": "Select User Type",
        "farmer": "Farmer",
        "researcher": "Researcher/Extension Officer",
        "farmer_header": "Farmer-Friendly Recommendations",
        "farmer_instruction": "Select your ward and describe your crop's condition to get tailored fertilizer recommendations for maize farming in Trans Nzoia.",
        "select_ward": "Select Your Ward",
        "select_language": "Select Language",
        "crop_state_header": "Describe Your Crop's Condition",
        "crop_symptoms": ["Yellowing leaves", "Stunted growth", "Poor flowering", "Wilting", "Leaf spots"],
        "recommendations_header": "Recommendations for {}",
        "no_data": "No soil data available for recommendations.",
        "optimal_soil": "Soil parameters are within optimal ranges for maize.",
        "dealers_header": "Where to Buy Fertilizers",
        "dealers_none": "No agro-dealers found for this ward. Check county-level suppliers in Kitale or Kwanza markets.",
        "dealer_info": "- **{}** ({}) - Phone: {}",
        "error_data": "Unable to load soil data. Please try again later.",
        "language_confirmation": "Language set to English.",
        "weather_info": "Current weather in Trans Nzoia: {}°C, {} mm precipitation. {}",
        "footer": "SoilSync AI by Kibabii University | Powered by KALRO Data | Contact: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Apply **agricultural lime** (1–2 tons/ha) to correct acidic soil (pH {:.2f}).",
        "rec_ph_alkaline": "Use **Ammonium Sulphate** (100–200 kg/ha) to lower alkaline soil (pH {:.2f}).",
        "rec_nitrogen": "Apply **DAP (100–150 kg/ha)** at planting and **CAN (100–200 kg/ha)** or **Urea (50–100 kg/ha)** for top-dressing to address nitrogen deficiency.",
        "rec_phosphorus": "Apply **DAP (100–150 kg/ha)** or **TSP (100–150 kg/ha)** at planting for phosphorus deficiency.",
        "rec_potassium": "Use **NPK 17:17:17 or 23:23:0** (100–150 kg/ha) at planting for potassium deficiency.",
        "rec_zinc": "Apply **Mavuno Maize Fertilizer** or **YaraMila Cereals** for zinc deficiency, or use zinc sulfate foliar spray (5–10 kg/ha).",
        "rec_boron": "Apply **borax** (1–2 kg/ha) for boron deficiency.",
        "rec_organic": "Apply **compost/manure (5–10 tons/ha)** or **Mazao Organic** to boost organic matter.",
        "rec_salinity": "Implement leaching with irrigation and use **Ammonium Sulphate** to manage high salinity.",
        "weather_warning": "Avoid top-dressing fertilizers due to high precipitation ({} mm). Wait for drier conditions.",
        "model_error": "Model training failed. Using threshold-based recommendations."
    },
    "sw": {
        "title": "SoilSync AI: Mapendekezo ya Mbolea ya Usahihi kwa Mahindi",
        "select_user_type": "Chagua Aina ya Mtumiaji",
        "farmer": "Mkulima",
        "researcher": "Mtafiti/Afisa Ugani",
        "farmer_header": "Mapendekezo Yanayofaa Mkulima",
        "farmer_instruction": "Chagua wadi yako na elezea hali ya mazao yako ili kupata mapendekezo ya mbolea yanayofaa kwa kilimo cha mahindi huko Trans Nzoia.",
        "select_ward": "Chagua Wadi Yako",
        "select_language": "Chagua Lugha",
        "crop_state_header": "Elezea Hali ya Mazao Yako",
        "crop_symptoms": ["Majani yanageuka manjano", "Ukuaji umedumaa", "Maua duni", "Kunyauka", "Madoa kwenye majani"],
        "recommendations_header": "Mapendekezo kwa {}",
        "no_data": "Hakuna data ya udongo inayopatikana kwa mapendekezo.",
        "optimal_soil": "Vigezo vya udongo viko ndani ya viwango bora kwa mahindi.",
        "dealers_header": "Wapi pa Kununua Mbolea",
        "dealers_none": "Hakuna wauzaji wa mbolea waliopatikana kwa wadi hii. Angalia wauzaji wa ngazi ya kaunti katika soko za Kitale au Kwanza.",
        "dealer_info": "- **{}** ({}) - Simu: {}",
        "error_data": "Imeshindwa kupakia data ya udongo. Tafadhali jaribu tena baadaye.",
        "language_confirmation": "Lugha imewekwa kwa Kiswahili.",
        "weather_info": "Hali ya hewa ya sasa huko Trans Nzoia: {}°C, mvua {} mm. {}",
        "footer": "SoilSync AI na Chuo Kikuu cha Kibabii | Inatumia Data ya KALRO | Wasiliana: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Tumia **chokaa cha kilimo** (tani 1–2 kwa hekta) kurekebisha udongo wenye tindikali (pH {:.2f}).",
        "rec_ph_alkaline": "Tumia **Ammonium Sulphate** (kg 100–200 kwa hekta) kupunguza udongo wa alkali (pH {:.2f}).",
        "rec_nitrogen": "Tumia **DAP (kg 100–150 kwa hekta)** wakati wa kupanda na **CAN (kg 100–200 kwa hekta)** au **Urea (kg 50–100 kwa hekta)** kwa kurutubisha juu ili kushughulikia upungufu wa nitrojeni.",
        "rec_phosphorus": "Tumia **DAP (kg 100–150 kwa hekta)** au **TSP (kg 100–150 kg kwa hekta)** wakati wa kupanda kwa upungufu wa fosforasi.",
        "rec_potassium": "Tumia **NPK 17:17:17 au 23:23:0** (kg 100–150 kwa hekta) wakati wa kupanda kwa upungufu wa potasiamu.",
        "rec_zinc": "Tumia **Mbolea ya Mavuno Maize** au **YaraMila Cereals** kwa upungufu wa zinki, au tumia dawa ya zinki ya sulfate (kg 5–10 kwa hekta).",
        "rec_boron": "Tumia **borax** (kg 1–2 kwa hekta) kwa upungufu wa boron.",
        "rec_organic": "Tumia **mbolea ya kikaboni/samadi (tani 5–10 kwa hekta)** au **Mazao Organic** kuongeza vitu vya kikaboni.",
        "rec_salinity": "Tekeleza uchukuzi wa maji na umwagiliaji na tumia **Ammonium Sulphate** kushughulikia chumvi nyingi.",
        "weather_warning": "Epuka kurutubisha juu kwa sababu ya mvua nyingi ({} mm). Subiri hali ya hewa kavu.",
        "model_error": "Ufundishaji wa modeli umeshindwa. Tumia mapendekezo ya msingi wa kizingiti."
    }
}

# === FETCH WEATHER DATA FUNCTION ===
def fetch_weather_data(lat=1.0167, lon=35.0023):  # Kitale coordinates
    url = f"{WEATHER_API_URL}?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        temp = data["main"]["temp"]
        precipitation = data.get("rain", {}).get("1h", 0)  # Rainfall in last hour (mm)
        description = data["weather"][0]["description"]
        return {"temp": temp, "precipitation": precipitation, "description": description}
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return None

# === TRAIN RANDOM FOREST MODEL ===
def train_soil_model(soil_data):
    if soil_data is None or soil_data.empty:
        logger.warning("Soil data is None or empty")
        return None, None, []
    features = ["soil_pH", "total_Nitrogen_percent_", "phosphorus_Olsen_ppm", "potassium_meq_percent_", "zinc_ppm", "boron_ppm"]
    features = [f for f in features if f in soil_data.columns]
    if not features:
        logger.warning("No valid features in soil data")
        return None, None, []
    X = soil_data[features].dropna()
    if X.empty:
        logger.warning("No data after dropping NaN values")
        return None, None, features
    # Create synthetic fertility labels
    y = []
    for _, row in X.iterrows():
        score = (
            (row.get("soil_pH", 7.0) >= 5.5 and row.get("soil_pH", 7.0) <= 7.0) * 1 +
            (row.get("total_Nitrogen_percent_", 0.3) >= 0.2) * 1 +
            (row.get("phosphorus_Olsen_ppm", 20) >= 15) * 1 +
            (row.get("potassium_meq_percent_", 0.3) >= 0.2) * 1
        )
        if score >= 3:
            y.append("high")
        elif score >= 1:
            y.append("medium")
        else:
            y.append("low")
    if len(set(y)) < 2:
        logger.warning("Single-class labels detected")
        return None, None, features
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        logger.info("Model trained successfully")
        return model, scaler, features
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None, None, features

# === PREDICT SOIL FERTILITY ===
def predict_soil_fertility(model, scaler, features, input_data):
    if model is None or scaler is None or not features:
        return None, None
    try:
        input_df = pd.DataFrame([input_data], columns=features)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        importance = model.feature_importances_
        explanation = {f: i for f, i in zip(features, importance)}
        return prediction, explanation
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None

# === FETCH SOIL DATA FUNCTION ===
def fetch_soil_data(county_name, crop="maize"):
    url = f"{SOIL_API_URL}/{county_name}"
    headers = {"Authorization": f"Token {API_TOKEN}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        relevant_columns = [
            "county", "constituency", "ward", "latitude", "longitude", "soil_pH",
            "total_Nitrogen_percent_", "total_Org_Carbon_percent_", "phosphorus_Olsen_ppm",
            "potassium_meq_percent_", "calcium_meq_percent_", "magnesium_meq_percent_",
            "zinc_ppm", "boron_ppm", "electr_Conductivity_mS_per_cm", "crop"
        ]
        available_columns = [col for col in relevant_columns if col in df.columns]
        df_filtered = df[available_columns].copy()  # Create a copy to avoid SettingWithCopyWarning
        if "crop" in df_filtered.columns:
            df_filtered.loc[:, "crop"] = df_filtered["crop"].astype(str)  # Use .loc for assignment
            maize_mask = df_filtered["crop"].str.lower().str.contains(crop.lower(), na=False)
            df_filtered = df_filtered[maize_mask]
        core_params = [
            "soil_pH", "total_Nitrogen_percent_", "total_Org_Carbon_percent_",
            "phosphorus_Olsen_ppm", "potassium_meq_percent_"
        ]
        core_params = [col for col in core_params if col in df_filtered.columns]
        df_filtered = df_filtered.dropna(subset=core_params, how='any')
        numeric_cols = [col for col in core_params + ["zinc_ppm", "boron_ppm", "electr_Conductivity_mS_per_cm"] if col in df_filtered.columns]
        for col in numeric_cols:
            df_filtered.loc[:, col] = pd.to_numeric(df_filtered[col], errors='coerce')
        df_filtered = df_filtered.rename(columns={
            "county": "County", "constituency": "Constituency", "ward": "Ward",
            "latitude": "Latitude", "longitude": "Longitude"
        })
        return df_filtered
    except Exception as e:
        logger.error(f"Soil data fetch error: {e}")
        st.error(translations["en"]["error_data"])
        return None

# === FETCH AGRO-DEALER DATA FUNCTION ===
def fetch_agrodealer_data(county_name, constituencies, wards):
    headers = {"Authorization": f"Token {API_TOKEN}", "Content-Type": "application/json"}
    all_dealers = []
    for constituency, ward in zip(constituencies, wards):
        url = f"{AGRODEALER_API_URL}/{county_name}/{constituency}/{ward}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "dealers" in data:
                all_dealers.extend(data["dealers"])
        except:
            continue
    if not all_dealers:
        url = f"{AGRODEALER_API_URL}/{county_name}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "dealers" in data:
                all_dealers.extend(data["dealers"])
        except:
            st.error(translations["en"]["error_data"])
            return None
    if all_dealers:
        df_dealers = pd.DataFrame(all_dealers)
        dealer_columns = ["county", "subcounty", "ward", "agrodealerName", "market", "gpsLatitude", "gpsLongitude", "agrodealerPhone"]
        df_dealers = df_dealers[[col for col in dealer_columns if col in df_dealers.columns]]
        df_dealers = df_dealers.rename(columns={
            "county": "County", "subcounty": "Constituency", "ward": "Ward",
            "gpsLatitude": "Latitude", "gpsLongitude": "Longitude"
        })
        df_dealers['Latitude'] = pd.to_numeric(df_dealers['Latitude'], errors='coerce')
        df_dealers['Longitude'] = pd.to_numeric(df_dealers['Longitude'], errors='coerce')
        return df_dealers
    return None

# === MERGE DATA FUNCTION ===
def merge_soil_agrodealer_data(soil_df, dealer_df):
    if soil_df is None:
        return None
    if dealer_df is None:
        return soil_df
    try:
        merged_df = pd.merge(
            soil_df, dealer_df, on=["County", "Constituency", "Ward"],
            how="left", suffixes=("_soil", "_dealer")
        )
        # Handle missing values explicitly to avoid FutureWarning
        merged_df['Latitude'] = merged_df['Latitude_soil'].fillna(merged_df['Latitude_dealer'])
        merged_df['Longitude'] = merged_df['Longitude_soil'].fillna(merged_df['Longitude_dealer'])
        merged_df = merged_df.drop(columns=['Latitude_soil', 'Longitude_soil', 'Latitude_dealer', 'Longitude_dealer'], errors='ignore')
        return merged_df
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return soil_df

# === FERTILIZER RECOMMENDATION FUNCTION FOR FARMERS ===
def get_fertilizer_recommendations_farmer(soil_data, ward, crop_symptoms, weather_data, lang="en"):
    recommendations = []
    if soil_data is None or soil_data.empty:
        return [translations[lang]["no_data"]]
    ward_data = soil_data[soil_data['Ward'] == ward]
    if ward_data.empty:
        ward_data = soil_data  # Fallback to county-level averages
    avg_data = ward_data.mean(numeric_only=True)
    
    # Map crop symptoms to deficiencies
    symptom_deficiencies = {
        "Yellowing leaves": ["nitrogen", "zinc"],
        "Stunted growth": ["nitrogen", "phosphorus", "potassium"],
        "Poor flowering": ["phosphorus", "potassium"],
        "Wilting": ["potassium", "organic"],
        "Leaf spots": ["zinc", "boron"],
        "Majani yanageuka manjano": ["nitrogen", "zinc"],
        "Ukuaji umedumaa": ["nitrogen", "phosphorus", "potassium"],
        "Maua duni": ["phosphorus", "potassium"],
        "Kunyauka": ["potassium", "organic"],
        "Madoa kwenye majani": ["zinc", "boron"]
    }
    
    # Weather considerations
    if weather_data and weather_data["precipitation"] > 10:
        recommendations.append(translations[lang]["weather_warning"].format(weather_data["precipitation"]))
    
    # Soil-based recommendations
    if 'soil_pH' in avg_data and avg_data['soil_pH'] < 5.5:
        recommendations.append(translations[lang]["rec_ph_acidic"].format(avg_data['soil_pH']))
    elif 'soil_pH' in avg_data and avg_data['soil_pH'] > 7.0:
        recommendations.append(translations[lang]["rec_ph_alkaline"].format(avg_data['soil_pH']))
    
    # Symptom and soil-based recommendations
    deficiencies = set()
    for symptom in crop_symptoms:
        deficiencies.update(symptom_deficiencies.get(symptom, []))
    
    if "nitrogen" in deficiencies or ('total_Nitrogen_percent_' in avg_data and avg_data['total_Nitrogen_percent_'] < 0.2):
        recommendations.append(translations[lang]["rec_nitrogen"])
    if "phosphorus" in deficiencies or ('phosphorus_Olsen_ppm' in avg_data and avg_data['phosphorus_Olsen_ppm'] < 15):
        recommendations.append(translations[lang]["rec_phosphorus"])
    if "potassium" in deficiencies or ('potassium_meq_percent_' in avg_data and avg_data['potassium_meq_percent_'] < 0.2):
        recommendations.append(translations[lang]["rec_potassium"])
    if "zinc" in deficiencies or ('zinc_ppm' in avg_data and avg_data['zinc_ppm'] < 1):
        recommendations.append(translations[lang]["rec_zinc"])
    if "boron" in deficiencies or ('boron_ppm' in avg_data and avg_data['boron_ppm'] < 0.5):
        recommendations.append(translations[lang]["rec_boron"])
    if "organic" in deficiencies or ('total_Org_Carbon_percent_' in avg_data and avg_data['total_Org_Carbon_percent_'] < 1):
        recommendations.append(translations[lang]["rec_organic"])
    if 'electr_Conductivity_mS_per_cm' in avg_data and avg_data['electr_Conductivity_mS_per_cm'] > 1:
        recommendations.append(translations[lang]["rec_salinity"])
    
    return recommendations if recommendations else [translations[lang]["optimal_soil"]]

# === FERTILIZER RECOMMENDATION FUNCTION FOR RESEARCHERS ===
def get_fertilizer_recommendations_researcher(input_data, model, scaler, features, lang="en"):
    recommendations = []
    prediction, explanation = predict_soil_fertility(model, scaler, features, input_data)
    
    if prediction is None:
        recommendations.append(translations[lang]["model_error"])
    
    # Recommendations based on input
    if input_data.get("soil_pH", 7.0) < 5.5:
        recommendations.append(translations[lang]["rec_ph_acidic"].format(input_data["soil_pH"]))
    elif input_data.get("soil_pH", 7.0) > 7.0:
        recommendations.append(translations[lang]["rec_ph_alkaline"].format(input_data["soil_pH"]))
    if input_data.get("total_Nitrogen_percent_", 0.3) < 0.2:
        recommendations.append(translations[lang]["rec_nitrogen"])
    if input_data.get("phosphorus_Olsen_ppm", 20) < 15:
        recommendations.append(translations[lang]["rec_phosphorus"])
    if input_data.get("potassium_meq_percent_", 0.3) < 0.2:
        recommendations.append(translations[lang]["rec_potassium"])
    if input_data.get("zinc_ppm", 2) < 1:
        recommendations.append(translations[lang]["rec_zinc"])
    if input_data.get("boron_ppm", 1) < 0.5:
        recommendations.append(translations[lang]["rec_boron"])
    
    # Explanation for farmer communication
    advice = "No model prediction available." if prediction is None else f"Soil fertility predicted as {prediction}. "
    if prediction == "low":
        advice += "Explain to the farmer that low soil fertility is due to deficiencies in: "
        advice += ", ".join([f"{k} ({v:.2%})" for k, v in (explanation or {}).items() if v > 0.1])
        advice += ". Recommend applying fertilizers as listed and improving soil management."
    elif prediction == "medium":
        advice += "Tell the farmer that soil fertility is moderate. Address specific deficiencies listed to boost yields."
    elif prediction == "high":
        advice += "Inform the farmer that soil is fertile but maintain nutrient balance with listed recommendations."
    
    return recommendations if recommendations else [translations[lang]["optimal_soil"]], advice

# === STREAMLIT APP ===
st.title(translations["en"]["title"])

# Sidebar for User Type Selection
user_type = st.sidebar.selectbox(translations["en"]["select_user_type"], [translations["en"]["farmer"], translations["en"]["researcher"]], key="user_type")

# Initialize Session State
if 'soil_data' not in st.session_state:
    st.session_state.soil_data = None
if 'dealer_data' not in st.session_state:
    st.session_state.dealer_data = None
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features' not in st.session_state:
    st.session_state.features = []

# Fetch Data Once
if st.session_state.soil_data is None:
    with st.spinner("Fetching soil data for Trans Nzoia..."):
        st.session_state.soil_data = fetch_soil_data("Trans Nzoia", crop="maize")
    with st.spinner("Fetching agro-dealer data..."):
        trans_nzoia_units = [
            {"constituency": "Kiminini", "ward": "Sirende"},
            {"constituency": "Trans Nzoia East", "ward": "Chepsiro/Kiptoror"},
            {"constituency": "Trans Nzoia East", "ward": "Sitatunga"},
            {"constituency": "Kwanza", "ward": "Kapomboi"},
            {"constituency": "Kwanza", "ward": "Kwanza"}
        ]
        constituencies = [unit["constituency"] for unit in trans_nzoia_units]
        wards = [unit["ward"] for unit in trans_nzoia_units]
        st.session_state.dealer_data = fetch_agrodealer_data("Trans Nzoia", constituencies, wards)
    with st.spinner("Fetching weather data..."):
        st.session_state.weather_data = fetch_weather_data()
    if st.session_state.soil_data is not None:
        st.session_state.merged_data = merge_soil_agrodealer_data(st.session_state.soil_data, st.session_state.dealer_data)
        # Train model
        st.session_state.model, st.session_state.scaler, st.session_state.features = train_soil_model(st.session_state.soil_data)

# Farmer Interface
if user_type == translations["en"]["farmer"]:
    # Language Selection
    lang = st.sidebar.selectbox(translations["en"]["select_language"], ["English", "Swahili"], key="language")
    lang_code = "sw" if lang == "Swahili" else "en"
    st.sidebar.write(translations[lang_code]["language_confirmation"])
    
    st.header(translations[lang_code]["farmer_header"])
    st.write(translations[lang_code]["farmer_instruction"])
    
    # Ward Selection
    wards = ["Sirende", "Chepsiro/Kiptoror", "Sitatunga", "Kapomboi", "Kwanza"]
    selected_ward = st.selectbox(translations[lang_code]["select_ward"], wards)
    
    # Crop State Input
    st.subheader(translations[lang_code]["crop_state_header"])
    crop_symptoms = st.multiselect(
        "Select observed crop symptoms",
        translations[lang_code]["crop_symptoms"],
        help="Choose all that apply to your maize crop."
    )
    
    # Weather Information
    if st.session_state.weather_data:
        st.write(translations[lang_code]["weather_info"].format(
            st.session_state.weather_data["temp"],
            st.session_state.weather_data["precipitation"],
            st.session_state.weather_data["description"]
        ))
    
    if st.session_state.merged_data is not None:
        recommendations = get_fertilizer_recommendations_farmer(
            st.session_state.merged_data, selected_ward, crop_symptoms, st.session_state.weather_data, lang=lang_code
        )
        st.subheader(translations[lang_code]["recommendations_header"].format(selected_ward))
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Agro-Dealer Information
        st.subheader(translations[lang_code]["dealers_header"])
        if st.session_state.dealer_data is not None:
            dealers = st.session_state.dealer_data[st.session_state.dealer_data['Ward'] == selected_ward]
            if not dealers.empty:
                st.write(translations[lang_code]["dealers_header"] + ":")
                for _, dealer in dealers.iterrows():
                    st.write(translations[lang_code]["dealer_info"].format(
                        dealer['agrodealerName'], dealer['market'], dealer.get('agrodealerPhone', 'N/A')))
                m = folium.Map(location=[dealers['Latitude'].mean(), dealers['Longitude'].mean()], zoom_start=12)
                for _, dealer in dealers.iterrows():
                    if pd.notnull(dealer['Latitude']) and pd.notnull(dealer['Longitude']):
                        folium.Marker(
                            [dealer['Latitude'], dealer['Longitude']],
                            popup=f"{dealer['agrodealerName']} ({dealer['market']})",
                            icon=folium.Icon(color="green")
                        ).add_to(m)
                folium_static(m)
            else:
                st.write(translations[lang_code]["dealers_none"])
    else:
        st.error(translations[lang_code]["error_data"])

# Researcher/Extension Officer Interface
else:
    st.header(translations["en"]["researcher"])
    st.write("Explore soil data, input soil parameters, and get model-based recommendations for Trans Nzoia maize farming.")
    
    if st.session_state.merged_data is not None:
        wards = st.session_state.merged_data['Ward'].unique().tolist()
        selected_ward = st.selectbox("Select Ward for Analysis", wards)
        ward_data = st.session_state.merged_data[st.session_state.merged_data['Ward'] == selected_ward]
        
        # Soil Parameter Input
        st.subheader("Input Soil Parameters")
        input_data = {}
        with st.form("soil_input_form"):
            input_data["soil_pH"] = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
            input_data["total_Nitrogen_percent_"] = st.number_input("Total Nitrogen (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            input_data["phosphorus_Olsen_ppm"] = st.number_input("Phosphorus (Olsen, ppm)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
            input_data["potassium_meq_percent_"] = st.number_input("Potassium (meq%)", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
            input_data["zinc_ppm"] = st.number_input("Zinc (ppm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            input_data["boron_ppm"] = st.number_input("Boron (ppm)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
            submit_button = st.form_submit_button("Submit Soil Data")
        
        if submit_button:
            recommendations, advice = get_fertilizer_recommendations_researcher(
                input_data, st.session_state.model, st.session_state.scaler, st.session_state.features, lang="en"
            )
            st.subheader("Model-Based Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
            st.write("**Advice for Farmer Communication**:")
            st.write(advice)
        
        # Soil Parameter Statistics
        st.subheader("Soil Parameter Statistics (Historical Data)")
        key_params = [
            "soil_pH", "total_Nitrogen_percent_", "total_Org_Carbon_percent_",
            "phosphorus_Olsen_ppm", "potassium_meq_percent_", "zinc_ppm", "boron_ppm",
            "electr_Conductivity_mS_per_cm"
        ]
        key_params = [col for col in key_params if col in ward_data.columns]
        if key_params:
            st.write(ward_data[key_params].describe())
            st.subheader("Soil Parameter Visualization")
            param = st.selectbox("Select Parameter to Visualize", key_params)
            chart_data = ward_data[param].dropna()
            if not chart_data.empty:
                st.bar_chart(chart_data)
        
        # Agro-Dealer Information
        st.subheader("Agro-Dealer Network")
        if st.session_state.dealer_data is not None:
            dealers = st.session_state.dealer_data[st.session_state.dealer_data['Ward'] == selected_ward]
            if not dealers.empty:
                st.write(dealers[['agrodealerName', 'market', 'agrodealerPhone', 'Latitude', 'Longitude']])
                m = folium.Map(location=[dealers['Latitude'].mean(), dealers['Longitude'].mean()], zoom_start=12)
                for _, dealer in dealers.iterrows():
                    if pd.notnull(dealer['Latitude']) and pd.notnull(dealer['Longitude']):
                        folium.Marker(
                            [dealer['Latitude'], dealer['Longitude']],
                            popup=f"{dealer['agrodealerName']} ({dealer['market']})",
                            icon=folium.Icon(color="green")
                        ).add_to(m)
                folium_static(m)
            else:
                st.write("No agro-dealers found for this ward.")
        
        # Download Data
        st.subheader("Download Data")
        csv = ward_data.to_csv(index=False)
        st.download_button("Download Ward Soil Data", csv, f"{selected_ward}_soil_data.csv", "text/csv")
    else:
        st.error(translations["en"]["error_data"])

# Footer
st.markdown("---")
st.markdown(translations["en"]["footer"])
