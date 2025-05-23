# app.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
import folium
from streamlit_folium import folium_static

# === CONFIGURE ===
import os
API_TOKEN = os.getenv("API_TOKEN")
SOIL_API_URL = "https://farmerdb.kalro.org/api/SoilData/legacy/county"
AGRODEALER_API_URL = "https://farmerdb.kalro.org/api/SoilData/agrodealers"

# === TRANSLATIONS DICTIONARY ===
translations = {
    "en": {
        "title": "SoilSync AI: Precision Fertilizer Recommendations for Maize",
        "select_user_type": "Select User Type",
        "farmer": "Farmer",
        "researcher": "Researcher/Extension Officer",
        "farmer_header": "Farmer-Friendly Recommendations",
        "farmer_instruction": "Select your ward to get tailored fertilizer recommendations for maize farming in Trans Nzoia.",
        "select_ward": "Select Your Ward",
        "recommendations_header": "Recommendations for {}",
        "no_data": "No soil data available for recommendations.",
        "optimal_soil": "Soil parameters are within optimal ranges for maize.",
        "dealers_header": "Where to Buy Fertilizers",
        "dealers_none": "No agro-dealers found for this ward. Check county-level suppliers in Kitale or Kwanza markets.",
        "dealer_info": "- **{}** ({}) - Phone: {}",
        "error_data": "Unable to load soil data. Please try again later.",
        "footer": "SoilSync AI by Kibabii University | Powered by KALRO Data | Contact: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Apply **agricultural lime** (1–2 tons/ha) to correct acidic soil (pH {:.2f}).",
        "rec_ph_alkaline": "Use **Ammonium Sulphate** (100–200 kg/ha) to lower alkaline soil (pH {:.2f}).",
        "rec_nitrogen": "Apply **DAP (100–150 kg/ha)** at planting and **CAN (100–200 kg/ha)** or **Urea (50–100 kg/ha)** for top-dressing to address nitrogen deficiency.",
        "rec_phosphorus": "Apply **DAP (100–150 kg/ha)** or **TSP (100–150 kg/ha)** at planting for phosphorus deficiency.",
        "rec_potassium": "Use **NPK 17:17:17 or 23:23:0** (100–150 kg/ha) at planting for potassium deficiency.",
        "rec_zinc": "Apply **Mavuno Maize Fertilizer** or **YaraMila Cereals** for zinc deficiency, or use zinc sulfate foliar spray (5–10 kg/ha).",
        "rec_boron": "Apply **borax** (1–2 kg/ha) for boron deficiency.",
        "rec_organic": "Apply **compost/manure (5–10 tons/ha)** or **Mazao Organic** to boost organic matter.",
        "rec_salinity": "Implement leaching with irrigation and use **Ammonium Sulphate** to manage high salinity."
    },
    "sw": {
        "title": "SoilSync AI: Mapendekezo ya Mbolea ya Usahihi kwa Mahindi",
        "select_user_type": "Chagua Aina ya Mtumiaji",
        "farmer": "Mkulima",
        "researcher": "Mtafiti/Afisa Ugani",
        "farmer_header": "Mapendekezo Yanayofaa Mkulima",
        "farmer_instruction": "Chagua wadi yako ili kupata mapendekezo ya mbolea yanayofaa kwa kilimo cha mahindi huko Trans Nzoia.",
        "select_ward": "Chagua Wadi Yako",
        "recommendations_header": "Mapendekezo kwa {}",
        "no_data": "Hakuna data ya udongo inayopatikana kwa mapendekezo.",
        "optimal_soil": "Vigezo vya udongo viko ndani ya viwango bora kwa mahindi.",
        "dealers_header": "Wapi pa Kununua Mbolea",
        "dealers_none": "Hakuna wauzaji wa mbolea waliopatikana kwa wadi hii. Angalia wauzaji wa ngazi ya kaunti katika soko za Kitale au Kwanza.",
        "dealer_info": "- **{}** ({}) - Simu: {}",
        "error_data": "Imeshindwa kupakia data ya udongo. Tafadhali jaribu tena baadaye.",
        "footer": "SoilSync AI na Chuo Kikuu cha Kibabii | Inatumia Data ya KALRO | Wasiliana: peter.barasa@kibu.ac.ke",
        "rec_ph_acidic": "Tumia **chokaa cha kilimo** (tani 1–2 kwa hekta) kurekebisha udongo wenye tindikali (pH {:.2f}).",
        "rec_ph_alkaline": "Tumia **Ammonium Sulphate** (kg 100–200 kwa hekta) kupunguza udongo wa alkali (pH {:.2f}).",
        "rec_nitrogen": "Tumia **DAP (kg 100–150 kwa hekta)** wakati wa kupanda na **CAN (kg 100–200 kwa hekta)** au **Urea (kg 50–100 kwa hekta)** kwa kurutubisha juu ili kushughulikia upungufu wa nitrojeni.",
        "rec_phosphorus": "Tumia **DAP (kg 100–150 kwa hekta)** au **TSP (kg 100–150 kwa hekta)** wakati wa kupanda kwa upungufu wa fosforasi.",
        "rec_potassium": "Tumia **NPK 17:17:17 au 23:23:0** (kg 100–150 kwa hekta) wakati wa kupanda kwa upungufu wa potasiamu.",
        "rec_zinc": "Tumia **Mbolea ya Mavuno Maize** au **YaraMila Cereals** kwa upungufu wa zinki, au tumia dawa ya zinki ya sulfate (kg 5–10 kwa hekta).",
        "rec_boron": "Tumia **borax** (kg 1–2 kwa hekta) kwa upungufu wa boron.",
        "rec_organic": "Tumia **mbolea ya kikaboni/samadi (tani 5–10 kwa hekta)** au **Mazao Organic** kuongeza vitu vya kikaboni.",
        "rec_salinity": "Tekeleza uchukuzi wa maji na umwagiliaji na tumia **Ammonium Sulphate** kushughulikia chumvi nyingi."
    }
}

# === FETCH SOIL DATA FUNCTION (Adapted from Provided Code) ===
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
        df_filtered = df[available_columns]
        if "crop" in df_filtered.columns:
            df_filtered["crop"] = df_filtered["crop"].astype(str)
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
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        df_filtered = df_filtered.rename(columns={
            "county": "County", "constituency": "Constituency", "ward": "Ward",
            "latitude": "Latitude", "longitude": "Longitude"
        })
        return df_filtered
    except Exception as e:
        st.error(translations["en"]["error_data"])  # Default to English for errors
        return None

# === FETCH AGRO-DEALER DATA FUNCTION (Adapted from Provided Code) ===
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

# === MERGE DATA FUNCTION (Adapted from Provided Code) ===
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
        merged_df['Latitude'] = merged_df['Latitude_soil'].combine_first(merged_df['Latitude_dealer'])
        merged_df['Longitude'] = merged_df['Longitude_soil'].combine_first(merged_df['Longitude_dealer'])
        merged_df = merged_df.drop(columns=['Latitude_soil', 'Longitude_soil', 'Latitude_dealer', 'Longitude_dealer'], errors='ignore')
        return merged_df
    except:
        return soil_df

# === FERTILIZER RECOMMENDATION FUNCTION ===
def get_fertilizer_recommendations(soil_data, ward, lang="en"):
    recommendations = []
    if soil_data.empty:
        return [translations[lang]["no_data"]]
    ward_data = soil_data[soil_data['Ward'] == ward]
    if ward_data.empty:
        ward_data = soil_data  # Fallback to county-level averages
    avg_data = ward_data.mean(numeric_only=True)
    
    # pH Recommendations
    if 'soil_pH' in avg_data and avg_data['soil_pH'] < 5.5:
        recommendations.append(translations[lang]["rec_ph_acidic"].format(avg_data['soil_pH']))
    elif 'soil_pH' in avg_data and avg_data['soil_pH'] > 7.0:
        recommendations.append(translations[lang]["rec_ph_alkaline"].format(avg_data['soil_pH']))
    
    # Nitrogen Recommendations
    if 'total_Nitrogen_percent_' in avg_data and avg_data['total_Nitrogen_percent_'] < 0.2:
        recommendations.append(translations[lang]["rec_nitrogen"])
    
    # Phosphorus Recommendations
    if 'phosphorus_Olsen_ppm' in avg_data and avg_data['phosphorus_Olsen_ppm'] < 15:
        recommendations.append(translations[lang]["rec_phosphorus"])
    
    # Potassium Recommendations
    if 'potassium_meq_percent_' in avg_data and avg_data['potassium_meq_percent_'] < 0.2:
        recommendations.append(translations[lang]["rec_potassium"])
    
    # Micronutrient Recommendations
    if 'zinc_ppm' in avg_data and avg_data['zinc_ppm'] < 1:
        recommendations.append(translations[lang]["rec_zinc"])
    if 'boron_ppm' in avg_data and avg_data['boron_ppm'] < 0.5:
        recommendations.append(translations[lang]["rec_boron"])
    
    # Organic Matter Recommendations
    if 'total_Org_Carbon_percent_' in avg_data and avg_data['total_Org_Carbon_percent_'] < 1:
        recommendations.append(translations[lang]["rec_organic"])
    
    # Salinization Recommendations
    if 'electr_Conductivity_mS_per_cm' in avg_data and avg_data['electr_Conductivity_mS_per_cm'] > 1:
        recommendations.append(translations[lang]["rec_salinity"])
    
    return recommendations if recommendations else [translations[lang]["optimal_soil"]]

# === STREAMLIT APP ===
st.title(translations["en"]["title"])  # Default title in English

# Sidebar for User Type Selection
user_type = st.sidebar.selectbox(translations["en"]["select_user_type"], [translations["en"]["farmer"], translations["en"]["researcher"]], key="user_type")

# Initialize Session State
if 'soil_data' not in st.session_state:
    st.session_state.soil_data = None
if 'dealer_data' not in st.session_state:
    st.session_state.dealer_data = None
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None

# Fetch Data Once
if st.session_state.soil_data is None:
    with st.spinner("Fetching soil data for Trans Nzoia..." if user_type == translations["en"]["farmer"] else translations["en"]["error_data"]):
        st.session_state.soil_data = fetch_soil_data("Trans Nzoia", crop="maize")
    with st.spinner("Fetching agro-dealer data..." if user_type == translations["en"]["farmer"] else translations["en"]["error_data"]):
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
    if st.session_state.soil_data is not None:
        st.session_state.merged_data = merge_soil_agrodealer_data(st.session_state.soil_data, st.session_state.dealer_data)

# Farmer Interface
if user_type == translations["en"]["farmer"]:
    # Language Selection
    lang = st.sidebar.selectbox("Chagua Lugha / Select Language", ["English", "Swahili"], key="language")
    lang_code = "sw" if lang == "Swahili" else "en"
    
    st.header(translations[lang_code]["farmer_header"])
    st.write(translations[lang_code]["farmer_instruction"])
    
    wards = ["Sirende", "Chepsiro/Kiptoror", "Sitatunga", "Kapomboi", "Kwanza"]
    selected_ward = st.selectbox(translations[lang_code]["select_ward"], wards)
    
    if st.session_state.merged_data is not None:
        recommendations = get_fertilizer_recommendations(st.session_state.merged_data, selected_ward, lang=lang_code)
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
                # Map Visualization
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
    st.write("Explore soil data, fertilizer recommendations, and analytics for Trans Nzoia maize farming.")
    
    if st.session_state.merged_data is not None:
        wards = st.session_state.merged_data['Ward'].unique().tolist()
        selected_ward = st.selectbox("Select Ward for Analysis", wards)
        ward_data = st.session_state.merged_data[st.session_state.merged_data['Ward'] == selected_ward]
        
        # Fertilizer Recommendations
        st.subheader(f"Recommendations for {selected_ward}")
        recommendations = get_fertilizer_recommendations(st.session_state.merged_data, selected_ward, lang="en")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Soil Parameter Statistics
        st.subheader("Soil Parameter Statistics")
        key_params = [
            "soil_pH", "total_Nitrogen_percent_", "total_Org_Carbon_percent_",
            "phosphorus_Olsen_ppm", "potassium_meq_percent_", "zinc_ppm", "boron_ppm",
            "electr_Conductivity_mS_per_cm"
        ]
        key_params = [col for col in key_params if col in ward_data.columns]
        if key_params:
            st.write(ward_data[key_params].describe())
            # Bar Chart for Key Parameters
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
