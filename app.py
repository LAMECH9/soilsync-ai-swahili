# Data loading for institutional interface
@st.cache_data
def load_and_preprocess_data(source="github"):
    try:
        if source == "github":
            github_raw_url = "https://raw.githubusercontent.com/lamech9/soil-ai/main/cleaned_soilsync_dataset.csv"
            response = requests.get(github_raw_url)
            if response.status_code == 404:
                return None, None, None, None
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        else:
            df = pd.read_csv(source)

        features = ['soil ph', 'total nitrogen', 'phosphorus olsen', 'potassium meq', 
                    'calcium meq', 'magnesium meq', 'manganese meq', 'copper', 'iron', 
                    'zinc', 'sodium meq', 'total org carbon']  # Corrected line
        target_nitrogen = 'total nitrogenclass'
        target_phosphorus = 'phosphorus olsen class'

        required_cols = [col for col in [target_nitrogen, target_phosphorus] if col in df.columns]
        df = df.dropna(subset=[col for col in features if col in df.columns] + required_cols)

        df['nitrogen_class_str'] = df[target_nitrogen] if target_nitrogen in df.columns else 'unknown'
        df['phosphorus_class_str'] = df[target_phosphorus] if target_phosphorus in df.columns else 'unknown'

        if target_nitrogen in df.columns:
            df[target_nitrogen] = df[target_nitrogen].str.lower().map({'low': 0, 'adequate': 1, 'high': 2})
        if target_phosphorus in df.columns:
            df[target_phosphorus] = df[target_phosphorus].str.lower().map({'low': 0, 'adequate': 1, 'high': 2})

        if (target_nitrogen in df.columns and df[target_nitrogen].isnull().any()) or \
           (target_phosphorus in df.columns and df[target_phosphorus].isnull().any()):
            df = df.dropna(subset=[col for col in [target_nitrogen, target_phosphorus] if col in df.columns])

        if 'county' not in df.columns:
            df['county'] = [f"County{i+1}" for i in range(len(df))]

        kenyan_counties = [
            "Kajiado", "Narok", "Nakuru", "Kiambu", "Machakos", "Murang'a", 
            "Nyeri", "Kitui", "Embu", "Meru", "Tharaka Nithi", "Laikipia"
        ]
        if df['county'].str.contains("County").any():
            county_mapping = {f"County{i+1}": kenyan_counties[i % len(kenyan_counties)] for i in range(len(df))}
            df['county'] = df['county'].map(county_mapping).fillna(df['county'])

        return df, features, target_nitrogen, target_phosphorus
    except Exception as e:
        return None, None, None, None
