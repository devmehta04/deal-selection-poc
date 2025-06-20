import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Load Data ---
@st.cache_data # Cache data loading for performance, only runs once
def load_data():
    """Loads fund data from CSV and performs initial data type conversions."""
    try:
        df = pd.read_csv('funds_data.csv')
    except FileNotFoundError:
        st.error("Error: funds_data.csv not found. Please ensure the file is in the same directory.")
        st.stop() # Stop execution if data file is missing

    # Convert 'InceptionDate' to datetime for age calculation
    df['InceptionDate'] = pd.to_datetime(df['InceptionDate'], errors='coerce') # Coerce errors to NaT (Not a Time)

    # Convert relevant columns to numeric, coercing errors to NaN
    # ADDED ROLLING RETURN COLUMNS HERE
    numeric_cols = [
        'CAGR_3Y', 'CAGR_5Y', 'MaxDrawdown', 'StdDev', 'SharpeRatio', 'SortinoRatio',
        'Alpha', 'UpsideCapture', 'DownsideCapture', 'AUM_Cr', 'TER_pct', 'MinInvestment_Cr',
        'RollingReturn_1Y', 'RollingReturn_3Y', 'RollingReturn_5Y' # NEW: Rolling Returns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce non-numeric values to NaN

    return df

funds_df = load_data()

# --- 2. Define Scoring Logic (Functions & Rubrics) ---

def normalize_score(value, range_min, range_max, higher_is_better=True):
    """
    Normalizes a value to a 0-100 score. 100 is always "best", 0 is "worst".
    Handles missing values and edge cases where min == max.
    """
    if pd.isna(value):
        return 50 # Neutral score for missing values

    # Avoid division by zero if all values in range are the same
    if range_min == range_max:
        # If all values are the same, assign full score if higher is better, else neutral or 0.
        return 100 if higher_is_better else 0

    # Calculate normalized value (0 to 1)
    normalized_val_0_1 = (value - range_min) / (range_max - range_min)

    if higher_is_better:
        return normalized_val_0_1 * 100 # Higher value -> higher score
    else: # lower_is_better, so invert the score (higher value -> lower score -> lower final score)
        return (1 - normalized_val_0_1) * 100

# Qualitative Scoring Rubrics: Define rules and their score impact
QUALITATIVE_RUBRICS = {
    "Qual_ManagerReview": {
        "score_range": (-3, 3), # Internal scoring range for raw keywords
        "rules": [
            ("Excellent / Highly experienced / Star / 20+ years / Stable team", +3),
            ("Good / Promising / Seasoned / 10+ years / Consistent team", +2),
            ("Solid / Veteran / 5+ years", +1),
            ("Average / Basic", 0),
            ("Young / Unproven / Still forming / Average responsiveness", -1),
            ("Weak / Limited experience / High turnover", -2),
            ("Incomplete / No data / Not applicable / nan", -3) # Catch missing/placeholder text for max penalty
        ]
    },
    "Qual_PhilosophyReview": {
        "score_range": (-3, 3),
        "rules": [
            ("Clear / Consistent / Robust / Deep dive / Well-defined", +3),
            ("Adaptive / Balanced / Strategic / Focused", +2),
            ("Solid / Conservative / Bottom-up", +1),
            ("Average / Standard", 0),
            ("Vague / Inconsistent / Unclear", -1),
            ("Aggressive (if not aligned with risk) / Unproven approach", -2),
            ("Incomplete / No data / nan", -3)
        ]
    },
    "Qual_HouseReview": {
        "score_range": (-3, 3),
        "rules": [
            ("Strong brand / Established / Clean regulatory record / High transparency", +3),
            ("Reputable / Growing reputation / No SEBI issues", +2),
            ("Solid / Reliable / Mid-sized", +1),
            ("Average / Standard", 0),
            ("Newer / Smaller / Gaining traction / One minor regulatory notice", -1),
            ("Unestablished / Problematic / Major SEBI issues / Lack of transparency", -3),
            ("Incomplete / No data / nan", -3)
        ]
    },
    "Qual_ClientServiceReview": {
        "score_range": (-3, 3),
        "rules": [
            ("Excellent / Very responsive / Personalized / Proactive / Bespoke reporting", +3),
            ("Good / Responsive / Detailed reports / Educational material", +2),
            ("Standard / Regular updates / Efficient", +1),
            ("Average / Basic reports / Impersonal", 0),
            ("Average responsiveness / Digital-first (if not preferred)", -1),
            ("Limited communication / Poor / Unresponsive", -2),
            ("Incomplete / No data / nan", -3)
        ]
    }
}

def score_qualitative_review(text, review_type):
    """
    Scores qualitative text reviews based on predefined rules/keywords.
    Penalizes heavily for 'incomplete' or missing data.
    """
    text_processed = str(text).lower() # Convert to string and lowercase for consistent matching
    score = 0
    rubric = QUALITATIVE_RUBRICS.get(review_type)
    if not rubric:
        return 0 # Fallback if review_type is not defined in rubrics

    # Check for explicit "incomplete" or NaN first, as it's a strong indicator for max penalty
    if "incomplete" in text_processed or pd.isna(text) or text_processed == 'nan' or text_processed.strip() == '':
        return rubric["rules"][-1][1] # Return the max penalty from the last rule (e.g., -3)

    # Iterate through rules and apply points if keywords are found
    for keyword_phrase, points in rubric["rules"]:
        # Skip the 'incomplete' rule here as it's handled above
        if "incomplete / no data" in keyword_phrase.lower():
            continue
            
        # Split the keyword_phrase by ' / ' to get individual keywords/phrases
        individual_keywords = [k.strip().lower() for k in keyword_phrase.split('/') if k.strip()] # Ensure no empty strings
        
        # Check if any of the individual keywords are present in the review text
        for individual_keyword in individual_keywords:
            if individual_keyword in text_processed:
                score += points
                break # Apply points for this rule once and move to next rule (avoid double counting from same rule)

    # Cap score within the defined range to prevent runaway scores
    min_score, max_score = rubric["score_range"]
    return max(min_score, min(max_score, score))

# Apply raw qualitative scoring to the DataFrame
funds_df['Qual_ManagerScore'] = funds_df.apply(lambda row: score_qualitative_review(row['Qual_ManagerReview'], 'Qual_ManagerReview'), axis=1)
funds_df['Qual_PhilosophyScore'] = funds_df.apply(lambda row: score_qualitative_review(row['Qual_PhilosophyReview'], 'Qual_PhilosophyReview'), axis=1)
funds_df['Qual_HouseScore'] = funds_df.apply(lambda row: score_qualitative_review(row['Qual_HouseReview'], 'Qual_HouseReview'), axis=1)
funds_df['Qual_ClientServiceScore'] = funds_df.apply(lambda row: score_qualitative_review(row['Qual_ClientServiceReview'], 'Qual_ClientServiceReview'), axis=1)

# Calculate and store Normalized Qualitative Scores (0-100)
funds_df['Qual_ManagerScore_Normalized'] = funds_df.apply(lambda row: normalize_score(row['Qual_ManagerScore'], QUALITATIVE_RUBRICS['Qual_ManagerReview']['score_range'][0], QUALITATIVE_RUBRICS['Qual_ManagerReview']['score_range'][1], True), axis=1)
funds_df['Qual_PhilosophyScore_Normalized'] = funds_df.apply(lambda row: normalize_score(row['Qual_PhilosophyScore'], QUALITATIVE_RUBRICS['Qual_PhilosophyReview']['score_range'][0], QUALITATIVE_RUBRICS['Qual_PhilosophyReview']['score_range'][1], True), axis=1)
funds_df['Qual_HouseScore_Normalized'] = funds_df.apply(lambda row: normalize_score(row['Qual_HouseScore'], QUALITATIVE_RUBRICS['Qual_HouseReview']['score_range'][0], QUALITATIVE_RUBRICS['Qual_HouseReview']['score_range'][1], True), axis=1)
funds_df['Qual_ClientServiceScore_Normalized'] = funds_df.apply(lambda row: normalize_score(row['Qual_ClientServiceScore'], QUALITATIVE_RUBRICS['Qual_ClientServiceReview']['score_range'][0], QUALITATIVE_RUBRICS['Qual_ClientServiceReview']['score_range'][1], True), axis=1)


def calculate_data_reliability(row):
    """Calculates a data reliability score based on completeness of critical fields."""
    # ADDED ROLLING RETURN COLUMNS HERE
    critical_fields = [
        'CAGR_3Y', 'MaxDrawdown', 'SharpeRatio', 'Alpha', 'AUM_Cr', 'TER_pct',
        'Qual_ManagerReview', 'Qual_PhilosophyReview', 'Qual_HouseReview', 'Qual_ClientServiceReview',
        'RollingReturn_1Y', 'RollingReturn_3Y', 'RollingReturn_5Y' # NEW: Rolling Returns for reliability check
    ]
    completed_fields_count = 0
    for field in critical_fields:
        value = row[field]
        # Check if a field has a valid, non-missing value
        if not pd.isna(value) and str(value).lower() not in ['incomplete', 'nan', 'not applicable', '']:
            completed_fields_count += 1

    reliability_score_percentage = (completed_fields_count / len(critical_fields)) * 100
    return reliability_score_percentage

funds_df['DataReliabilityScore'] = funds_df.apply(calculate_data_reliability, axis=1)


# Define Normalization Ranges (Crucial for consistent scoring across funds)
# These are dynamically calculated based on the *entire* dataset's min/max.
quant_min_max = {}
# ADDED ROLLING RETURN COLUMNS HERE
for col in [
    'CAGR_3Y', 'CAGR_5Y', 'MaxDrawdown', 'StdDev', 'SharpeRatio', 'SortinoRatio',
    'Alpha', 'UpsideCapture', 'DownsideCapture', 'AUM_Cr', 'TER_pct', 'MinInvestment_Cr',
    'RollingReturn_1Y', 'RollingReturn_3Y', 'RollingReturn_5Y' # NEW: Rolling Returns
]:
    actual_min = funds_df[col].min()
    actual_max = funds_df[col].max()

    # Handle cases where column might be empty or have no variance
    if pd.isna(actual_min) or pd.isna(actual_max):
        actual_min = 0 # Default min
        actual_max = 100 # Default max if data is entirely missing
    if actual_min == actual_max:
        actual_max += 0.01 # Add a tiny epsilon to prevent division by zero

    quant_min_max[col] = {'min': actual_min, 'max': actual_max}

# Add 'higher_is_better' flag explicitly to each metric for clarity
# This tells the normalize_score function whether higher values are better (True) or worse (False)
quant_min_max['CAGR_3Y']['higher_is_better'] = True
quant_min_max['CAGR_5Y']['higher_is_better'] = True
quant_min_max['MaxDrawdown']['higher_is_better'] = False # Lower (less negative) drawdown is better
quant_min_max['StdDev']['higher_is_better'] = False
quant_min_max['SharpeRatio']['higher_is_better'] = True
quant_min_max['SortinoRatio']['higher_is_better'] = True
quant_min_max['Alpha']['higher_is_better'] = True
quant_min_max['UpsideCapture']['higher_is_better'] = True
quant_min_max['DownsideCapture']['higher_is_better'] = False # Lower downside capture is better
quant_min_max['AUM_Cr']['higher_is_better'] = True
quant_min_max['TER_pct']['higher_is_better'] = False
quant_min_max['MinInvestment_Cr']['higher_is_better'] = False
# NEW: Rolling Returns are also higher is better
quant_min_max['RollingReturn_1Y']['higher_is_better'] = True
quant_min_max['RollingReturn_3Y']['higher_is_better'] = True
quant_min_max['RollingReturn_5Y']['higher_is_better'] = True


# Pre-calculate individual normalized quantitative metrics
# ADDED ROLLING RETURN COLUMNS HERE
for col in ['CAGR_3Y', 'CAGR_5Y', 'MaxDrawdown', 'StdDev', 'SharpeRatio', 'SortinoRatio',
            'Alpha', 'UpsideCapture', 'DownsideCapture', 'AUM_Cr', 'TER_pct', 'MinInvestment_Cr',
            'RollingReturn_1Y', 'RollingReturn_3Y', 'RollingReturn_5Y']: # NEW: Rolling Returns
    funds_df[f'{col}_Normalized'] = funds_df.apply(
        lambda row: normalize_score(row[col], quant_min_max[col]['min'], quant_min_max[col]['max'], quant_min_max[col]['higher_is_better']),
        axis=1
    )

# Pre-calculate aggregated normalized quantitative scores
# UPDATED RETURNS_SCORE_NORMALIZED TO INCLUDE ROLLING RETURNS
funds_df['Returns_Score_Normalized'] = funds_df.apply(
    lambda row: np.mean([
        row['CAGR_3Y_Normalized'] if pd.notna(row['CAGR_3Y_Normalized']) else 50,
        row['CAGR_5Y_Normalized'] if pd.notna(row['CAGR_5Y_Normalized']) else 50,
        row['RollingReturn_1Y_Normalized'] if pd.notna(row['RollingReturn_1Y_Normalized']) else 50,
        row['RollingReturn_3Y_Normalized'] if pd.notna(row['RollingReturn_3Y_Normalized']) else 50,
        row['RollingReturn_5Y_Normalized'] if pd.notna(row['RollingReturn_5Y_Normalized']) else 50
    ]),
    axis=1
)

funds_df['Risk_Score_Normalized'] = (funds_df['MaxDrawdown_Normalized'] + funds_df['StdDev_Normalized']) / 2
funds_df['RiskAdj_Score_Normalized'] = (funds_df['SharpeRatio_Normalized'] + funds_df['SortinoRatio_Normalized']) / 2
funds_df['ManagerSkill_Quant_Score_Normalized'] = (funds_df['Alpha_Normalized'] + funds_df['UpsideCapture_Normalized'] + funds_df['DownsideCapture_Normalized']) / 3
funds_df['FundChar_Score_Normalized'] = (funds_df['AUM_Cr_Normalized'] + funds_df['TER_pct_Normalized'] + funds_df['MinInvestment_Cr_Normalized']) / 3


def calculate_composite_score(row, weights):
    """
    Calculates the composite score for a fund based on weighted, normalized metrics.
    Now uses pre-calculated normalized scores.
    """
    total_score = 0

    # Retrieve pre-calculated normalized scores directly from the row
    score_returns = row['Returns_Score_Normalized']
    score_risk = row['Risk_Score_Normalized']
    score_risk_adj = row['RiskAdj_Score_Normalized']
    score_manager_skill_quant = row['ManagerSkill_Quant_Score_Normalized']
    score_fund_char = row['FundChar_Score_Normalized']

    norm_manager_qual_score = row['Qual_ManagerScore_Normalized']
    norm_philosophy_score = row['Qual_PhilosophyScore_Normalized']
    norm_house_score = row['Qual_HouseScore_Normalized']
    norm_client_service_score = row['Qual_ClientServiceScore_Normalized']


    # 3. Weighted Sum (apply user-defined weights to the normalized 0-100 scores)
    total_score += weights["Quantitative Returns"] * score_returns
    total_score += weights["Quantitative Risk"] * score_risk
    total_score += weights["Risk-Adjusted Returns"] * score_risk_adj
    total_score += weights["Manager Skill & Efficiency"] * score_manager_skill_quant
    total_score += weights["Investment Philosophy & Process"] * norm_philosophy_score
    total_score += weights["Fund House Strength & Goodwill"] * norm_house_score
    total_score += weights["Client Servicing & Operations"] * norm_client_service_score
    total_score += weights["Fund Characteristics"] * score_fund_char


    # 4. New Fund Adjustment (Penalty for very short track record)
    # Ensure InceptionDate is not NaT before calculating age
    if pd.notna(row['InceptionDate']):
        fund_age_years = (pd.to_datetime('today') - row['InceptionDate']).days / 365
        if fund_age_years < 3:
            total_score *= 0.95 # Small penalty for shorter track record stability

    return total_score

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="PMS/AIF Fund Rater")
st.title("PMS/AIF Fund Shortlisting & Rating Tool")

# --- How It Works Section (New) ---
st.header("💡 How It Works")
with st.expander("Understand the Methodology"):
    st.markdown("""
    This tool helps you shortlist and rank PMS/AIF funds based on a combined score of quantitative and qualitative metrics.

    1.  **Data Loading**: Reads fund data from `funds_data.csv`.
    2.  **Quantitative Scoring**: All quantitative metrics are normalized to a **0-100 scale**, where 100 is "best" and 0 is "worst" for that metric across the dataset. These normalized individual metrics are then aggregated into key quantitative categories:
        * **Quantitative Returns**: Includes CAGR_3Y (3-Year Compounded Annual Growth Rate), CAGR_5Y (5-Year Compounded Annual Growth Rate), **RollingReturn_1Y (1-Year Rolling Return), RollingReturn_3Y (3-Year Rolling Return), and RollingReturn_5Y (5-Year Rolling Return)**.
        * **Quantitative Risk**: Includes MaxDrawdown (Maximum Drawdown) and StdDev (Standard Deviation).
        * **Risk-Adjusted Returns**: Includes SharpeRatio (Sharpe Ratio) and SortinoRatio (Sortino Ratio).
        * **Manager Skill & Efficiency**: Includes Alpha, UpsideCapture, and DownsideCapture.
        * **Fund Characteristics**: Includes AUM_Cr (Assets Under Management in Crores), TER_pct (Total Expense Ratio percentage), and MinInvestment_Cr (Minimum Investment in Crores).
    3.  **Qualitative Scoring**: Assigns scores to qualitative reviews (e.g., Manager Quality, Fund House Strength) based on predefined keywords. These raw scores are then **normalized to a 0-100 scale**.
    4.  **Data Reliability**: Calculates a percentage score for each fund based on the completeness of its critical data points. Funds below a certain threshold are excluded.
    5.  **Composite Score**: Combines all normalized (0-100) quantitative and qualitative scores using your customized weights.
    6.  **Ranking**: Funds are ranked by their final composite score.
    """)

# --- Display Qualitative Rubrics ---
st.header("Qualitative Scoring Rubrics")
with st.expander("Click to view detailed scoring rules"):
    st.markdown("Each qualitative review (e.g., for Manager Quality) is scanned for keywords. Positive keywords add points, negative ones deduct them. The resulting **internal raw score** is then **normalized to a 0-100 scale** for consistent weighting.")
    st.markdown("**Note:** If a review explicitly states 'Incomplete', 'No data', or is empty/missing, it receives the lowest possible raw score for that category (-3), which normalizes to 0.")
    for q_type, rubric_details in QUALITATIVE_RUBRICS.items():
        st.subheader(f"Rules for {q_type.replace('Qual_', '').replace('Review', '').replace('_', ' ').title()}")
        st.markdown(f"**Internal Raw Score Range:** `{rubric_details['score_range'][0]}` to `{rubric_details['score_range'][1]}`")
        st.markdown("**Keyword Phrases & Internal Raw Score Impact:**")
        for keyword_phrase, score_impact in rubric_details["rules"]:
            st.markdown(f"- **'{keyword_phrase}'**: Raw score impact of `{score_impact}`")
    st.info("These internal raw scores are then **normalized to a 0-100 scale** before being weighted into the composite score.")


# --- Sidebar for Filters and Weights ---
st.sidebar.header("Customize Your Search")

# 1. Filters
st.sidebar.subheader("1. Filters")
all_fund_types = funds_df['FundType'].unique().tolist()
selected_fund_types = st.sidebar.multiselect("Fund Type", all_fund_types, default=all_fund_types)

all_strategies = funds_df['Strategy'].unique().tolist()
selected_strategies = st.sidebar.multiselect("Strategy", all_strategies, default=all_strategies)

min_inv_filter = st.sidebar.number_input("Minimum Investment (Cr)", min_value=0.0, value=0.0, format="%.2f")
min_aum_filter = st.sidebar.number_input("Minimum AUM (Cr)", min_value=0.0, value=0.0, format="%.2f")

# Dynamic Data Reliability Threshold (New User Control)
st.sidebar.markdown("---")
st.sidebar.subheader("3. Data Quality Threshold")
data_reliability_threshold = st.sidebar.slider(
    "Minimum Data Reliability Score (%)",
    min_value=0, max_value=100, value=70, step=5,
    help="Funds with less than this percentage of critical data points will be excluded."
)


# Apply Filters
# Ensure we operate on a copy to avoid SettingWithCopyWarning
filtered_df = funds_df[
    (funds_df['FundType'].isin(selected_fund_types)) &
    (funds_df['Strategy'].isin(selected_strategies)) &
    (funds_df['MinInvestment_Cr'].fillna(0) >= min_inv_filter) & # Fill NaN for filtering
    (funds_df['AUM_Cr'].fillna(0) >= min_aum_filter) # Fill NaN for filtering
].copy()


# Apply Data Reliability Threshold
initial_count_after_filters = len(filtered_df)
filtered_df = filtered_df[filtered_df['DataReliabilityScore'] >= data_reliability_threshold]
final_count_after_reliability = len(filtered_df)

if initial_count_after_filters > final_count_after_reliability:
    st.sidebar.info(f"Excluded {initial_count_after_filters - final_count_after_reliability} fund(s) due to Data Reliability < {data_reliability_threshold}%.")

# --- Weights ---
st.sidebar.subheader("2. Weighting (Total 100%)")
default_weights = {
    "Quantitative Returns": 15,
    "Quantitative Risk": 15,
    "Risk-Adjusted Returns": 20,
    "Manager Skill & Efficiency": 15,
    "Investment Philosophy & Process": 10,
    "Fund House Strength & Goodwill": 10,
    "Client Servicing & Operations": 5,
    "Fund Characteristics": 10,
}

weights = {}
total_weights_sum = 0
for category, default_w in default_weights.items():
    weights[category] = st.sidebar.slider(f"Weight for {category}", 0, 100, default_w, key=f"weight_{category}")
    total_weights_sum += weights[category]

# User feedback on total weights
if total_weights_sum != 100:
    st.sidebar.warning(f"Total weights sum to {total_weights_sum}%. Adjust to 100% for balanced scoring.")

# Normalize weights internally for calculation
normalized_weights = {}
if total_weights_sum > 0:
    for key, val in weights.items():
        normalized_weights[key] = val / total_weights_sum
else: # Fallback if all weights are zero (to avoid division by zero later)
    num_categories = len(default_weights)
    if num_categories > 0:
        for key in weights:
            normalized_weights[key] = 1 / num_categories # Equal distribution
    else:
        normalized_weights = {k: 0 for k in default_weights} # All weights zero if no categories

# --- Display Results ---
st.header("Ranked Funds Shortlist")

if not funds_df.empty and not filtered_df.empty:
    # Calculate Composite Score for filtered funds
    filtered_df['Composite Score'] = filtered_df.apply(lambda row: calculate_composite_score(row, normalized_weights), axis=1)

    # Ranking
    ranked_df = filtered_df.sort_values(by="Composite Score", ascending=False).reset_index(drop=True)
    ranked_df['Rank'] = ranked_df.index + 1

    # Display key columns in the main table with clear formatting
    display_cols = [
        'Rank', 'FundName', 'Composite Score', 'Returns_Score_Normalized', 'Risk_Score_Normalized',
        'RiskAdj_Score_Normalized', 'Qual_ManagerScore_Normalized', 'DataReliabilityScore', 'AUM_Cr', 'TER_pct'
    ]
    st.dataframe(ranked_df[display_cols].style.format({
        'Returns_Score_Normalized': "{:.0f}%",
        'Risk_Score_Normalized': "{:.0f}%",
        'RiskAdj_Score_Normalized': "{:.0f}%",
        'TER_pct': "{:.2f}%",
        'AUM_Cr': "₹{:,.0f} Cr", # Format AUM as Indian Rupees Crores
        'MinInvestment_Cr': "₹{:,.1f} Cr",
        'Qual_ManagerScore_Normalized': "{:.0f}%", # Display as percentage
        'DataReliabilityScore': "{:.0f}%",
        'Composite Score': "{:.2f}"
    }), use_container_width=True)

    st.subheader("View Detailed Fund Profile")
    # Ensure selectbox only shows funds actually present in ranked_df
    fund_names_for_selectbox = ranked_df['FundName'].tolist()
    if fund_names_for_selectbox:
        selected_fund_name = st.selectbox("Select a fund for detailed view:", fund_names_for_selectbox)

        if selected_fund_name:
            selected_fund_data = ranked_df[ranked_df['FundName'] == selected_fund_name].iloc[0]
            st.write(f"### Detailed Profile for {selected_fund_name}")
            with st.expander("Show All Details"):
                st.markdown(f"**Composite Score:** `{selected_fund_data['Composite Score']:.2f}`")
                st.markdown(f"**Data Reliability Score:** `{selected_fund_data['DataReliabilityScore']:.2f}%`")
                
                st.markdown("#### Quantitative Metrics & Scores")
                # Individual Raw Metrics & Their Normalized Scores
                st.markdown(f"**Returns (3Y CAGR):** `{selected_fund_data['CAGR_3Y']:.2f}%` (Normalized: `{selected_fund_data['CAGR_3Y_Normalized']:.0f}%)`")
                st.markdown(f"**Returns (5Y CAGR):** `{selected_fund_data['CAGR_5Y']:.2f}%` (Normalized: `{selected_fund_data['CAGR_5Y_Normalized']:.0f}%)`")
                st.markdown(f"**Rolling Return (1Y):** `{selected_fund_data['RollingReturn_1Y']:.2f}%` (Normalized: `{selected_fund_data['RollingReturn_1Y_Normalized']:.0f}%)`") # NEW
                st.markdown(f"**Rolling Return (3Y):** `{selected_fund_data['RollingReturn_3Y']:.2f}%` (Normalized: `{selected_fund_data['RollingReturn_3Y_Normalized']:.0f}%)`") # NEW
                st.markdown(f"**Rolling Return (5Y):** `{selected_fund_data['RollingReturn_5Y']:.2f}%` (Normalized: `{selected_fund_data['RollingReturn_5Y_Normalized']:.0f}%)`") # NEW
                st.markdown(f"**Max Drawdown:** `{selected_fund_data['MaxDrawdown']:.2f}%` (Normalized: `{selected_fund_data['MaxDrawdown_Normalized']:.0f}%)`")
                st.markdown(f"**Standard Deviation:** `{selected_fund_data['StdDev']:.2f}%` (Normalized: `{selected_fund_data['StdDev_Normalized']:.0f}%)`")
                st.markdown(f"**Sharpe Ratio:** `{selected_fund_data['SharpeRatio']:.2f}` (Normalized: `{selected_fund_data['SharpeRatio_Normalized']:.0f}%)`")
                st.markdown(f"**Sortino Ratio:** `{selected_fund_data['SortinoRatio']:.2f}` (Normalized: `{selected_fund_data['SortinoRatio_Normalized']:.0f}%)`")
                st.markdown(f"**Alpha:** `{selected_fund_data['Alpha']:.2f}` (Normalized: `{selected_fund_data['Alpha_Normalized']:.0f}%)`")
                st.markdown(f"**Upside Capture:** `{selected_fund_data['UpsideCapture']:.2f}%` (Normalized: `{selected_fund_data['UpsideCapture_Normalized']:.0f}%)`")
                st.markdown(f"**Downside Capture:** `{selected_fund_data['DownsideCapture']:.2f}%` (Normalized: `{selected_fund_data['DownsideCapture_Normalized']:.0f}%)`")
                st.markdown(f"**AUM (Cr):** `₹{selected_fund_data['AUM_Cr']:,} Cr` (Normalized: `{selected_fund_data['AUM_Cr_Normalized']:.0f}%)`")
                st.markdown(f"**TER (%):** `{selected_fund_data['TER_pct']:.2f}%` (Normalized: `{selected_fund_data['TER_pct_Normalized']:.0f}%)`")
                st.markdown(f"**Minimum Investment (Cr):** `₹{selected_fund_data['MinInvestment_Cr']:,} Cr` (Normalized: `{selected_fund_data['MinInvestment_Cr_Normalized']:.0f}%)`")
                
                st.markdown("---")
                st.markdown("#### Aggregated Quantitative Category Scores (Normalized)")
                st.markdown(f"**Returns Category Score:** `{selected_fund_data['Returns_Score_Normalized']:.0f}%`")
                st.markdown(f"**Risk Category Score:** `{selected_fund_data['Risk_Score_Normalized']:.0f}%`")
                st.markdown(f"**Risk-Adjusted Returns Category Score:** `{selected_fund_data['RiskAdj_Score_Normalized']:.0f}%`")
                st.markdown(f"**Manager Skill & Efficiency Category Score (Quant):** `{selected_fund_data['ManagerSkill_Quant_Score_Normalized']:.0f}%`")
                st.markdown(f"**Fund Characteristics Category Score:** `{selected_fund_data['FundChar_Score_Normalized']:.0f}%`")


                st.markdown("---")
                st.markdown("#### Qualitative Scores & Reviews")
                # Updated to show both normalized and raw scores for clarity
                st.markdown(f"**Manager & Team Quality:** `{selected_fund_data['Qual_ManagerScore_Normalized']:.0f}%` (Raw: `{selected_fund_data['Qual_ManagerScore']:.0f}`, Review: `'{selected_fund_data['Qual_ManagerReview']}'`)")
                st.markdown(f"**Investment Philosophy & Process:** `{selected_fund_data['Qual_PhilosophyScore_Normalized']:.0f}%` (Raw: `{selected_fund_data['Qual_PhilosophyScore']:.0f}`, Review: `'{selected_fund_data['Qual_PhilosophyReview']}'`)")
                st.markdown(f"**Fund House Strength & Goodwill:** `{selected_fund_data['Qual_HouseScore_Normalized']:.0f}%` (Raw: `{selected_fund_data['Qual_HouseScore']:.0f}`, Review: `'{selected_fund_data['Qual_HouseReview']}'`)")
                st.markdown(f"**Client Servicing & Operations:** `{selected_fund_data['Qual_ClientServiceScore_Normalized']:.0f}%` (Raw: `{selected_fund_data['Qual_ClientServiceScore']:.0f}`, Review: `'{selected_fund_data['Qual_ClientServiceReview']}'`)")

                st.markdown("---")
                st.markdown("#### General Fund Information")
                st.markdown(f"Fund Type / Strategy: `{selected_fund_data['FundType']}` / `{selected_fund_data['Strategy']}`")
                # Check if InceptionDate is NaT before formatting
                if pd.notna(selected_fund_data['InceptionDate']):
                    st.markdown(f"Inception Date: `{selected_fund_data['InceptionDate'].strftime('%Y-%m-%d')}`")
                else:
                    st.markdown("Inception Date: `N/A`")
                    
    else:
        st.info("No funds available to select for detailed view based on current filters.")

elif funds_df.empty:
    st.warning("The funds_data.csv file is empty or could not be loaded. Please check your data file.")
else:
    st.info("No funds match your current filters and data reliability threshold. Please adjust your criteria.")