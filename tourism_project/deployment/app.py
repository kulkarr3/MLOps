
import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏖️",
    layout="wide"
)

# =========================================================
# Model Path
# =========================================================

MODEL_PATH = "tourism_project/deployment/tourism_model.pkl"


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()


# =========================================================
# Training Feature Order
# IMPORTANT:
# This must match the columns used during model training.
# =========================================================

FEATURE_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "IncomeCategory",
    "AgeGroup"
]


# =========================================================
# Encoding Maps
#
# IMPORTANT:
# These mappings MUST be exactly the same as the mappings
# used while preparing the training dataset.
# =========================================================

TYPE_OF_CONTACT_MAP = {
    "Company Invited": 0,
    "Self Enquiry": 1
}

OCCUPATION_MAP = {
    "Salaried": 1,
    "Small Business": 2,
    "Large Business": 3,
    "Free Lancer": 4
}

GENDER_MAP = {
    "Male": 1,
    "Female": 0
}

PRODUCT_PITCHED_MAP = {
    "Basic": 0,
    "Standard": 1,
    "Deluxe": 2,
    "Super Deluxe": 3,
    "King": 4
}

MARITAL_STATUS_MAP = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

DESIGNATION_MAP = {
    "Executive": 1,
    "Manager": 2,
    "Senior Manager": 3,
    "AVP": 4,
    "VP": 5
}


# =========================================================
# Header
# =========================================================

st.title("🏖️ Tourism Package Purchase Prediction")

st.write(
    "Enter customer and travel details to predict whether "
    "the customer is likely to purchase the tourism package."
)


# =========================================================
# Input Form
# =========================================================

with st.form("tourism_prediction_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # Column 1
    # -----------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        occupation = st.selectbox(
            "Occupation",
            list(OCCUPATION_MAP.keys())
        )

        gender = st.selectbox(
            "Gender",
            list(GENDER_MAP.keys())
        )

        marital_status = st.selectbox(
            "Marital Status",
            list(MARITAL_STATUS_MAP.keys())
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )

    # -----------------------------------------------------
    # Column 2
    # -----------------------------------------------------

    with col2:

        type_of_contact = st.selectbox(
            "Type of Contact",
            list(TYPE_OF_CONTACT_MAP.keys())
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        duration_of_pitch = st.number_input(
            "Duration of Pitch (minutes)",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        number_of_children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=1.0
        )

    # -----------------------------------------------------
    # Column 3
    # -----------------------------------------------------

    with col3:

        number_of_followups = st.number_input(
            "Number of Follow-ups",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=1.0
        )

        number_of_trips = st.number_input(
            "Number of Trips",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=1.0
        )

        passport = st.selectbox(
            "Passport",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        own_car = st.selectbox(
            "Own Car",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Star",
            [3.0, 4.0, 5.0]
        )

    # =====================================================
    # Package / Pitch Information
    # =====================================================

    st.subheader("📦 Package & Pitch Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        product_pitched = st.selectbox(
            "Product Pitched",
            list(PRODUCT_PITCHED_MAP.keys())
        )

    with col2:

        pitch_satisfaction_score = st.selectbox(
            "Pitch Satisfaction Score",
            [1, 2, 3, 4, 5]
        )

    with col3:

        designation = st.selectbox(
            "Designation",
            list(DESIGNATION_MAP.keys())
        )

    # =====================================================
    # Prediction Button
    # =====================================================

    submitted = st.form_submit_button(
        "🔮 Predict Package Purchase",
        use_container_width=True
    )


# =========================================================
# Prediction
# =========================================================

if submitted:

    # =====================================================
    # Feature Engineering
    # =====================================================

    # AgeGroup
    #
    # IMPORTANT:
    # Make sure this mapping is identical to training.
    #
    # 0 = Young
    # 1 = Adult
    # 2 = MiddleAge
    # 3 = Senior
    # =====================================================

    if age <= 30:
        age_group = 0
    elif age <= 45:
        age_group = 1
    elif age <= 60:
        age_group = 2
    else:
        age_group = 3


    # =====================================================
    # IncomeCategory
    #
    # Based on the values visible in your training dataset:
    #
    # 1 = Low
    # 2 = Medium
    # 3 = High
    # 4 = VeryHigh
    #
    # Verify this against your actual training code.
    # =====================================================

    if monthly_income <= 25000:
        income_category = 1
    elif monthly_income <= 50000:
        income_category = 2
    elif monthly_income <= 100000:
        income_category = 3
    else:
        income_category = 4


    # =====================================================
    # Convert categorical values to training encodings
    # =====================================================

    type_of_contact_encoded = TYPE_OF_CONTACT_MAP[
        type_of_contact
    ]

    occupation_encoded = OCCUPATION_MAP[
        occupation
    ]

    gender_encoded = GENDER_MAP[
        gender
    ]

    product_pitched_encoded = PRODUCT_PITCHED_MAP[
        product_pitched
    ]

    marital_status_encoded = MARITAL_STATUS_MAP[
        marital_status
    ]

    designation_encoded = DESIGNATION_MAP[
        designation
    ]


    # =====================================================
    # Create Prediction DataFrame
    # =====================================================

    input_data = pd.DataFrame({

        "Age": [age],

        "TypeofContact": [
            type_of_contact_encoded
        ],

        "CityTier": [
            city_tier
        ],

        "DurationOfPitch": [
            duration_of_pitch
        ],

        "Occupation": [
            occupation_encoded
        ],

        "Gender": [
            gender_encoded
        ],

        "NumberOfPersonVisiting": [
            number_of_person_visiting
        ],

        "NumberOfFollowups": [
            number_of_followups
        ],

        "ProductPitched": [
            product_pitched_encoded
        ],

        "PreferredPropertyStar": [
            preferred_property_star
        ],

        "MaritalStatus": [
            marital_status_encoded
        ],

        "NumberOfTrips": [
            number_of_trips
        ],

        "Passport": [
            passport
        ],

        "PitchSatisfactionScore": [
            pitch_satisfaction_score
        ],

        "OwnCar": [
            own_car
        ],

        "NumberOfChildrenVisiting": [
            number_of_children_visiting
        ],

        "Designation": [
            designation_encoded
        ],

        "MonthlyIncome": [
            monthly_income
        ],

        "IncomeCategory": [
            income_category
        ],

        "AgeGroup": [
            age_group
        ]
    })


    # =====================================================
    # Force Exact Training Column Order
    # =====================================================

    input_data = input_data[FEATURE_COLUMNS]


    # =====================================================
    # Prediction
    # =====================================================

    try:

        prediction = model.predict(input_data)[0]

        # -------------------------------------------------
        # Probability
        # -------------------------------------------------

        purchase_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            # Handle both binary-class ordering cases safely
            if hasattr(model, "classes_"):

                classes = list(model.classes_)

                if 1 in classes:
                    purchase_probability = probabilities[
                        classes.index(1)
                    ]

            else:

                purchase_probability = probabilities[1]


        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        st.divider()

        st.subheader("🎯 Prediction Result")

        if prediction == 1:

            st.success(
                "🎉 The customer is likely to purchase "
                "the tourism package."
            )

        else:

            st.warning(
                "The customer is unlikely to purchase "
                "the tourism package."
            )


        # -------------------------------------------------
        # Probability
        # -------------------------------------------------

        if purchase_probability is not None:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Purchase Probability",
                    f"{purchase_probability:.2%}"
                )

            with col2:

                st.progress(
                    float(purchase_probability)
                )


        # -------------------------------------------------
        # Encoded Input Data
        # -------------------------------------------------

        with st.expander("🔍 View Model Input Data"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


        # -------------------------------------------------
        # Human-readable Input Data
        # -------------------------------------------------

        with st.expander("👤 View User Input"):

            display_data = pd.DataFrame({
                "Feature": [
                    "Age",
                    "Type of Contact",
                    "City Tier",
                    "Duration of Pitch",
                    "Occupation",
                    "Gender",
                    "Persons Visiting",
                    "Follow-ups",
                    "Product Pitched",
                    "Property Star",
                    "Marital Status",
                    "Number of Trips",
                    "Passport",
                    "Pitch Satisfaction",
                    "Own Car",
                    "Children Visiting",
                    "Designation",
                    "Monthly Income",
                    "Income Category",
                    "Age Group"
                ],

                "Value": [
                    age,
                    type_of_contact,
                    city_tier,
                    duration_of_pitch,
                    occupation,
                    gender,
                    number_of_person_visiting,
                    number_of_followups,
                    product_pitched,
                    preferred_property_star,
                    marital_status,
                    number_of_trips,
                    "Yes" if passport == 1 else "No",
                    pitch_satisfaction_score,
                    "Yes" if own_car == 1 else "No",
                    number_of_children_visiting,
                    designation,
                    monthly_income,
                    income_category,
                    age_group
                ]
            })

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )
