from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_artifacts" / "questionnaire_catboost.cbm"


RAW_FEATURES = [
    "AGE_AT_VISIT",
    "ANYFAMPD",
    "ARMLGSHK",
    "MVSLOW",
    "FTSTUCK",
    "SHUFFLE",
    "POORBAL",
    "WRTSMLR",
    "LSSXPRSS",
    "VOICSFTR",
    "NP1COG",
]


CATEGORICAL_FEATURES = [
    "ANYFAMPD",
    "ARMLGSHK",
    "MVSLOW",
    "FTSTUCK",
    "SHUFFLE",
    "POORBAL",
    "WRTSMLR",
    "LSSXPRSS",
    "VOICSFTR",
]


MOTOR_FEATURES = [
    "ARMLGSHK",
    "MVSLOW",
    "FTSTUCK",
    "SHUFFLE",
    "POORBAL",
    "WRTSMLR",
    "LSSXPRSS",
]


UNCERTAINTY_FEATURES = [
    "ARMLGSHK",
    "MVSLOW",
    "FTSTUCK",
    "SHUFFLE",
    "POORBAL",
    "WRTSMLR",
    "LSSXPRSS",
    "VOICSFTR",
]


MODEL_AGE_MIN = 29.3
MODEL_AGE_MAX = 86.7

LOW_MODEL_THRESHOLD = 0.45
HIGH_MODEL_THRESHOLD = 0.79


_model = CatBoostClassifier()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Questionnaire model not found at: {MODEL_PATH}"
    )

_model.load_model(str(MODEL_PATH))


def validate_inputs(raw_values):
    missing = [
        feature
        for feature in RAW_FEATURES
        if feature not in raw_values
    ]

    if missing:
        raise ValueError(
            f"Missing questionnaire inputs: {', '.join(missing)}"
        )

    try:
        age = float(raw_values["AGE_AT_VISIT"])
    except (TypeError, ValueError):
        raise ValueError("Age must be numeric.")

    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100.")

    family_history = int(raw_values["ANYFAMPD"])

    if family_history not in [0, 1]:
        raise ValueError(
            "Family history must be 0 or 1."
        )

    symptom_features = [
        "ARMLGSHK",
        "MVSLOW",
        "FTSTUCK",
        "SHUFFLE",
        "POORBAL",
        "WRTSMLR",
        "LSSXPRSS",
        "VOICSFTR",
    ]

    for feature in symptom_features:
        value = int(raw_values[feature])

        if value not in [0, 1, 2]:
            raise ValueError(
                f"{feature} must be 0, 1 or 2."
            )

    cognition = int(raw_values["NP1COG"])

    if cognition not in [0, 1, 2, 3, 4]:
        raise ValueError(
            "Cognition score must be between 0 and 4."
        )


def normalize_inputs(raw_values):
    return {
        "AGE_AT_VISIT": float(
            raw_values["AGE_AT_VISIT"]
        ),
        "ANYFAMPD": int(
            raw_values["ANYFAMPD"]
        ),
        "ARMLGSHK": int(
            raw_values["ARMLGSHK"]
        ),
        "MVSLOW": int(
            raw_values["MVSLOW"]
        ),
        "FTSTUCK": int(
            raw_values["FTSTUCK"]
        ),
        "SHUFFLE": int(
            raw_values["SHUFFLE"]
        ),
        "POORBAL": int(
            raw_values["POORBAL"]
        ),
        "WRTSMLR": int(
            raw_values["WRTSMLR"]
        ),
        "LSSXPRSS": int(
            raw_values["LSSXPRSS"]
        ),
        "VOICSFTR": int(
            raw_values["VOICSFTR"]
        ),
        "NP1COG": int(
            raw_values["NP1COG"]
        ),
    }


def add_engineered_features(data):
    engineered = data.copy()

    motor_binary = engineered[
        MOTOR_FEATURES
    ].replace({
        0: 0.0,
        1: 1.0,
        2: np.nan,
    })

    engineered["motor_score"] = (
        motor_binary.mean(axis=1)
    )

    engineered["motor_uncertainty"] = (
        engineered[MOTOR_FEATURES]
        .eq(2)
        .mean(axis=1)
    )

    voice_score = engineered[
        "VOICSFTR"
    ].replace({
        0: 0.0,
        1: 1.0,
        2: np.nan,
    })

    cognition_score = (
        engineered["NP1COG"] / 4.0
    )

    engineered[
        "speech_cognition_score"
    ] = pd.concat(
        [
            voice_score,
            cognition_score,
        ],
        axis=1,
    ).mean(axis=1)

    return engineered


def prepare_model_input(raw_values):
    validate_inputs(raw_values)

    normalized = normalize_inputs(
        raw_values
    )

    raw_df = pd.DataFrame(
        [normalized],
        columns=RAW_FEATURES,
    )

    model_input = add_engineered_features(
        raw_df
    )

    for column in CATEGORICAL_FEATURES:
        model_input[column] = (
            model_input[column]
            .astype(str)
        )

    return (
        raw_df.iloc[0],
        model_input,
    )


def calculate_response_reliability(row):
    uncertain_count = sum(
        int(row[column]) == 2
        for column in UNCERTAINTY_FEATURES
    )

    if uncertain_count <= 2:
        reliability_score = 1.00
        reliability_level = "High"

    elif uncertain_count == 3:
        reliability_score = 0.75
        reliability_level = "Moderate"

    elif uncertain_count <= 5:
        reliability_score = 0.50
        reliability_level = "Low"

    else:
        reliability_score = 0.25
        reliability_level = "Low"

    return (
        uncertain_count,
        reliability_score,
        reliability_level,
    )


def calculate_motor_pattern(row):
    positive_motor_count = sum(
        int(row[column]) == 1
        for column in MOTOR_FEATURES
    )

    motor_score = (
        positive_motor_count
        / len(MOTOR_FEATURES)
    )

    if positive_motor_count <= 2:
        motor_pattern = "Low"

    elif positive_motor_count == 3:
        motor_pattern = "Borderline"

    else:
        motor_pattern = "Strong"

    return (
        positive_motor_count,
        motor_score,
        motor_pattern,
    )


def calculate_secondary_support(row):
    voice_positive = (
        int(row["VOICSFTR"]) == 1
    )

    cognition_value = int(
        row["NP1COG"]
    )

    cognition_positive = (
        cognition_value >= 2
    )

    family_history_positive = (
        int(row["ANYFAMPD"]) == 1
    )

    secondary_count = (
        int(voice_positive)
        + int(cognition_positive)
    )

    voice_score = (
        1.0
        if voice_positive
        else 0.0
    )

    cognition_score = (
        cognition_value / 4.0
    )

    speech_cognition_score = (
        voice_score
        + cognition_score
    ) / 2.0

    if secondary_count == 2:
        support_level = "Strong"

    elif secondary_count == 1:
        support_level = "Moderate"

    else:
        support_level = "Low"

    return {
        "voice_positive":
            voice_positive,

        "cognition_positive":
            cognition_positive,

        "family_history_positive":
            family_history_positive,

        "secondary_count":
            secondary_count,

        "speech_cognition_score":
            speech_cognition_score,

        "support_level":
            support_level,
    }


def get_model_zone(model_score):
    if model_score < LOW_MODEL_THRESHOLD:
        return "Low"

    if model_score < HIGH_MODEL_THRESHOLD:
        return "Moderate"

    return "Elevated"


def determine_final_prediction(
    age,
    motor_count,
    secondary_info,
    uncertain_count,
):
    secondary_count = (
        secondary_info[
            "secondary_count"
        ]
    )

    flags = []

    if motor_count <= 2:
        if secondary_count == 2:
            final_prediction = "Borderline"
        else:
            final_prediction = "Healthy"

    elif motor_count == 3:
        if secondary_count == 2:
            final_prediction = (
                "Elevated Parkinsonian Risk"
            )
        else:
            final_prediction = "Borderline"

    else:
        final_prediction = (
            "Elevated Parkinsonian Risk"
        )

    if uncertain_count >= 4:
        final_prediction = "Borderline"

        flags.append(
            "Four or more uncertain answers "
            "make the questionnaire result "
            "inconclusive."
        )

    age_in_model_range = (
        MODEL_AGE_MIN
        <= age
        <= MODEL_AGE_MAX
    )

    if not age_in_model_range:
        flags.append(
            "Age is outside the model-development "
            "range; ML support is less reliable."
        )

        if (
            final_prediction
            == "Elevated Parkinsonian Risk"
        ):
            if (
                motor_count < 4
                or secondary_count < 1
            ):
                final_prediction = (
                    "Borderline"
                )

    elif age < 40:
        flags.append(
            "Younger-age participant; "
            "interpret questionnaire evidence "
            "with additional caution."
        )

    return (
        final_prediction,
        age_in_model_range,
        flags,
    )


def calculate_module_confidence(
    age,
    model_score,
    final_prediction,
    motor_count,
    secondary_count,
    uncertain_count,
    reliability_score,
):
    model_zone = get_model_zone(
        model_score
    )

    if final_prediction == "Healthy":
        symptom_strength = (
            1.0
            if motor_count <= 1
            else 0.80
        )

    elif final_prediction == "Borderline":
        symptom_strength = 0.65

    else:
        symptom_strength = (
            1.0
            if motor_count >= 4
            else 0.90
        )

    if final_prediction == "Healthy":
        if model_zone == "Low":
            model_agreement = 1.0
        elif model_zone == "Moderate":
            model_agreement = 0.65
        else:
            model_agreement = 0.35

    elif final_prediction == "Borderline":
        if model_zone == "Moderate":
            model_agreement = 0.75
        else:
            model_agreement = 0.55

    else:
        if model_zone == "Elevated":
            model_agreement = 1.0
        elif model_zone == "Moderate":
            model_agreement = 0.70
        else:
            model_agreement = 0.40

    confidence = (
        0.50 * symptom_strength
        + 0.30 * reliability_score
        + 0.20 * model_agreement
    )

    if (
        age < MODEL_AGE_MIN
        or age > MODEL_AGE_MAX
    ):
        confidence *= 0.70

    elif age < 40:
        confidence *= 0.90

    if uncertain_count >= 4:
        confidence *= 0.75

    confidence = float(
        np.clip(
            confidence,
            0,
            1,
        )
    )

    if confidence >= 0.70:
        confidence_level = "High"

    elif confidence >= 0.45:
        confidence_level = "Moderate"

    else:
        confidence_level = "Low"

    return (
        confidence,
        confidence_level,
    )


def patient_interpretation(
    final_prediction
):
    if final_prediction == "Healthy":
        return (
            "Low Parkinsonian screening concern: "
            "your questionnaire responses show "
            "limited confirmed evidence of a "
            "Parkinsonian symptom pattern. "
            "This result will be considered together "
            "with the other screening modules."
        )

    if final_prediction == "Borderline":
        return (
            "Borderline Parkinsonian risk: some "
            "questionnaire findings are associated "
            "with Parkinsonian symptoms, but the "
            "overall pattern is not strong or "
            "consistent enough for an elevated-risk "
            "classification. The remaining screening "
            "modules will provide additional evidence "
            "for the final assessment."
        )

    return (
        "Elevated Parkinsonian risk: multiple "
        "questionnaire findings form a stronger "
        "Parkinsonian symptom pattern. This result "
        "does not confirm Parkinson's disease and "
        "will be combined with the other screening "
        "modules for the final assessment."
    )


def questionnaire_assessment(
    raw_row,
    model_score,
):
    age = float(
        raw_row["AGE_AT_VISIT"]
    )

    (
        uncertain_count,
        reliability_score,
        reliability_level,
    ) = calculate_response_reliability(
        raw_row
    )

    (
        motor_count,
        motor_score,
        motor_pattern,
    ) = calculate_motor_pattern(
        raw_row
    )

    secondary_info = (
        calculate_secondary_support(
            raw_row
        )
    )

    (
        final_prediction,
        age_in_model_range,
        flags,
    ) = determine_final_prediction(
        age,
        motor_count,
        secondary_info,
        uncertain_count,
    )

    model_zone = get_model_zone(
        model_score
    )

    if (
        final_prediction == "Healthy"
        and model_zone == "Elevated"
    ):
        flags.append(
            "The internal ML signal is elevated "
            "but is not supported by the confirmed "
            "symptom pattern."
        )

    elif (
        final_prediction
        == "Elevated Parkinsonian Risk"
        and model_zone == "Low"
    ):
        flags.append(
            "The strong symptom pattern is not "
            "supported by the internal ML signal."
        )

    (
        confidence_score,
        confidence_level,
    ) = calculate_module_confidence(
        age,
        model_score,
        final_prediction,
        motor_count,
        secondary_info[
            "secondary_count"
        ],
        uncertain_count,
        reliability_score,
    )

    return {
        "final_prediction":
            final_prediction,

        "module_confidence":
            round(
                confidence_score,
                4,
            ),

        "confidence_level":
            confidence_level,

        "motor_score":
            round(
                motor_score,
                4,
            ),

        "motor_pattern":
            motor_pattern,

        "positive_motor_symptoms":
            motor_count,

        "speech_cognition_score":
            round(
                secondary_info[
                    "speech_cognition_score"
                ],
                4,
            ),

        "secondary_support_level":
            secondary_info[
                "support_level"
            ],

        "voice_positive":
            secondary_info[
                "voice_positive"
            ],

        "cognition_positive":
            secondary_info[
                "cognition_positive"
            ],

        "family_history_positive":
            secondary_info[
                "family_history_positive"
            ],

        "response_reliability":
            round(
                reliability_score,
                4,
            ),

        "reliability_level":
            reliability_level,

        "uncertain_answers":
            uncertain_count,

        "age_in_model_range":
            age_in_model_range,

        "internal_model_score":
            round(
                model_score,
                4,
            ),

        "internal_model_zone":
            model_zone,

        "safety_flags":
            flags,

        "patient_interpretation":
            patient_interpretation(
                final_prediction
            ),

        "requires_fusion_review": (
            final_prediction == "Borderline"
            or confidence_level == "Low"
            or not age_in_model_range
            or age < 40
        ),
    }


def predict_questionnaire(
    raw_values
):
    raw_row, model_input = (
        prepare_model_input(
            raw_values
        )
    )

    model_score = float(
        _model.predict_proba(
            model_input
        )[0, 1]
    )

    return questionnaire_assessment(
        raw_row,
        model_score,
    )