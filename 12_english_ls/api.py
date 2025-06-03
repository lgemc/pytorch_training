import json
from langchain.chains import SequentialChain

from chains import (
    start_chain,
    assessment_chain,
    explanation_chain,
)

# Helper: Parse JSON string into dict safely
def _parse_json_output(llm_output: str) -> dict:
    try:
        return json.loads(llm_output.strip())
    except json.JSONDecodeError:
        # Fallback: use a simple heuristic or wrap in {}
        return {}


# Build a function that wraps the SequentialChain execution:
def run_tense_modifier_pipeline(tense: str, modifier: str):
    # 1. Start chain
    start_response = start_chain.run(
        {"tense": tense, "modifier": modifier, "topic": ""}  # topic blank
    )
    start_data = _parse_json_output(start_response)
    session_type = start_data.get("session_type")

    # 2. Assessment chain
    assessment_response = assessment_chain.run({
        "session_type": session_type,
        "tense": tense,
        "modifier": modifier,
        "topic": ""
    })
    assessment_data = _parse_json_output(assessment_response)
    score = assessment_data.get("assessment_score_estimate", 0)
    notes = assessment_data.get("notes", "")

    # 3. Explanation chain
    explanation_response = explanation_chain.run({
        "tense": tense,
        "modifier": modifier,
        "assessment_score_estimate": score,
        "notes": notes
    })
    explanation_data = _parse_json_output(explanation_response)

    return {
        "start": start_data,
        "assessment": assessment_data,
        "explanation": explanation_data
    }