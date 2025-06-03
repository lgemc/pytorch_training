from langchain.prompts import PromptTemplate

start_prompt = PromptTemplate(
    input_variables=["tense", "modifier", "topic"],
    template="""
You are an English instructor. The student is learning about:
- Tense: {tense}
- Modifier: {modifier}
- Topic: {topic}

Modifier can be present, past, future, or perfect.

You need to explain the session 

Return a JSON with fields:
{{
  "explanation": a short explanation about the topic the student is learning,
  "examples": ["example 1", "example 2", "example 3"]
}}
""",
)


assessment_prompt = PromptTemplate(
    input_variables=["tense", "modifier", "topic"],
    template="""
You are an English instructor. A student wants an assessment based on:
- Tense: {tense}
- Modifier: {modifier}
- Topic: {topic}

Create a short diagnostic:
1. Two or three multiple-choice or fill-in-the-blank questions to test the student's current grasp.
2. Provide the correct answers and a brief difficulty rating (easy/medium/hard).
3. Output a JSON with fields:
   {{
     "questions": [{{"question": ..., "options": [...], "answer": ...}}, ...],
     "assessment_score_estimate": a number from 0 to 100,
     "notes": a short rationale for the score.
   }}
"""
)

explanation_prompt = PromptTemplate(
    input_variables=["tense", "modifier", "assessment_score_estimate", "notes"],
    template="""
You are an expert English teacher. The student scored {assessment_score_estimate} on the assessment for the {tense} tense with {modifier} form.
1. Provide a concise grammar explanation of the {tense} tense with {modifier} (e.g., when to use, structure).
2. Offer three illustrative example sentences.
3. Suggest two short practice exercises.

Return a JSON with:
{{
  "explanation": "...",
  "examples": ["...", "...", "..."],
  "practice_exercises": ["...", "..."]
}}
"""
)

lesson_planner_prompt = PromptTemplate(
    input_variables=["tense", "topic", "assessment_score_estimate", "notes"],
    template="""
You are an English curriculum designer. The student scored {assessment_score_estimate} on the assessment for {tense} tense in the context of {topic}. Design a 15-minute lesson including:
1. A brief introduction to the topic vocabulary.
2. Three guided practice activities (e.g., role-play prompts, fill-in-the-blank dialogues).
3. One short reading passage or dialogue illustrating {tense} in {topic}.
4. Two follow-up homework prompts.

Return a JSON with fields:
{{
  "vocabulary_list": ["...", "...", "..."],
  "guided_activities": ["...", "...", "..."],
  "reading_passage": "...",
  "homework_prompts": ["...", "..."]
}}
"""
)

evaluation_prompt = PromptTemplate(
    input_variables=["tense", "modifier", "topic", "lesson_content"],
    template="""
You are a teacher evaluating a student’s understanding. Based on the lesson content:
{lesson_content}

Generate:
1. Two quiz questions (with answers) that assess mastery of the lesson objectives.
2. A short rubric describing how to score the student (e.g., 0–2 points per question).

Return a JSON with:
{{
  "evaluation_questions": [{{"question": ..., "answer": ...}}, ...],
  "rubric": "..."
}}
"""
)