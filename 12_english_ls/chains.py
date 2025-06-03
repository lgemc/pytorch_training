from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from templates import start_prompt, assessment_prompt, explanation_prompt, lesson_planner_prompt, evaluation_prompt

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0.7)  # you can adjust temperature

# 1. Start Chain
start_chain = LLMChain(
    llm=llm,
    prompt=start_prompt,
    output_key="start_output",  # The raw JSON string
)

# 2. Assessment Chain
assessment_chain = LLMChain(
    llm=llm,
    prompt=assessment_prompt,
    output_key="assessment_output",
    memory=ConversationBufferMemory()
)

# 3a. Explanation Chain (for Tense/Modifier path)
explanation_chain = LLMChain(
    llm=llm,
    prompt=explanation_prompt,
    output_key="explanation_output",
    memory=ConversationBufferMemory()
)

# 3b. Lesson Planner Chain (for Tense/Topic path)
lesson_planner_chain = LLMChain(
    llm=llm,
    prompt=lesson_planner_prompt,
    output_key="lesson_plan_output",
    memory=ConversationBufferMemory()
)

# 4. Evaluation Chain
evaluation_chain = LLMChain(
    llm=llm,
    prompt=evaluation_prompt,
    output_key="evaluation_output",
    memory=ConversationBufferMemory()
)
