import streamlit as st
from crewai import Agent, Task, Crew, Process,  LLM  # Import LLM from crewai
import os
import fitz  # PyMuPDF for PDF handling
import tempfile
from dotenv import load_dotenv
import textwrap

# Load environment variables
load_dotenv()

# --- Helper Functions ---

def read_pdf(file):
    """Reads text from a PDF file."""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def wrap_text(text, width=80):
    """Wraps text to a specified width."""
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=width))
    return '\n'.join(wrapped_lines)


# --- CrewAI Agents and Tasks ---

def create_crew(job_description, resume_text):
    """Creates the CrewAI crew with agents and tasks."""

    # Configure the LLM using CrewAI's LLM class
    llm_config = LLM(
        model_name="gemini-1.5-flash-002",  # Use direct model name
        api_key=os.environ.get("GOOGLE_API_KEY"), # api_key needed for google gen ai
        temperature=0.7,
        #convert_system_message_to_human=True  # No longer needed with CrewAI LLM
    )

    # 1. Resume Analyzer Agent
    resume_analyzer = Agent(
        role='Resume Analysis Expert',
        goal='Thoroughly analyze the provided resume and identify its strengths and weaknesses.',
        backstory="""You are a seasoned resume analyst with years of experience
        in helping job seekers optimize their resumes.  You are adept at
        identifying key skills, experiences, and formatting issues.""",
        verbose=True,
        allow_delegation=False,
        llm=llm_config  # Use the CrewAI LLM configuration
    )

    # 2. Job Description Analyzer Agent
    job_analyzer = Agent(
        role='Job Description Expert',
        goal='Analyze the job description and extract key requirements, skills, and keywords.',
        backstory="""You are a highly skilled job description analyst.
        You excel at identifying the core requirements, desired skills, and
        important keywords from any job posting.""",
        verbose=True,
        allow_delegation=False,
        llm=llm_config  # Use the CrewAI LLM configuration
    )

    # 3. Resume Improvement Suggestor Agent
    improvement_suggestor = Agent(
        role='Resume Improvement Specialist',
        goal='Provide specific, actionable suggestions to improve the resume based on the job description.',
        backstory="""You are a master resume writer and career coach.  You
        are known for your ability to craft compelling resumes that highlight
        a candidate's strengths and align them perfectly with job requirements.
        You provide concrete, easy-to-implement suggestions.""",
        verbose=True,
        allow_delegation=False,
        llm=llm_config  # Use the CrewAI LLM configuration
    )

    # --- Tasks ---

    # Task 1: Analyze the Resume
    task_analyze_resume = Task(
        description=f"""Analyze the following resume content and identify key skills, experiences,
        and potential areas for improvement.  Focus on the overall structure, clarity,
        and impact of the resume. Output should be a structured report, not just raw thoughts.
        Resume:
        --------------
        {resume_text}
        --------------
        """,
        agent=resume_analyzer,
        expected_output="A structured report summarizing the resume's strengths, weaknesses, key skills, and areas for improvement."
    )

    # Task 2: Analyze the Job Description
    task_analyze_job_description = Task(
        description=f"""Analyze the following job description and extract the key requirements,
        desired skills, preferred qualifications, and any important keywords.
        Be specific and comprehensive in your analysis. Output a structured summary.
        Job Description:
        --------------
        {job_description}
        --------------
        """,
        agent=job_analyzer,
        expected_output="A structured summary of the job description, including key requirements, desired skills, qualifications, and important keywords."
    )

    # Task 3: Suggest Improvements
    task_suggest_improvements = Task(
        description=f"""Based on the analysis of the resume and the job description,
        provide specific and actionable suggestions to improve the resume. Address:
        1.  **Content:**  Suggest additions, deletions, or modifications to the resume content to better match the job requirements.
        2.  **Keywords:** Identify keywords from the job description that should be incorporated into the resume.
        3.  **Formatting:**  Suggest any formatting changes to improve readability and impact.
        4.  **Overall Strategy:**  Provide an overall strategy for tailoring the resume to the specific job.

        The resume analysis is: {task_analyze_resume.output}
        The job description analysis is: {task_analyze_job_description.output}
        """,
        agent=improvement_suggestor,
        expected_output="A list of specific, actionable suggestions for improving the resume, covering content, keywords, formatting, and overall strategy, tailored to the job description."
    )
    # --- Crew ---
    crew = Crew(
        agents=[resume_analyzer, job_analyzer, improvement_suggestor],
        tasks=[task_analyze_resume, task_analyze_job_description, task_suggest_improvements],
        verbose=True,
        process=Process.sequential
    )

    return crew

# --- Streamlit App ---

st.title("Resume Tailoring Assistant")

# Input: Job Description
job_description = st.text_area("Paste the Job Description Here:", height=200)

# Input: Resume (PDF Upload)
resume_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")

if st.button("Tailor Resume"):
    if job_description and resume_file:
        with st.spinner("Analyzing and generating suggestions..."):
            resume_text = read_pdf(resume_file)
            if resume_text:
                # Create and run the Crew
                crew = create_crew(job_description, resume_text)
                result = crew.kickoff()
                st.subheader("Suggested Improvements:")
                st.write(wrap_text(result))
            else:
                st.error("Failed to read the resume content.")
    else:
        st.warning("Please provide both the job description and your resume.")