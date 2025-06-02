import streamlit as st
import openai
import os
from dotenv import load_dotenv
import json
import time

# Load environment variables
def load_api_key():
    # Try to load from different .env files
    env_paths = [
        '.env',
        'lm_fine_tuner/.env',
        '../lm_fine_tuner/.env',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                return api_key
    
    return None

# Set up OpenAI client
api_key = load_api_key()
if api_key:
    client = openai.OpenAI(api_key=api_key)
    st.sidebar.success("API Key loaded successfully!")
else:
    st.sidebar.error("API Key not found. Please check your .env file.")
    client = None

# App title and description
st.title('Empathetic Response Generator')
st.markdown("""
    This application uses a fine-tuned GPT model to generate empathetic responses 
    for various emotional situations. You can test the model, start new fine-tuning jobs, 
    or check the status of existing jobs.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Test Model", "Fine-Tuning", "Job Status"])

# Tab 1: Test Model
with tab1:
    st.header("Test Your Fine-Tuned Model")
    
    # Model selection
    model_options = [
        "ft:gpt-3.5-turbo-0125:valis::BYfKr10K",  # Your fine-tuned model
        "gpt-3.5-turbo",  # Base model for comparison
    ]
    selected_model = st.selectbox("Select Model", model_options)
    
    # System message
    system_message = st.text_area(
        "System Message", 
        "You are a compassionate emotional support assistant. Your role is to provide empathetic and supportive responses to people experiencing emotional distress.",
        height=100
    )
    
    # User prompt
    prompt = st.text_area("Enter your message", height=100)
    
    # Temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Generate response
    if st.button("Generate Response"):
        if not prompt:
            st.error("Please enter a message.")
        elif not client:
            st.error("API Key not configured properly.")
        else:
            with st.spinner("Generating response..."):
                try:
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=150
                    )
                    
                    st.markdown("### Response:")
                    st.write(response.choices[0].message.content)
                    
                    # Save the response to a file for reference
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    with open(f"model_output/response_{timestamp}.json", "w") as f:
                        json.dump({
                            "model": selected_model,
                            "prompt": prompt,
                            "response": response.choices[0].message.content,
                            "temperature": temperature
                        }, f, indent=2)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Tab 2: Fine-Tuning
with tab2:
    st.header("Start a New Fine-Tuning Job")
    
    # Upload dataset
    dataset = st.file_uploader("Upload Dataset (JSONL Format)", type="jsonl")
    
    # Base model selection
    base_model = st.selectbox(
        "Base Model", 
        ["gpt-3.5-turbo", "gpt-4"]
    )
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=3)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4)
    with col3:
        learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.1, value=0.00005, format="%.5f")
    
    # Start fine-tuning button
    if st.button("Start Fine-Tuning"):
        if dataset is None:
            st.error("Please upload a dataset.")
        elif not client:
            st.error("API Key not configured properly.")
        else:
            with st.spinner("Starting fine-tuning job..."):
                try:
                    # Save the uploaded file
                    with open("uploaded_dataset.jsonl", "wb") as f:
                        f.write(dataset.getvalue())
                    
                    # Upload the file to OpenAI
                    with open("uploaded_dataset.jsonl", "rb") as f:
                        upload_response = client.files.create(
                            file=f,
                            purpose="fine-tune"
                        )
                    
                    file_id = upload_response.id
                    st.success(f"File uploaded successfully. File ID: {file_id}")
                    
                    # Start fine-tuning job
                    response = client.fine_tuning.jobs.create(
                        training_file=file_id,
                        model=base_model,
                        hyperparameters={
                            "n_epochs": epochs,
                            "batch_size": batch_size,
                            "learning_rate_multiplier": learning_rate
                        }
                    )
                    
                    # Save job information
                    jobs_file = "fine_tuning_jobs.json"
                    try:
                        if os.path.exists(jobs_file):
                            with open(jobs_file, "r") as f:
                                jobs_data = json.load(f)
                        else:
                            jobs_data = {"jobs": []}
                        
                        # Add new job
                        jobs_data["jobs"].append({
                            "id": response.id,
                            "status": response.status,
                            "created_at": response.created_at,
                            "model": base_model,
                            "training_file": file_id
                        })
                        
                        # Save updated jobs data
                        with open(jobs_file, "w") as f:
                            json.dump(jobs_data, f, indent=2)
                    except Exception as e:
                        st.warning(f"Could not save job information: {str(e)}")
                    
                    st.success(f"Fine-tuning job started! Job ID: {response.id}")
                    st.info("You can check the status of your job in the 'Job Status' tab.")
                except Exception as e:
                    st.error(f"Error starting fine-tuning: {str(e)}")

# Tab 3: Job Status
with tab3:
    st.header("Check Fine-Tuning Job Status")
    
    # Create a jobs file if it doesn't exist
    jobs_file = "fine_tuning_jobs.json"
    if not os.path.exists(jobs_file):
        with open(jobs_file, "w") as f:
            json.dump({"jobs": []}, f)
    
    # Load existing jobs
    try:
        with open(jobs_file, "r") as f:
            jobs_data = json.load(f)
    except Exception:
        jobs_data = {"jobs": []}
    
    # Display saved jobs
    if jobs_data["jobs"]:
        st.subheader("Your Fine-Tuning Jobs")
        job_options = [f"{job['id']} - {job['status']} - {job.get('model_id', 'N/A')}" for job in jobs_data["jobs"]]
        job_options.insert(0, "Select a job")
        
        selected_job_option = st.selectbox("Select from your jobs", job_options)
        
        if selected_job_option != "Select a job":
            job_id = selected_job_option.split(" - ")[0]
        else:
            job_id = ""
    else:
        st.info("You haven't submitted any fine-tuning jobs yet.")
        job_id = ""
    
    # Manual job ID input
    manual_job_id = st.text_input("Or enter a Job ID manually")
    if manual_job_id:
        job_id = manual_job_id
    
    # Check status button
    if st.button("Check Status"):
        if not job_id:
            st.error("Please select or enter a Job ID.")
        elif not client:
            st.error("API Key not configured properly.")
        else:
            with st.spinner("Checking job status..."):
                try:
                    response = client.fine_tuning.jobs.retrieve(job_id)
                    
                    # Display job information
                    st.markdown("### Job Information:")
                    st.write(f"**Job ID:** {response.id}")
                    st.write(f"**Status:** {response.status}")
                    st.write(f"**Created at:** {response.created_at}")
                    
                    if hasattr(response, 'finished_at') and response.finished_at:
                        st.write(f"**Finished at:** {response.finished_at}")
                    
                    if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
                        st.write(f"**Fine-tuned model:** {response.fine_tuned_model}")
                        st.success("Fine-tuning completed successfully!")
                    
                    # Update job in the list if it exists
                    job_exists = False
                    for job in jobs_data["jobs"]:
                        if job["id"] == response.id:
                            job["status"] = response.status
                            if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
                                job["model_id"] = response.fine_tuned_model
                            job_exists = True
                            break
                    
                    # Add job to the list if it doesn't exist
                    if not job_exists:
                        new_job = {
                            "id": response.id,
                            "status": response.status,
                            "created_at": response.created_at
                        }
                        if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
                            new_job["model_id"] = response.fine_tuned_model
                        jobs_data["jobs"].append(new_job)
                    
                    # Save updated jobs data
                    with open(jobs_file, "w") as f:
                        json.dump(jobs_data, f, indent=2)
                    
                    # Add a button to refresh the page
                    if st.button("Refresh Job List"):
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error checking job status: {str(e)}")

# Sidebar information
st.sidebar.header("Information")
st.sidebar.markdown("""
    ### About this app
    This application allows you to:
    - Test the fine-tuned empathy model
    - Start new fine-tuning jobs
    - Check the status of existing jobs
    
    ### Current fine-tuned model
    - Model ID: ft:gpt-3.5-turbo-0125:valis::BYfKr10K
    - Base model: gpt-3.5-turbo
    - Training data: empathy_openai_format.jsonl
""")

# Create necessary directories
os.makedirs("model_output", exist_ok=True)

