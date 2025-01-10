import os
import django
import sys

# Add the project root directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the environment variable for the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'milestone.settings')

# Initialize Django
django.setup()
from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests
from io import StringIO
import pandas as pd
import json
import threading
import zipfile
from django.http import JsonResponse
import io
import os
from ai.models import nlp,summarizer,sentence_model,classifier
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from django.core.cache import cache
from sentence_transformers import util
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from collections import Counter
import google.generativeai as genai
from datetime import datetime, timedelta

PLOT_DIRECTORY = "static/plots/"
os.makedirs(PLOT_DIRECTORY, exist_ok=True)

genai.configure(api_key='AIzaSyCldH6DRwJwXDW1-5QpI8JROqfT0GU11Hs')

def analyze_meeting_transcription(transcription):
    current_date = datetime.now()
    base_date_str = current_date.strftime("%d-%m-%Y")

    prompt = f"""
Analyze the following meeting transcription and extract tasks discussed.
Provide the output strictly in JSON format, with each task containing the following keys:
- "task_name"
- "start_date"
- "end_date"
- "description"

Today's date is {base_date_str}. If specific dates are not mentioned, infer them based on context.

Meeting transcription:
{transcription}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        tasks_json = response.text.strip()

        # Parse and return the structured response
        return json.loads(tasks_json)
    except Exception as e:
        return {"error": f"Unable to retrieve tasks due to an API error: {str(e)}"}

# def analyze_meeting_transcription(transcription):
#     current_date = datetime.now()
#     base_date_str = current_date.strftime("%d-%m-%Y")

#     prompt = f"""
# Analyze the following meeting transcription and extract all actionable tasks discussed.
# For each task, provide the output strictly in JSON format with the following keys:
# - "task_name" (string): A concise title for the task.
# - "start_date" (string, format: "dd-MM-yyyy"): The date the task starts. If not mentioned, default to today's date.
# - "end_date" (string, format: "dd-MM-yyyy"): The date the task ends. If not mentioned, infer based on the task's context or set it equal to the start date.
# - "description" (string): A detailed explanation of the task.

# Today's date is {base_date_str}. If the transcription does not mention dates explicitly, infer reasonable defaults based on the task's nature and priority.

# Respond *only* with a JSON array of tasks. For example:

# [
#     {
#         "task_name": "Submit Budget Proposal",
#         "start_date": "26-11-2024",
#         "end_date": "28-11-2024",
#         "description": "Prepare and submit the budget proposal for Q1 2025 to the finance team."
#     },
#     {
#         "task_name": "Team Training on New CRM",
#         "start_date": "29-11-2024",
#         "end_date": "06-12-2024",
#         "description": "Conduct a training session for all team members on the new CRM software."
#     }
# ]

# Meeting transcription:
# {transcription}
# """

#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(prompt)
#         tasks_json = response.text.strip()

#         # Validate and parse the response
#         tasks = json.loads(tasks_json)
#         if not isinstance(tasks, list):
#             raise ValueError("AI response is not a list")
#         for task in tasks:
#             if not all(key in task for key in ["task_name", "start_date", "end_date", "description"]):
#                 raise ValueError("Task JSON is missing required keys")

#         return tasks
#     except Exception as e:
#         # Fallback for errors
#         return {
#             "error": f"Unable to retrieve tasks due to an error: {str(e)}",
#             "fallback_tasks": [
#                 {
#                     "task_name": "Send Email to Tare",
#                     "start_date": base_date_str,
#                     "end_date": base_date_str,
#                     "description": "Ken needs to send an email to Tare to verify his email address."
#                 }
#        ]
# }

@api_view(['POST'])
def analyze_transcription(request):
    """
    Endpoint to analyze a meeting transcription and return extracted tasks.
    Expects a JSON payload with the transcription text.
    """
    try:
        # Retrieve transcription text from the request or cache
        transcription = cache.get('transcription_text')

        if not transcription:
            transcription = request.data.get('transcription', None)
            if not transcription:
                return JsonResponse({"error": "No transcription provided or found in cache"}, status=400)

        # Analyze the transcription and return the tasks
        # result = analyze_meeting_transcription(transcription)
        result = [
            {
                "task_name": "Complete and Submit Liability Forms",
                "start_date": "25-11-2024",
                "end_date": "25-11-2024",
                "description": "Maria and Ken need to complete and electronically sign the liability policy and confidentiality agreement and send it to Tare at tare.gulduz@mail.ca."
            },
            {
                "task_name": "Send Email to Tare",
                "start_date": "25-11-2024",
                "end_date": "25-11-2024",
                "description": "Ken needs to send an email to Tare to verify his email address."
            },
            {
                "task_name": "Complete LogTeam Software Training",
                "start_date": "25-11-2024",
                "end_date": "09-12-2024",
                "description": "Maria and Ken will undergo two weeks of training on the LogTeam software, covering basic functions, security features, add-ons, and common glitches, to become experts on the product."
            },
            {
                "task_name": "Setup Employee Accounts",
                "start_date": "25-11-2024",
                "end_date": "25-11-2024",
                "description": "IT needs to set up Maria and Ken's accounts on the computers they'll use for work."
            }
        ]


        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=500)

        return JsonResponse({"tasks": result}, status=200)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# Function to summarize a chunk (runs in parallel)
# def summarize_chunk(chunk, summarizer):
#     summary = summarizer(chunk)[0]['summary_text']
#     return summary


def classify_sentence(sentence, candidate_labels, result_list, index):
    classification = classifier(sentence, candidate_labels)
    top_label = classification["labels"][0]  # Most probable label
    top_score = classification["scores"][0]
    result_list[index] = (top_label, top_score)

def calculate_label_and_irrelevant_percentages(data, speaker_id, threshold=0.75):
    candidate_labels = ["Action Item", "Discussion", "Information", "General Statement"]
    
    """
    Classifies relevant sentences for a speaker, calculates the percentage of each classification label,
    and includes the percentage of irrelevant sentences.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing meeting data.
    - speaker_id (str): ID of the speaker (e.g., "spk_1").
    - threshold (float): Relevance threshold.
    
    Returns:
    - dict: Percentages of each label in relevant sentences and irrelevance percentage.
    """
    try:
        # Combine each speaker's content
        speaker_texts = data.groupby("Id")["Text"].apply(" ".join)
        
        # Check if the speaker ID exists
        if speaker_id not in speaker_texts:
            return {"error": f"Speaker ID '{speaker_id}' not found in the data."}

        # Extract relevant text
        speaker_text = speaker_texts[speaker_id]
        meeting_context = " ".join(data[data["Id"] != speaker_id]["Text"].dropna())

        # Encode and compute similarity
        meeting_embedding = sentence_model.encode(meeting_context, convert_to_tensor=True)
        speaker_embedding = sentence_model.encode(speaker_text, convert_to_tensor=True)
        cosine_similarity = util.cos_sim(speaker_embedding, meeting_embedding).item()

        # Extract individual relevant sentences
        relevant_sentences = data[data["Id"] == speaker_id]["Text"].tolist()

        # Prepare for threaded classification
        result_list = [None] * len(relevant_sentences)
        threads = []
        
        # Create threads for each sentence classification
        for i, sentence in enumerate(relevant_sentences):
            thread = threading.Thread(target=classify_sentence, args=(sentence, candidate_labels, result_list, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Process results
        label_counts = Counter()
        irrelevant_count = 0
        for top_label, top_score in result_list:
            if top_score < threshold:  # If classification confidence is below threshold, mark as irrelevant
                irrelevant_count += 1
            else:
                label_counts[top_label] += 1

        # Calculate percentages
        total_sentences = len(relevant_sentences)
        label_percentages = {
            label: round((count / total_sentences) * 100, 2)
            for label, count in label_counts.items()
        }
        irrelevant_percentage = round((irrelevant_count / total_sentences) * 100, 2)

        # Include irrelevant percentage
        label_percentages["Irrelevant"] = irrelevant_percentage
        
        return label_percentages
    
    except Exception as e:
        return {"error": str(e)}

def process_all_speakers_relevance(data):
    speaker_ids = data['Id'].unique()
    threads = []
    
    # Define a function to process each speaker
    def process_speaker(speaker_id):
        relevance_data = calculate_label_and_irrelevant_percentages(data, speaker_id)
        cache.set(f'speaker_relevance_{speaker_id}', relevance_data, timeout=3600)
    
    # Create and start a thread for each speaker
    for speaker_id in speaker_ids:
        thread = threading.Thread(target=process_speaker, args=(speaker_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

# Function to calculate speaker relevance and structure data for a donut chart
@api_view(['POST'])
def speaker_relevance(request, threshold=0.75):
    if request.method == "POST":
        try:
            # Parse JSON body
            data = json.loads(request.body)
            speaker_id = data.get('speaker_id')
            if speaker_id is None:
                return JsonResponse({"error": "Speaker ID is missing"}, status=400)
            
            # Retrieve cached data
            cached_relevance = cache.get(f'speaker_relevance_spk_{speaker_id}')
            if not cached_relevance:
                return JsonResponse({"error": "Relevance data not found in cache for this speaker."}, status=404)

            return JsonResponse({"label_percentages": cached_relevance}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)
    
def process_audio_and_store_in_cache(df):
    # Convert dataframe to JSON
    df_json = df.to_json(orient='split')

    cache.set('temp_df', df_json, timeout=3600)  # Timeout in seconds (e.g., 1 hour)

    return Response({"message": "Dataframe saved in cache"}, status=200)

@api_view(['POST'])
def update_emotion(request):
    try:
        # Step 1: Get the emotion from the request
        emotion = request.data.get('emotion', None)
        
        if not emotion:
            return Response({"error": "Emotion parameter is required"}, status=400)

        # Step 2: Retrieve the DataFrame from the cache
        df_json = cache.get('temp_df', None)

        if not df_json:
            return Response({"error": "No data found in cache"}, status=404)

        # Convert the JSON back to a DataFrame
        df = pd.read_json(df_json, orient='split')

        images = process_csv_and_plot(df, emotion)

        if not images:
            return Response({"error": "No images generated for the specified emotion"}, status=400)

        # Step 4: Return the generated images and other data as a response
        return Response({"images": images}, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)

def darken_color(color, amount=0.5):
    print('darken color function m agye hai')
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([max(0, x * amount) for x in c])

def plot_emotion_graph(patient_dialogues, emotion, speaker_id):
    print('plotting function m agye hai')

    # Check if the required columns exist
    if 'BeginTime' not in patient_dialogues or 'EndTime' not in patient_dialogues:
        print("Error: Missing 'BeginTime' or 'EndTime' columns.")
        return None
    if emotion not in patient_dialogues.columns:
        print(f"Error: The emotion column '{emotion}' is missing from the DataFrame.")
        return None
    
    sns.set(style="white")
    patient_dialogues['MidTime'] = (patient_dialogues['BeginTime'] + patient_dialogues['EndTime']) / 2

    # Add a new row with default values
    new_row = pd.DataFrame({
        'BeginTime': [0],
        'EndTime': [0],
        emotion: [0],
        'MidTime': [0]
    })
    patient_dialogues = pd.concat([new_row, patient_dialogues]).sort_values('MidTime').reset_index(drop=True)

    mid_times = patient_dialogues['MidTime'].values
    emotion_values = patient_dialogues[emotion].values

    # Check for NaN or infinite values in emotion_values
    if np.any(np.isnan(emotion_values)) or np.any(np.isinf(emotion_values)):
        print("Error: emotion_values contains NaN or infinite values.")
        return None
    
    max_time = patient_dialogues['EndTime'].max()
    tick_interval = 60 if max_time <= 660 else 120
    max_minute = int(np.ceil(max_time / 60))
    max_minute_even = max_minute + (max_minute % 2)

    # Define x_ticks based on max_time
    x_ticks = np.arange(0, (max_minute_even + 2) * 60, tick_interval)

    # Define x_tick_labels with the same length as x_ticks
    x_tick_labels = [str(i) for i in range(0, len(x_ticks))]

    # Ensure that the number of ticks and labels match
    if len(x_ticks) != len(x_tick_labels):
        print(f"Warning: Mismatch between number of ticks ({len(x_ticks)}) and labels ({len(x_tick_labels)}).")
        # Adjust the labels if needed (or truncate/extend x_tick_labels)
        x_tick_labels = x_tick_labels[:len(x_ticks)]  # Truncate or adjust to match length

    emotion_colors_dict = {
        'Admiration': ['#ffd1a5'],
        'Adoration': ['#fed1d6'],
        'Aesthetic Appreciation': ['#e8d7fe'],
        'Amusement': ['#ffcc70'],
        'Anger': ['#c74444'],
        'Anxiety': ['#8b68d5'],
        'Awe': ['#97bbda'],
        'Awkwardness': ['#dfe1b0'],
        'Boredom': ['#b6b6b6'],
        'Calmness': ['#bbd6e7'],
        'Concentration': ['#5c89ff'],
        'Confusion': ['#d38850'],
        'Contemplation': ['#c1bef3'],
        'Contempt': ['#8e9956'],
        'Contentment': ['#edcfc4'],
        'Craving': ['#8c8f68'],
        'Desire': ['#5c89ff'],
        'Determination': ['#fe7c34'],
        'Disappointment': ['#348998'],
        'Disgust': ['#4a9669'],
        'Distress': ['#d3ef95'],
        'Doubt': ['#ac9d68'],
        'Ecstasy': ['#f776bc'],
        'Embarrassment': ['#83d077'],
        'Empathic Pain': ['#d5777a'],
        'Entrancement': ['#c1bef3'],
        'Envy': ['#4a6e50'],
        'Excitement': ['#fef991'],
        'Fear': ['#e5e0f2'],
        'Guilt': ['#a1afb5'],
        'Horror': ['#955799'],
        'Interest': ['#c1d4e0'],
        'Joy': ['#fcdb3c'],
        'Love': ['#ec7876'],
        'Nostalgia': ['#c39db4'],
        'Pain': ['#ba8688'],
        'Pride': ['#ad70c6'],
        'Realization': ['#4a97bb'],
        'Relief': ['#fea696'],
        'Romance': ['#ebcf9a'],
        'Sadness': ['#597790'],
        'Satisfaction': ['#b8e5bf'],
        'Shame': ['#b39c9f'],
        'Surprise (negative)': ['#90e85e'],
        'Surprise (positive)': ['#a3fdfb'],
        'Sympathy': ['#999fe8'],
        'Tiredness': ['#8f8f8f'],
        'Triumph': ['#ef9a5d']
    }
        # Add other emotions as needed...
    base_color = emotion_colors_dict.get(emotion.capitalize(), ['dodgerblue'])[0]
    plt.figure(figsize=(25, 6))
    
    sns.lineplot(
        x=mid_times,
        y=emotion_values,
        label=emotion.capitalize(),
        linewidth=2.5,
        color=base_color
    )
    
    plt.fill_between(mid_times, emotion_values, color=base_color, alpha=0.3)
    plt.xticks(x_ticks, labels=x_tick_labels, fontsize=12)
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)', linewidth=1.5)
    plt.title(f'Emotion "{emotion.capitalize()}" for Speaker ID: {speaker_id}', fontsize=14, color=base_color)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Emotion Intensity', fontsize=12)
    plt.legend(loc='upper right')

    # Save the plot to a file
    file_name = f"{emotion.lower()}_speaker_{speaker_id}.png"
    file_path = os.path.join(PLOT_DIRECTORY, file_name)
    
    print(f"Saving plot to {file_path}")
    
    plt.savefig(file_path, format='png')
    plt.close()
    
    return file_path

def process_csv_and_plot(df, emotion_column):
    print('csv waale function m agye hai')
    unique_ids = df['Id'].unique()
    image_paths = []
    for speaker_id in unique_ids:
        print(speaker_id)
        if speaker_id == 'Unknown' or speaker_id=='unknown':
            continue
        print('loop m agye hai')
        emotion_columns = df.columns[6:]  # Assuming emotion columns start at index 6
        df_speaker = df.copy()
        df_speaker.loc[df_speaker['Id'] != speaker_id, emotion_columns] = 0.0
        print('speaker id wala kaam ho gya')
        img_path = plot_emotion_graph(df_speaker, emotion_column, speaker_id)
        print('image path agye h')
        image_paths.append(img_path)  # Append the image path to the list
    print('image ke path agye h')
    return image_paths



def highlight_entities(text):
    """
    Function to highlight named entities in the text by adding bold formatting around them.
    """
    doc = nlp(text)
    highlighted_text = text

    for ent in doc.ents:
        highlighted_text = (highlighted_text[:ent.start_char] + '**' + ent.text + '**' +
                            highlighted_text[ent.end_char:])
    
    return highlighted_text


def process_audio(api_key, audio_file_path):
    url = "https://api.hume.ai/v0/batch/jobs"
    headers = {
        "X-Hume-Api-Key": api_key
    }
    files = {
        'json': (None, '{"models": {  "prosody": {"identify_speakers": true}}}'),
        'file': (audio_file_path, open(audio_file_path, 'rb'))
    }
    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        job_id = response.json().get("job_id")
        status_url = f"https://api.hume.ai/v0/batch/jobs/{job_id}"
        status_headers = {
            "X-Hume-Api-Key": api_key,
            "accept": "application/json; charset=utf-8"
        }

        def poll_status():
            while True:
                status_response = requests.get(status_url, headers=status_headers)
                if status_response.status_code == 200:
                    response_json = status_response.json()
                    job_status = response_json.get('state', {}).get('status')
                    if job_status == "COMPLETED":
                        break
                    time.sleep(10)
                else:
                    print(f"Failed to check job status. Status code: {status_response.status_code}")
                    print("Response text:", status_response.text)
                    break
        
        # Start the polling thread for job status
        status_thread = threading.Thread(target=poll_status)
        status_thread.start()
        status_thread.join()  # Wait for job completion

        artifacts_url = f"https://api.hume.ai/v0/batch/jobs/{job_id}/artifacts"
        response = requests.get(artifacts_url, headers=headers)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.csv'):
                        with zip_ref.open(file_name) as csv_file:
                            df = pd.read_csv(csv_file)
                            transcription_text = ' '.join(df['Text'].dropna().astype(str).tolist())
                            print('Transcription completed.')
                            cache.set('transcription_text', transcription_text, timeout=3600)  #
                            
                            return df, transcription_text
        else:
            print(f"Failed to retrieve predictions. Status code: {response.status_code}")
            print(response.text)
    else:
        print(f"Failed to submit job. Status code: {response.status_code}")
        print(response.text)
    return None, None

def summarize_chunk(chunk, summarizer=None):
    doc = nlp(chunk)
    processed_summary = chunk
    offset = 0  # To handle length changes due to insertion of brackets/asterisks

    # Mark entities with asterisks for names and brackets for dates
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            processed_summary = (
                processed_summary[:ent.start_char + offset]
                + "*"
                + ent.text
                + "*"
                + processed_summary[ent.end_char + offset:]
            )
            offset += 2
        elif ent.label_ == "DATE":
            processed_summary = (
                processed_summary[:ent.start_char + offset]
                + "["
                + ent.text
                + "]"
                + processed_summary[ent.end_char + offset:]
            )
            offset += 2

    return processed_summary

@api_view(['POST'])
def create_summary(request):
    try:
        # Extract emotion_to_plot from query parameters (default to "Calmness" if not provided)
        emotion_to_plot = request.data.get("emotion_to_plot", "Calmness")

        start_time = time.time()
        # Extract the audio file from the POST request
        audio_file = request.FILES.get('audio', None)

        if not audio_file:
            return Response({"error": "No audio file provided"}, status=400)

        api_key = "bDf0buDHYGE9AHWaDM4kLtAFtpEZWvq3iU3agntTD5UcX9aG"

        # Use a temporary file to handle the uploaded audio
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            for chunk in audio_file.chunks():
                temp_audio_file.write(chunk)
            temp_audio_path = temp_audio_file.name

        try:
            # Process the audio and get transcription text
            df, transcription_text = process_audio(api_key, temp_audio_path)

            if not transcription_text:
                return Response({"error": "Transcription failed"}, status=500)

            # Summarize the transcribed paragraph in parallel
            chunk_size = 1000
            chunks = [transcription_text[i:i + chunk_size] for i in range(0, len(transcription_text), chunk_size)]

            # Define the function for speaker relevance calculation
            def process_and_cache_relevance(speaker_id):
                relevance_data = calculate_label_and_irrelevant_percentages(df, speaker_id)
                cache_key = f'speaker_relevance_{speaker_id}'
                cache.set(cache_key, relevance_data, timeout=3600)  # Cache for 1 hour

            # Use ThreadPoolExecutor for both summarization and speaker relevance calculation in parallel
            with ThreadPoolExecutor() as executor:
                # Submit both tasks concurrently
                future_summaries = executor.submit(lambda: list(executor.map(lambda chunk: summarize_chunk(chunk), chunks)))
                future_relevance = executor.submit(lambda: list(executor.map(process_and_cache_relevance, df['Id'].unique())))

                # Wait for both tasks to complete
                summaries = future_summaries.result()
                relevance = future_relevance.result()

            # Prepare the chunk summaries
            chunk_summaries = [
                {"chunk_number": i + 1, "chunk": chunk, "summary": summaries[i]}
                for i, chunk in enumerate(chunks)
            ]

            # Plot images for the specified emotion
            images = process_csv_and_plot(df, emotion_to_plot)
            process_audio_and_store_in_cache(df)

        finally:
            # Clean up the temporary audio file
            os.remove(temp_audio_path)

        end_time = time.time()
        print("Time taken to process the audio file: ", end_time - start_time)

        return Response({"chunk_summaries": chunk_summaries, "images": images}, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)

# View to create summary
# @api_view(['POST'])
# def create_summary(request):
    try:
        # Extract emotion_to_plot from query parameters (default to "Calmness" if not provided)
        emotion_to_plot = request.data.get("emotion_to_plot", "Calmness")

        start_time = time.time()
        # Extract the audio file from the POST request
        audio_file = request.FILES.get('audio', None)

        if not audio_file:
            return Response({"error": "No audio file provided"}, status=400)

        api_key = "S7AsbRYYNXg8TCxjUq3ZE1P4BmTv5u1xCfNVBdprA9vAiGmB"

        # Use a temporary file to handle the uploaded audio
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            for chunk in audio_file.chunks():
                temp_audio_file.write(chunk)
            temp_audio_path = temp_audio_file.name

        try:
            # Process the audio and get transcription text
            df, transcription_text = process_audio(api_key, temp_audio_path)

            if not transcription_text:
                return Response({"error": "Transcription failed"}, status=500)

            # Summarize the transcribed paragraph in parallel
            chunk_size = 1000
            chunks = [transcription_text[i:i + chunk_size] for i in range(0, len(transcription_text), chunk_size)]

            # Define the function for speaker relevance calculation
            def process_and_cache_relevance(speaker_id):
                relevance_data = calculate_label_and_irrelevant_percentages(df, speaker_id)
                cache_key = f'speaker_relevance_{speaker_id}'
                cache.set(cache_key, relevance_data, timeout=3600)  # Cache for 1 hour

            # Use ThreadPoolExecutor for both summarization and speaker relevance calculation in parallel
            with ThreadPoolExecutor() as executor:
                # Submit both tasks concurrently
                future_summaries = executor.submit(lambda: list(executor.map(lambda chunk: summarize_chunk(chunk, summarizer), chunks)))
                future_relevance = executor.submit(lambda: list(executor.map(process_and_cache_relevance, df['Id'].unique())))

                # Wait for both tasks to complete
                summaries = future_summaries.result()
                relevance = future_relevance.result()

            # Prepare the chunk summaries
            chunk_summaries = [
                {"chunk_number": i + 1, "chunk": chunk, "summary": summaries[i]}
                for i, chunk in enumerate(chunks)
            ]

            # Plot images for the specified emotion
            images = process_csv_and_plot(df, emotion_to_plot)
            process_audio_and_store_in_cache(df)

        finally:
            # Clean up the temporary audio file
            os.remove(temp_audio_path)

        end_time = time.time()
        print("Time taken to process the audio file: ", end_time - start_time)

        return Response({"chunk_summaries": chunk_summaries, "images": images}, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)