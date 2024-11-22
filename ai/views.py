from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests
from io import StringIO
import pandas as pd
import json
import zipfile
from django.http import JsonResponse
import io
import os
from .models import nlp,summarizer,sentence_model
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
from django.views.decorators.csrf import csrf_exempt

PLOT_DIRECTORY = "static/plots/"
os.makedirs(PLOT_DIRECTORY, exist_ok=True)

# Function to calculate speaker relevance and structure data for a donut chart
@api_view(['POST'])
def speaker_relevance(request, threshold=0.75):
    if request.method == "POST":
        try:
            # Parse JSON body
            data = json.loads(request.body)
            print(data)
            speaker_id = data.get('speaker_id')
            print(speaker_id)
            if speaker_id is None:
                return JsonResponse({"error": "Speaker ID is missing"}, status=400)
            
            # Load cached data
            df_json = cache.get('temp_df', None)
            if not df_json:
                return JsonResponse({"error": "No data found in cache"}, status=404)

            # Convert JSON back to DataFrame
            data = pd.read_json(StringIO(df_json), orient='split')
            speaker_id = f"spk_{speaker_id}"
            print(speaker_id)  # Debugging purposes
            
            # Combine each speaker's content
            speaker_texts = data.groupby("Id")["Text"].apply(" ".join)
            print(speaker_texts)  # Debugging purposes
            # Check if the speaker ID exists
            if str(speaker_id) not in speaker_texts:
                return JsonResponse({"error": f"Speaker ID '{speaker_id}' not found in the data."}, status=404)

            # Rest of your processing code...
            speaker_text = speaker_texts[speaker_id]
            meeting_context = " ".join(data[data["Id"] != speaker_id]["Text"].dropna())

            # Encode and compute similarity
            meeting_embedding = sentence_model.encode(meeting_context, convert_to_tensor=True)
            speaker_embedding = sentence_model.encode(speaker_text, convert_to_tensor=True)
            cosine_similarity = util.cos_sim(speaker_embedding, meeting_embedding).item()

            # Classify relevance
            chart_data = {
                "Relevant": cosine_similarity ,
                "Not Relevant": 1 - cosine_similarity 
            }

            return JsonResponse({"chart_data": chart_data}, status=200)
        
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
        if speaker_id == 'Unknown':
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

# Function to process audio and get transcription text
def process_audio(api_key,audio_file_path, emotion_to_plot="Calmness"):
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
        while True:
            status_response = requests.get(status_url, headers=status_headers)
            if status_response.status_code == 200:
                response_json = status_response.json()
                job_status = response_json.get('state', {}).get('status')
                if job_status == "COMPLETED":
                    break
                else:
                    time.sleep(10)
            else:
                print(f"Failed to check job status. Status code: {status_response.status_code}")
                print("Response text:", status_response.text)
                break
        artifacts_url = f"https://api.hume.ai/v0/batch/jobs/{job_id}/artifacts"
        response = requests.get(artifacts_url, headers=headers)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.csv'):
                        with zip_ref.open(file_name) as csv_file:
                            df = pd.read_csv(csv_file)
                            transcription_text = ' '.join(df['Text'].dropna().astype(str).tolist())
                            print('transcription agyi h')
                            # Plot the graph for the specified emotion and return images
                            images = process_csv_and_plot(df, emotion_to_plot)
                            process_audio_and_store_in_cache(df)
                            return transcription_text,images
        else:
            print(f"Failed to retrieve predictions. Status code: {response.status_code}")
            print(response.text)
    else:
        print(f"Failed to submit job. Status code: {response.status_code}")
        print(response.text)
    return None, None

# Create your view here
@api_view(['POST'])
def create_summary(request):
    try:
        start_time=time.time()
        # Extract the audio file from the POST request
        audio_file = request.FILES.get('audio', None)
        
        if not audio_file:
            return Response({"error": "No audio file provided"}, status=400)
        
        api_key = "S7AsbRYYNXg8TCxjUq3ZE1P4BmTv5u1xCfNVBdprA9vAiGmB"

        audio_file_path = f"temp_{audio_file.name}"
        
        # Save the file temporarily to process it
        with open(audio_file_path, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        
        # Process the audio and get transcription text
        transcription_text,images = process_audio(api_key,audio_file_path)
        
        # Clean up the temporary audio file after processing
        os.remove(audio_file_path)
        
        if not transcription_text:
            return Response({"error": "Transcription failed"}, status=500)

        # Summarize the transcribed paragraph
        chunk_size = 1000
        chunks = [transcription_text[i:i + chunk_size] for i in range(0, len(transcription_text), chunk_size)]

        # Summarize each chunk and store the results
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = summarizer(chunk)[0]['summary_text']
            # highlighted_summary = highlight_entities(summary)
            chunk_summaries.append({
                "chunk_number": i + 1,
                "chunk": chunk,
                "summary":summary          })
        end_time=time.time()
        print("Time taken to process the audio file: ", end_time - start_time)
        print(images)

        return Response({"chunk_summaries": chunk_summaries,"images":images}, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
