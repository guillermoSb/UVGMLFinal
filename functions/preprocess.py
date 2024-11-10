from flask import Flask, request, jsonify
import pandas as pd
from google.cloud import storage
import os
import tempfile
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

def clean_and_feature_engineer(data):
    # Renaming
    data.rename(columns={
    'id': 'id',
    'Marathon': 'marathon',
    'Name': 'name',
    'Category': 'category',
    'km4week': 'km_per_week',
    'sp4week': 'speed_per_week',
    'CrossTraining': 'cross_training',
    'Wall21': 'wall_21',
    'MarathonTime': 'marathon_time',
    'CATEGORY': 'category_type'
    }, inplace=True)
    data.drop(columns=['name', 'id', 'category_type', 'marathon', 'cross_training'], inplace=True)
    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])
    cols_to_check = ['category', 'km_per_week', 'speed_per_week', 'wall_21', 'marathon_time']
    data[cols_to_check] = data[cols_to_check].apply(pd.to_numeric, errors='coerce')
    data.dropna(subset=cols_to_check, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data
@app.route('/', methods=['POST'])
def process_file(event):
    print("RUNNING FUNCTION:")
    request_json = request.get_json()
    print(request_json)
    # Set up client
    client = storage.Client()
    bucket_name = request_json['bucket']
    blob_name = request_json['name']
    if not blob_name.startswith("input/"):
        print("Ignoring file outside specific-folder/")
        return ('', 200)  # Ignore files outside the specified folder

    # Download file from GCS to a temporary location
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    _, temp_local_filename = tempfile.mkstemp()
    blob.download_to_filename(temp_local_filename)

    # Load data and process
    data = pd.read_csv(temp_local_filename)
    cleaned_data = clean_and_feature_engineer(data)

    # Save the processed data to a new file and upload back to GCS
    processed_filename = f"train.csv"
    cleaned_data.to_csv(processed_filename, index=False)
    processed_blob = bucket.blob(f"processed/{processed_filename}")
    processed_blob.upload_from_filename(processed_filename)

    # Clean up temporary files
    os.remove(temp_local_filename)
    os.remove(processed_filename)

    print(f"File processed and saved to: processed/{processed_filename}")
    return jsonify({'message': f'File processed and saved as processed/{processed_filename}'}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))