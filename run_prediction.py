import requests, os, sys

# Path to the uploaded image (the user uploaded to user_uploaded folder)
image_path = r'C:\Users\Swastik Sen Sharma\.gemini\antigravity\brain\0fd3953b-6de3-4d52-833b-ca47239727d1\.user_uploaded\media_1794925670572_3c77ffa3.png'

# Ensure the server is reachable
url = 'http://127.0.0.1:8000/api/v1/ml-inference/predict/taxonomy'

if not os.path.isfile(image_path):
    print('Image not found at', image_path)
    sys.exit(1)

files = {'file': open(image_path, 'rb')}

try:
    response = requests.post(url, files=files)
    print('Status code:', response.status_code)
    print('Response JSON:', response.json())
except Exception as e:
    print('Error contacting the server:', e)
