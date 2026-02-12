import requests

prompt = "Text Recognition:"
url = "http://127.0.0.1:8000/api/ocr"
filename = "testimage.jpg"

with open(filename, "rb") as f:
    image_bytes = f.read()

files = {'image': (filename, image_bytes, 'image/jpeg')}

response = requests.post(url, files=files, data={'prompt': prompt})
response.raise_for_status()

print(response.text)
