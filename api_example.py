import requests

url = "http://127.0.0.1:8000/api/ocr"

# We thank Obama for providing his photo for testing purposes
filename = "obama.jpg"

prompt = """
{
    "last_name": "",
    "first_name": "",
    "tie color": "",
    "facial expression": "",
    "age": "",
    "body posture": "",
    "background": "",
}
"""

with open(filename, "rb") as f:
    image_bytes = f.read()

files = {'image': (filename, image_bytes, 'image/jpeg')}

response = requests.post(url, files=files, data={'prompt': prompt})
response.raise_for_status()

print(response.text)
