# Simple-GLM-OCR

Simple optical character recognition based on [GLM-OCR](https://github.com/zai-org/GLM-OCR) with fewer dependencies.

## Example

```python
from simpleglmocr import SimpleGlmOcr

model = SimpleGlmOcr()

text = model.run("Text Recognition:", "testimage.jpg")

print(text)
```

This will print the following text for the image shown below:

```
Hello, GLM-OCR!
This is a test image.
The quick brown fox jumps
over the lazy dog.
```

#### Image

<img width="700" alt="testimage.jpg" src="https://raw.githubusercontent.com/99991/Simple-GLM-OCR/refs/heads/main/testimage.jpg" />

# Installation

* Prerequisites: A Python environment with `python`, `pip` and `git`.
* First, install PyTorch according to the instructions [on their website](https://pytorch.org/).
* Next, install the following Python libraries with pip:

```
pip install regex pillow safetensors
```

* Now you can clone the repository and run the example.

```bash
git clone https://github.com/99991/Simple-GLM-OCR.git
cd Simple-GLM-OCR
python example.py
```

# Server

You can start a server for a web-based OCR experience by running the following command in the Simple-GLM-OCR directory:

```
python server.py
```

You can then visit the website at http://127.0.0.1:8000 to upload images for text recognition, or you can use the API (see below).

<img width="819" height="362" alt="server" src="https://github.com/user-attachments/assets/8afd3e9f-c6e9-44a9-9276-641c6d9ea6fa" />

# API

After you have started the server, you can use the API:

```python
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
```

#### Image

<img width="500" alt="obama" src="https://raw.githubusercontent.com/99991/Simple-GLM-OCR/refs/heads/main/obama.jpg" />

#### Output

    ```json
    {
        "last_name": "OBAMA",
        "first_name": "BARACK",
        "tie color": "blue",
        "facial expression": "smiling",
        "age": "47",
        "body posture": "crossed arms",
        "background": "American flag and presidential seal"
    }
    ```

# Prompts

GLM-OCR supports [multiple prompt formats](https://huggingface.co/zai-org/GLM-OCR#prompt-limited):

* `Text Recognition:` (for general text recognition)
* `Table Recognition:` (for tables as HTML)
* `Formula Recognition:` (for equations in LaTeX)
* Schema-based JSON extraction
