from simpleglmocr import SimpleGlmOcr

model = SimpleGlmOcr()

text = model.run("Text Recognition:", "testimage.jpg")

print(text)
