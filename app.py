from flask import Flask , render_template , request
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

modal_name = 'Helsinki-NLP/opus-mt-en-hi'

model = MarianMTModel.from_pretrained(modal_name)
tokenizer = MarianTokenizer.from_pretrained(modal_name)

# create translation function to translate text from English to Hindi
def translation(data):
    # tokenize the input text
    inputs = tokenizer(data, return_tensors="pt", padding=True)

    # generate translation using the model
    translated_token = model.generate(**inputs)

    # decode the generated translation
    output = tokenizer.decode(translated_token[0], skip_special_tokens=True)
    return output

@app.route('/', methods=['GET', 'POST'])

def index():
    translated_text = ""
    if request.method == 'POST':
        data = request.form['data']
        translated_text = translation(data)


    return render_template('index.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)