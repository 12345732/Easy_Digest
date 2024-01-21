from flask import Flask, render_template, request,url_for

from transformers import BartTokenizer, BartForConditionalGeneration
import pyttsx3
import requests
from bs4 import BeautifulSoup

app = Flask(__name__,template_folder='templates')

@app.route('/')
def enter():
    return render_template('Enter.html')

@app.route('/main')
def main():
    return render_template('main.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/services')
def services():
    return render_template('services.html')
@app.route('/paragraphs', methods=['GET', 'POST'])
def handle_paragraphs():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        if input_text is not None and input_text.strip() != '':
            summarized_text = summarize_text(input_text)
            if summarized_text is None:
                error_message = "Error: Failed to summarize the text."
                return render_template('paragraphs.html', input_text=input_text, summarized_text='', error_message=error_message)
            return render_template('paragraphs.html', input_text=input_text, summarized_text=summarized_text, error_message='')
        else:
            return render_template('paragraphs.html', input_text=input_text, summarized_text='', error_message='')
    else:
        return render_template('paragraphs.html', input_text='', summarized_text='', error_message='')

@app.route('/articles', methods=['GET', 'POST'])
def handle_articles():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        if input_text is not None and input_text.strip() != '':
            summarized_text = summarize_text(input_text)
            if summarized_text is None:
                error_message = "Error: Failed to summarize the text."
                return render_template('articles.html', input_text=input_text,summarized_text='', error_message=error_message)
            return render_template('articles.html', input_text=input_text,summarized_text=summarized_text, error_message='')
        else:
            return render_template('articles.html', input_text=input_text,summarized_text='', error_message='')
    else:
        return render_template('articles.html', input_text=input_text,summarized_text='', error_message='')


@app.route('/news', methods=['GET', 'POST'])
def handle_news():
    if request.method == 'POST':
        input_url = request.form.get('input_url')
        if input_url:
            article_content = scrape_article(input_url)
            if article_content:
                summarized_text = summarize_text(article_content)
                return render_template('news.html', input_url=input_url, summarized_text=summarized_text)
            else:
                error_message = "Error: Failed to scrape the article."
                return render_template('news.html', input_url=input_url, summarized_text='', error_message=error_message)
        else:
            return render_template('news.html', input_url='', summarized_text='', error_message='')
    else:
        return render_template('news.html', input_url='', summarized_text='', error_message='')

def scrape_article(url):
    try:
        # Send an HTTP GET request to the provided URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find and extract the main content of the article
            article_content = ""
            main_content = soup.find('article')  # You may need to adjust this selector
            if main_content:
                paragraphs = main_content.find_all('p')
                for p in paragraphs:
                    article_content += p.get_text() + '\n'

            return article_content
        else:
            return None
    except Exception as e:
        print("Error during web scraping:", e)
        return None



@app.route('/blogs', methods=['GET', 'POST'])
def handle_blogs():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        if input_text is not None and input_text.strip() != '':
            summarized_text = summarize_text(input_text)
            if summarized_text is None:
                error_message = "Error: Failed to summarize the text."
                return render_template('blogs.html', input_text=input_text,summarized_text='', error_message=error_message)
            return render_template('blogs.html', input_text=input_text,summarized_text=summarized_text, error_message='')
        else:
            return render_template('blogs.html', input_text=input_text,summarized_text='', error_message='')
    else:
        return render_template('blogs.html', input_text=input_text,summarized_text='', error_message='')

@app.route('/voice_output', methods=['POST'])
def voice_output():
    output_text = request.form['output_text']
    if output_text:
        generate_voice_output(output_text)
        return "Success"
    else:
        return "Error: No output text provided."


def summarize_text(input_text):
    if isinstance(input_text, str):
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        inputs = tokenizer.batch_encode_plus([input_text], max_length=1024, truncation=True, padding='longest', return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        summary_ids = model.generate(input_ids, attention_mask=attention_mask, num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    else:
        raise ValueError('Invalid input. Text input must be a string.')
def generate_voice_output(output_text):
    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()

    # Set the properties for voice output (optional)
    # You can customize voice, volume, and speed settings
    engine.setProperty('voice', 'en-us')  # Example: English (US)
    engine.setProperty('volume', 1.0)     # Example: Volume level 1.0 (max)
    engine.setProperty('rate', 150)       # Example: Speaking rate of 150 words per minute

    # Say the output text
    engine.say(output_text)

    # Wait for speech to complete
    engine.runAndWait()

if __name__ == '__main__':
    app.run(debug=True)