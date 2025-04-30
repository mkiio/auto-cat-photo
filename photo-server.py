"""
Super Simple local photo server for  Pi
---------------------------------------------------------------------
• Hosts any photos found in the /photos directory relative to the script
• Intended to allow remotely checking on auto-cat-photo
• Run script and access at the pi's address on port 5000 (e.g. 192.168.1.100:5000)
"""

from flask import Flask, send_from_directory, render_template_string, request
import os
import math

PHOTO_FOLDER = 'photos'

app = Flask(__name__)

@app.route('/')
def index():
    # Gather and sort photos
    photos = sorted(
        f for f in os.listdir(PHOTO_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
    )

    # Pagination parameters
    per_page = 50
    page = request.args.get('page', 1, type=int)
    total_photos = len(photos)
    total_pages = math.ceil(total_photos / per_page)

    # Slice out the photos for this page
    start = (page - 1) * per_page
    end = start + per_page
    page_photos = photos[start:end]

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Photo Gallery</title>
        <style>
            body {
                font-family: sans-serif;
                background: #f4f4f4;
                padding: 20px;
                margin: 0;
            }
            h1 {
                margin-bottom: 20px;
            }
            .gallery {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                grid-gap: 10px;
            }
            .gallery img {
                width: 100%;
                height: auto;
                display: block;
                border: 1px solid #ccc;
                background: white;
            }
            .pagination {
                margin-top: 20px;
                text-align: center;
            }
            .pagination a {
                margin: 0 5px;
                text-decoration: none;
                color: #007bff;
            }
            .pagination span {
                margin: 0 5px;
                color: #555;
            }
        </style>
    </head>
    <body>
        <h1>Photo Gallery</h1>
        <div class="gallery">
            {% for photo in photos %}
            <a href="/photos/{{ photo }}">
                <img src="/photos/{{ photo }}" alt="{{ photo }}">
            </a>
            {% endfor %}
        </div>

        {% if total_pages > 1 %}
        <div class="pagination">
            {% if page > 1 %}
                <a href="/?page={{ page - 1 }}">&laquo; Previous</a>
            {% else %}
                <span>&laquo; Previous</span>
            {% endif %}

            <span>Page {{ page }} of {{ total_pages }}</span>

            {% if page < total_pages %}
                <a href="/?page={{ page + 1 }}">Next &raquo;</a>
            {% else %}
                <span>Next &raquo;</span>
            {% endif %}
        </div>
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(
        html,
        photos=page_photos,
        page=page,
        total_pages=total_pages
    )

@app.route('/photos/<path:filename>')
def serve_photo(filename):
    return send_from_directory(PHOTO_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
