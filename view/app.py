import os
import html
from flask import Flask, render_template, request
from datasets import load_from_disk

app = Flask(__name__)

# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'data', 'cleaned_ai4privacy_300k_pii')

try:
    dataset = load_from_disk(DATASET_PATH)
except Exception as e:
    dataset = None
    print(f"Error loading dataset: {e}")

def highlight_pii(text, mask):
    if not mask:
        return html.escape(text)
    
    # Sort by start index
    sorted_mask = sorted(mask, key=lambda x: x['start'])
    html_parts = []
    last_idx = 0
    
    for item in sorted_mask:
        start = item['start']
        end = item['end']
        label = item['label']
        
        # Guard against overlapping or invalid indices
        if start < last_idx or start > len(text) or end > len(text):
            continue
            
        # Add normal text preceding this entity
        html_parts.append(html.escape(text[last_idx:start]))
        
        # Add highlighted entity HTML
        val = text[start:end]
        escaped_val = html.escape(val)
        escaped_label = html.escape(label)
        
        html_parts.append(
            f'<mark class="pii-entity pii-{escaped_label.lower()}" data-label="{escaped_label}">'
            f'{escaped_val}'
            f'<span class="pii-label">{escaped_label}</span>'
            f'</mark>'
        )
        last_idx = end
        
    html_parts.append(html.escape(text[last_idx:]))
    return "".join(html_parts)

@app.route("/")
def index():
    if not dataset:
        return "Dataset not found or failed to load. Please check the dataset path.", 500
    
    # Allow picking split: train, validation, test
    split = request.args.get('split', 'train')
    if split not in dataset:
        split = 'train'
        
    # Get total examples in this split
    total_examples = len(dataset[split])
    
    # Get index from query param
    try:
        index = int(request.args.get('index', 0))
    except ValueError:
        index = 0
        
    # Ensure index is within bounds
    if index < 0:
        index = 0
    elif index >= total_examples:
        index = total_examples - 1
        
    example = dataset[split][index]
    
    # Generate highlighted HTML text
    highlighted_text = highlight_pii(example['source_text'], example.get('privacy_mask', []))
    
    # Extract unique labels in this example for a handy legend
    unique_labels = sorted(list(set(item['label'] for item in example.get('privacy_mask', []))))
    
    return render_template(
        "index.html",
        example=example,
        highlighted_text=highlighted_text,
        split=split,
        index=index,
        total_examples=total_examples,
        unique_labels=unique_labels
    )
