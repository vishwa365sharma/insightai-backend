from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import io
import os
import json
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

current_df = None

@app.route('/')
def home():
    return jsonify({"status": "InsightAI Backend is Live with Google Gemini!"})

def analyze_dataframe(df):
    rows, cols = df.shape
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    num_cols = len(df.select_dtypes(include=['number']).columns)
    cat_cols = cols - num_cols

    total_cells = rows * cols
    health_score = round(100 - ((missing_count / total_cells) * 100), 1) if total_cells > 0 else 100

    preview_df = df.head(10).fillna('NULL')
    columns = list(preview_df.columns)
    records = preview_df.values.tolist()

    return {
        "rows": rows, "cols": cols, "missing": missing_count, "duplicates": duplicate_count,
        "health_score": health_score, "num_cols": num_cols, "cat_cols": cat_cols,
        "columns": columns, "records": records
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df
    if 'file' not in request.files: return jsonify({"error": "Koi file select nahi ki gayi"}), 400
    file = request.files['file']
    filename = file.filename

    try:
        if filename.endswith('.csv'): current_df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')): current_df = pd.read_excel(file)
        else: return jsonify({"error": "Sirf CSV ya Excel file upload karein"}), 400

        result = analyze_dataframe(current_df)
        result["message"] = f"{filename} successfully analyze ho gayi!"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clean', methods=['POST'])
def clean_data():
    global current_df
    if current_df is None: return jsonify({"error": "Pehle dataset upload karein!"}), 400

    action = request.json.get('action')
    msg = ""

    if action == 'drop_missing':
        current_df = current_df.dropna()
        msg = "Saari missing rows delete kar di gayi hain."
    elif action == 'fill_missing':
        method = request.json.get('method', 'mean')
        for col in current_df.columns:
            if current_df[col].dtype in ['int64', 'float64']:
                if method == 'mean': current_df[col] = current_df[col].fillna(current_df[col].mean())
                elif method == 'median': current_df[col] = current_df[col].fillna(current_df[col].median())
                elif method == 'max': current_df[col] = current_df[col].fillna(current_df[col].max())
                elif method == 'min': current_df[col] = current_df[col].fillna(current_df[col].min())
                elif method == 'mode':
                    mode_vals = current_df[col].mode()
                    if not mode_vals.empty: current_df[col] = current_df[col].fillna(mode_vals[0])
            else:
                current_df[col] = current_df[col].fillna('Unknown')
        msg = f"Missing values ko {method.upper()} se fill kar diya gaya hai."
    elif action == 'drop_duplicates':
        current_df = current_df.drop_duplicates()
        msg = "Duplicate rows hata di gayi hain."
    else:
        return jsonify({"error": "Invalid action"}), 400

    result = analyze_dataframe(current_df)
    result["message"] = msg
    return jsonify(result)

@app.route('/generate-insights', methods=['POST'])
def generate_insights():
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return jsonify({"error": "GEMINI_API_KEY server par set nahi hai."}), 500

    data_summary = request.json
    
    prompt = f"""
    Analyze the following dataset summary and provide exactly 2 actionable business insights.
    Total Rows: {data_summary.get('rows')}
    Columns: {data_summary.get('columns')}
    Sample Data (First 10 rows): {data_summary.get('records')}
    
    Return ONLY a valid JSON object with this exact structure. Do not output markdown, code blocks, or any intro text.
    {{
        "insight1": {{
            "title": "Short Title 1",
            "confidence": "95%",
            "description": "2-3 sentences explaining the insight."
        }},
        "insight2": {{
            "title": "Short Title 2",
            "confidence": "91%",
            "description": "2-3 sentences explaining the insight."
        }}
    }}
    """

    try:
        genai.configure(api_key=gemini_key)
        # Gemini 1.5 Flash - Sabse fast aur JSON mode support karta hai
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        return jsonify(json.loads(response_text))
        
    except Exception as e:
        return jsonify({"error": f"Gemini Error: {str(e)}"}), 500

@app.route('/download-csv')
def download_csv():
    global current_df
    if current_df is None: return "No dataset uploaded", 400
    buffer = io.StringIO()
    current_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(io.BytesIO(buffer.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='cleaned_dataset.csv')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
