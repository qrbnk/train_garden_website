import os
import re
import shutil
import tempfile
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import pandas as pd
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer, util
import torch
import gc 
from threading import local 

# ----- Environment & Setup -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 5 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Data structure for per-worker isolation (used for initialization)
worker_data = local()

# --- Custom Error Handler for File Size ---
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File size exceeds the 5MB limit"}), 413

# ----- Core Helper Functions (Defined FIRST) -----

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_col(names, cols_lower_map):
    """Helper to find the correct column name based on aliases."""
    for k, orig in cols_lower_map.items():
        for n in names:
            if n in k:
                return orig
    return None

def preprocess_and_chunk(text):
    # [Regex helpers (Global Compiled Patterns)]
    CLEAN_RE = re.compile(r"[^\w\s,+\-#]")
    MULTISPACE_RE = re.compile(r"\s+")
    SPLIT_RE = re.compile(r'[\n,;]+')
    
    if not text:
        return []
    t = text.replace("•", ",").replace("·", ",")
    t = re.sub(r"[\r\t]+", " ", t)
    t = re.sub(r"[():/]", " ", t)
    t = CLEAN_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    parts = SPLIT_RE.split(t)
    final = []
    for p in parts:
        if len(p) > 200:
            final.extend([s.strip() for s in re.split(r'[.!?]+', p) if s.strip()])
        else:
            final.append(p)
    return [f.lower() for f in final if f]

def extract_text_from_file(path):
    text = ""
    try:
        if path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            for p in reader.pages:
                text += (p.extract_text() or "") + "\n"
        elif path.lower().endswith(".docx"):
            doc = docx.Document(path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return text.strip()

# ----------------------------------------------------
# 🌟 Model Isolation/Lazy Loader 🌟
# ----------------------------------------------------
def get_isolated_resources():
    """Loads model, embeddings, and skills data if not already loaded in this worker process."""
    # This runs only once per worker/process!
    if not hasattr(worker_data, 'model'):
        print(f"--- Worker {os.getpid()} initializing ML resources... ---")
        
        SKILLS_CSV = "skills.csv"
        skills_df_local = pd.read_csv(SKILLS_CSV)
        skills_df_local.columns = [c.strip() for c in skills_df_local.columns]

        # --- Column Identification and Canonicalization on local copy ---
        cols_lower = {c.lower().replace(" ", "").replace("_", ""): c for c in skills_df_local.columns}
        
        # Use helper find_col with the local map
        skill_name_col = find_col(["skillname", "skill"], cols_lower) or list(skills_df_local.columns)[0]
        skill_cat_col = find_col(["skillcategory", "category"], cols_lower)
        skill_desc_col = find_col(["skilldescription", "description"], cols_lower)
        is_ai_col = find_col(["is_ai_recognizable", "isai"], cols_lower)
        prof_col = find_col(["identifiedproficiencylevel", "proficiency"], cols_lower)

        skills_df_local[skill_name_col] = skills_df_local[skill_name_col].fillna("").astype(str)
        skills_df_local["skill_name_clean"] = skills_df_local[skill_name_col].str.strip()
        skills_df_local["skill_name_lower"] = skills_df_local["skill_name_clean"].str.lower()
        skills_df_local["skill_category"] = skills_df_local.get(skill_cat_col, "General").fillna("General").astype(str)
        skills_df_local["skill_description"] = skills_df_local.get(skill_desc_col, "").fillna("").astype(str)
        skills_df_local["is_ai_recognizable"] = skills_df_local.get(is_ai_col, 1).fillna(1)
        skills_df_local["identified_proficiency_level"] = skills_df_local.get(prof_col, "").fillna("")

        skill_names_lower_local = skills_df_local["skill_name_lower"].tolist()
        
        worker_data.model = SentenceTransformer("all-MiniLM-L6-v2")
        worker_data.embeddings = worker_data.model.encode(
            skill_names_lower_local, convert_to_tensor=True
        )
        
        worker_data.skills_df = skills_df_local
        worker_data.skill_names_lower = skill_names_lower_local
        worker_data.skill_name_col = skill_name_col

    return worker_data.model, worker_data.embeddings, worker_data.skills_df, \
           worker_data.skill_names_lower, worker_data.skill_name_col

# ----------------------------------------------------
# 🌟 Analysis Function (Uses Local Resources) 🌟
# ----------------------------------------------------
def extract_skills_enhanced(cv_text, threshold=0.48, top_k=6):
    
    # Retrieve isolated resources for this worker
    MODEL_LOCAL, SKILL_EMBEDDINGS_LOCAL, skills_df_local, \
    SKILL_NAMES_LOWER_LOCAL, skill_name_col_local = get_isolated_resources()
    
    chunks = preprocess_and_chunk(cv_text)
    matched_idx = set()
    tensors_to_delete = []

    for chunk in chunks:
        emb = None 
        try:
            # Use local MODEL
            emb = MODEL_LOCAL.encode([chunk], convert_to_tensor=True)
            tensors_to_delete.append(emb) 

            # Use local SKILL_EMBEDDINGS
            sims = util.cos_sim(emb, SKILL_EMBEDDINGS_LOCAL)[0]
            
            scores_cpu = sims.cpu().detach() 
            top_scores, top_indices = torch.topk(scores_cpu, min(top_k, scores_cpu.size(0)))
            
            for score, idx in zip(top_scores, top_indices):
                if score.item() >= threshold:
                    matched_idx.add(int(idx.item()))
            
            # Explicitly delete the comparison tensor variables
            del sims
            del scores_cpu
            
        except Exception as e:
            print(f"Embedding error: {e}")
        
        # Keyword fallback
        for i, sname in enumerate(SKILL_NAMES_LOWER_LOCAL):
            tokens = [t for t in re.split(r'[\s/]+', sname) if len(t) > 2]
            if sname in chunk or any(tok in chunk for tok in tokens[:3]):
                matched_idx.add(i)

    # Final cleanup of all tensors explicitly created
    for t in tensors_to_delete:
        del t
    
    # Return results using local dataframes and columns
    return [{
        "skill_name": skills_df_local.iloc[i][skill_name_col_local],
        "skill_category": skills_df_local.iloc[i]["skill_category"],
        "skill_description": skills_df_local.iloc[i]["skill_description"],
        "is_ai_recognizable": int(skills_df_local.iloc[i]["is_ai_recognizable"]),
        "identified_proficiency_level": str(skills_df_local.iloc[i]["identified_proficiency_level"])
    } for i in sorted(list(matched_idx))]


# ----- Routes -----
from flask import render_template

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/upload_cv", methods=["POST"])
def upload_cv_route():
    """
    Handles file upload, analysis, temporary file cleanup, 
    and model memory cleanup for independent runs.
    """
    if "cv" not in request.files:
        return jsonify({"error": "Missing file (key 'cv') in the request"}), 400

    file = request.files["cv"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Must be PDF, DOCX, or TXT."}), 400

    # 1. Setup temporary environment
    temp_dir = tempfile.mkdtemp()
    filename = secure_filename(file.filename)
    filepath = os.path.join(temp_dir, filename)
    
    try:
        # Ensure model is initialized for this worker
        get_isolated_resources()

        # 2. Save the file
        file.save(filepath)

        # 3. Extract text and analyze
        text = extract_text_from_file(filepath)
        if not text:
            return jsonify({"skills": [], "message": "Could not extract text from file."}), 200

        skills = extract_skills_enhanced(text)
        return jsonify({"skills": skills})
        
    except Exception as e:
        print(f"Analysis or I/O Error: {e}")
        return jsonify({"error": f"An internal server error occurred during analysis: {e}"}), 500
        
    finally:
        # 4. 🔥 EXTREME Model and Memory Cleanup (Destroying the local model instance) 🔥
        
        # A. Explicitly delete the PyTorch embeddings and model from the worker's local storage
        if hasattr(worker_data, 'embeddings') and worker_data.embeddings is not None:
            del worker_data.embeddings
        if hasattr(worker_data, 'model') and worker_data.model is not None:
            del worker_data.model
            
        # B. Clear the entire worker_data object state to force re-initialization on next request
        worker_data.__dict__.clear()
        
        # C. System-level cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        gc.collect() 
        
        # 5. Clean up the temporary file directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Failed to remove temp directory {temp_dir}: {e}")


# ----- Run App -----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)