import uuid
import re
import os
import shutil
import traceback
import pandas as pd
import sqlite3 
from flask import Flask, render_template, request, jsonify 
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import torch
import random 
import hashlib 
import json 
from datetime import datetime 

# Conditional import for DOCX parsing
try:
    from docx import Document
except ImportError:
    print("WARNING: 'python-docx' library not found. DOCX processing will fail. Install with 'pip install python-docx'")
    class Document:
        def __init__(self, *args, **kwargs):
            raise ImportError("python-docx is not installed.")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = "user_uploads"

SKILLS_FILE = os.path.join(BASE_DIR, "skills.csv") 
ROLES_FILE = os.path.join(BASE_DIR, "roles.csv")
DB_FILE = os.path.join(BASE_DIR, "verified_skills.db") 

MATCH_THRESHOLD = 0.40             
EVIDENCE_VERIFICATION_THRESHOLD = 0.45 
CONTEXT_WINDOW_SIZE = 3            

MODEL_NAME = "all-mpnet-base-v2"
DEVICE = 'cpu'

# --- PERMANENT FIX: HARDCODED KEYWORD MAPPING ---
SKILL_ALIASES = {
    "Financial Accounting": ["VAT return", "tax return", "ledger", "balance sheet", "financial statement", "bookkeeping", "accounts payable", "accounts receivable", "trial balance", "reconciliation"],
    "Financial Reporting": ["financial statement", "monthly report", "year-end", "profit and loss", "balance sheet", "cash flow", "reporting standards", "IFRS", "GAAP"],
    "Corporate Tax": ["corporation tax", "tax return", "HMRC", "tax liability", "tax computation"],
    "Personal Taxation": ["self-assessment", "personal tax", "income tax", "tax return"],
    "Legal & Regulatory Compliance": ["compliance", "legislation", "HMRC", "companies house", "regulatory", "anti-money laundering", "AML", "KYC", "GDPR", "law", "legal"],
    "Data Security": ["confidentiality", "data protection", "secure database", "GDPR", "privacy", "sensitive information", "encryption", "backup"],
    "Risk Management": ["risk", "threat", "fraud", "audit", "compliance", "due diligence", "mitigation"],
    "Client Management": ["client", "customer", "portfolio", "relationship", "stakeholder", "account management", "advising"],
    "Financial Software Usage": ["Excel", "Sage", "Xero", "QuickBooks", "FreeAgent", "SAP", "Oracle", "accounting software", "payroll manager"],
    "Microsoft Excel": ["Excel", "pivot table", "vlookup", "formula", "spreadsheet", "macros", "VBA"],
    "Time Management": ["deadline", "schedule", "prioritise", "time management", "timely", "efficiency"],
    "Attention to Detail": ["accuracy", "detail", "precision", "error", "discrepancy", "meticulous"],
    "Business Knowledge": ["economics", "business", "commercial awareness", "industry knowledge", "market"],
    "Fraud Detection": ["fraud", "tampering", "anti-money laundering", "suspicious activity", "investigation"],
    "Project Management": ["project", "planning", "scrum", "agile", "stakeholder management", "deadline", "budget"],
    "Data Analysis": ["data", "analyse", "report", "dashboard", "SQL", "Python", "R", "statistics"]
}

app = Flask(__name__)
app.config["UPLOAD_ROOT"] = UPLOAD_ROOT
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# --- DATA STORE CLASS ---
class DataStore:
    def __init__(self):
        self.model = None
        self.skills_df = pd.DataFrame()
        self.roles_df = pd.DataFrame()
        self.skill_embeddings = None
        self.conn = None 
        self.cursor = None 
        self.is_initialized = False

    def initialize(self):
        if self.is_initialized: return
        print(f"Loading Model ({DEVICE})...")
        
        try:
            self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            self.skills_df = pd.read_csv(SKILLS_FILE, on_bad_lines='skip')
            self.skills_df.columns = self.skills_df.columns.str.strip()
            self.roles_df = pd.read_csv(ROLES_FILE, on_bad_lines='skip')
            self.roles_df.columns = self.roles_df.columns.str.strip()
        except FileNotFoundError as e:
            print(f"Error: Required CSV file not found: {e}.")
            return
        except Exception as e:
            print(f"Initialization Error: {e}")
            return

        if 'SkillName' not in self.skills_df.columns or 'SkillCategory' not in self.skills_df.columns:
            print("ERROR: skills.csv must contain 'SkillName' and 'SkillCategory' columns.")
            return

        if 'RoleTitle' not in self.roles_df.columns or 'RequiredSkills' not in self.roles_df.columns:
             print("ERROR: roles.csv must contain 'RoleTitle' and 'RequiredSkills' columns.")
             if 'RoleName' in self.roles_df.columns and 'RoleTitle' not in self.roles_df.columns:
                 self.roles_df.rename(columns={'RoleName': 'RoleTitle'}, inplace=True)
                 print("Renamed 'RoleName' to 'RoleTitle' in roles.csv data.")
             elif 'RoleTitle' not in self.roles_df.columns:
                 return
        
        # --- FIX: Ensure RoleTitle and RequiredSkills are strings and stripped ---
        self.roles_df['RoleTitle'] = self.roles_df['RoleTitle'].astype(str).str.strip()
        self.roles_df['RequiredSkills'] = self.roles_df['RequiredSkills'].astype(str).str.strip()
        print("Role titles and RequiredSkills cleaned and forced to string type.")
        # ------------------------------------------------------------------------
        
        # Initialize SQLite Database
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_verified_skills_table() 

        self.skills_df['embed_text'] = self.skills_df['SkillName'] + " " + self.skills_df['Optimized SkillDescription'].fillna("")
        self.skill_embeddings = self.model.encode(
            self.skills_df['embed_text'].tolist(), 
            convert_to_tensor=True, 
            device=DEVICE
        )
        self.is_initialized = True
        print("System Initialized Locally with SQLite Database.")
        
    def create_verified_skills_table(self):
        """Creates the table if it doesn't exist."""
        sql = """
        CREATE TABLE IF NOT EXISTS verified_skills (
            id INTEGER PRIMARY KEY,
            cv_hash TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            evidence_quote TEXT NOT NULL,
            verified_at TEXT NOT NULL,
            UNIQUE(cv_hash, skill_name) 
        );
        """
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            print("Verified skills table ensured.")
        except Exception as e:
            print(f"Error creating table: {e}")

    def get_verified_skills(self, cv_hash):
        """Retrieves verified skills and evidence for a specific CV hash."""
        sql = "SELECT skill_name, evidence_quote FROM verified_skills WHERE cv_hash = ?"
        self.cursor.execute(sql, (cv_hash,))
        
        verified_data = {}
        for skill_name, evidence_quote in self.cursor.fetchall():
            verified_data[skill_name] = {
                "evidence_quote": evidence_quote,
                "timestamp": "Stored in DB" 
            }
        return verified_data

    def save_verified_skill(self, cv_hash, skill_name, evidence_quote):
        """Inserts or updates a skill verification."""
        timestamp = datetime.now().isoformat()
        sql = """
        INSERT INTO verified_skills (cv_hash, skill_name, evidence_quote, verified_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cv_hash, skill_name) DO UPDATE SET
            evidence_quote = excluded.evidence_quote,
            verified_at = excluded.verified_at;
        """
        try:
            self.cursor.execute(sql, (cv_hash, skill_name, evidence_quote, timestamp))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving verification: {e}")
            return False

STATE = DataStore()

# --- UTILITIES (No change) ---
def extract_text_from_file(filepath):
    if filepath.lower().endswith('.docx'):
        try:
            document = Document(filepath) 
            full_text = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            return '\n'.join(full_text)
        except Exception:
            return ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                text = f.read()
            return text
        except Exception:
            return ""
    except Exception:
        return ""

def split_text_to_sentences(text):
    sentences = re.split(r'([.!?])\s*', text)
    sentence_list = []
    
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i]
        d = sentences[i+1] if i+1 < len(sentences) else ''
        sentence = (s + d).strip()
        
        if len(sentence.split()) > 40:
             sub_splits = re.split(r'[\n\r]+|\s*•\s*|\s*-{2,}\s*', sentence)
             for sub in sub_splits:
                 if sub.strip():
                     sentence_list.append(sub.strip())
        elif sentence:
            sentence_list.append(sentence)

    final_sentences = [s for s in sentence_list if len(s.split()) > 3 and len(s) > 10]
    return final_sentences

def create_context_windows(sentences, window_size):
    windows = []
    half_window = window_size // 2
    for i in range(len(sentences)):
        start = max(0, i - half_window)
        end = min(len(sentences), i + half_window + 1)
        window = " ".join(sentences[start:end])
        windows.append(window)
    return windows

def chunk_text(text, chunk_size=40):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    for line in lines:
        words = line.split()
        if not words: continue
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def extract_uk_postcode(text):
    postcode_pattern = r'([A-Za-z]{1,2}[0-9][A-Za-z0-9]?\s*[0-9][A-Za-z]{2})'
    match = re.search(postcode_pattern, text.upper(), re.IGNORECASE)
    if match: return match.group(0).strip()
    return None

def url_safe_query(text):
    return text.replace(' ', '+').replace('&', '%26').replace('=', '%3D')

def get_cv_hash(cv_text):
    """Generates a stable, unique hash for the CV text."""
    return hashlib.sha256(cv_text.strip().encode('utf-8')).hexdigest()

# --- google_search_grounding and GENERIC_BUSINESS_RESOURCES (No change) ---
GENERIC_BUSINESS_RESOURCES = {
    "course_1": {
        "category": "Soft Skill Course",
        "link_title": "Effective Communication and Client Engagement (LinkedIn Learning)",
        "url": "https://www.linkedin.com/learning/topics/communication",
        "description": "Courses focusing on active listening, negotiation, clear writing, and managing difficult stakeholder conversations.",
    },
    "course_2": {
        "category": "Productivity Course",
        "link_title": "Time Management Specialization (Coursera)",
        "url": "https://www.coursera.org/specializations/time-management-professional",
        "description": "Learn key strategies for prioritization, goal-setting, and overcoming procrastination to boost efficiency.",
    },
    "event_1": {
        "category": "Networking/Events",
        "link_title": "Local Business Mentoring Groups (Generic Search)",
        "url": "https://www.google.com/search?q=Local+business+mentoring+and+networking+groups+near+{postcode_formatted}",
        "description": "Search for local chambers of commerce or business development groups for networking near your area.",
    },
    "event_2": {
        "category": "Webinar/Conference",
        "link_title": "Industry-Specific Webinars (Eventbrite Search)",
        "url": "https://www.eventbrite.co.uk/d/online/professional--events/",
        "description": "A general platform for finding free or low-cost professional development webinars and virtual events.",
    },
    "volunteer_1": {
        "category": "Skill-Based Volunteering",
        "link_title": "Reach Volunteering UK",
        "url": "https://reachvolunteering.org.uk/",
        "description": "Find governance, finance, or operations roles in charities to gain experience in a professional environment.",
    },
    "volunteer_2": {
        "category": "Local Community Experience",
        "link_title": "NCVO - Find Volunteering Opportunities",
        "url": "https://www.ncvo.org.uk/volunteer-opportunities/",
        "description": "Search for local non-profit roles to apply soft skills (e.g., event planning, admin) in a real-world setting.",
    }
}
def google_search_grounding(skill_gap, postcode):
    postcode_formatted = postcode if postcode else 'LU1 2BQ'
    skill_lower = skill_gap.lower()
    
    course_resources = []
    event_resources = []
    volunteering_resources = []
    
    if "project management" in skill_lower or "scrum" in skill_lower:
        course_resources = [
            {
                "category": "Course: Certification (Beginner)",
                "link_title": "Google Project Management Professional Certificate",
                "url": "https://www.coursera.org/professional-certificates/google-project-management",
                "description": "A 6-month program. Learn core Project Management, Agile, and Scrum frameworks.",
            },
            {
                "category": "Course: Certification (Advanced)",
                "link_title": "Project Management Professional (PMP)® - PMI",
                "url": "https://www.pmi.org/certifications/project-management-pmp",
                "description": "The world's leading certification for experienced Project Managers (requires 3-5 years experience).",
            },
        ]
        event_resources = [
            {
                "category": "Event: Professional Body",
                "link_title": "APM Regional & Interest Networks (UK)",
                "url": f"https://www.apm.org.uk/community/search/?q={postcode_formatted}",
                "description": f"The Association for Project Management (APM) local groups near {postcode_formatted} for networking.",
            },
            {
                "category": "Event: Agile/Scrum Meetup",
                "link_title": "Search Local Tech/Agile Meetups",
                "url": f"https://www.meetup.com/find/?keywords={url_safe_query('Agile OR Scrum')}&location={postcode_formatted}",
                "description": "Find local Meetup groups focused on Agile, Scrum, or Kanban methodologies to learn and network.",
            },
        ]
        volunteering_resources = [
            {
                "category": "Volunteering: Practical Experience",
                "link_title": "Charity Project Coordinator Roles (Skills-Based)",
                "url": "https://www.charityjob.co.uk/volunteer-jobs/project-management",
                "description": "Find practical charity roles that require project planning, coordination, and stakeholder management.",
            },
            GENERIC_BUSINESS_RESOURCES["volunteer_1"],
        ]

    elif any(s in skill_lower for s in ["financial accounting", "financial reporting", "taxation", "corporate tax", "personal taxation"]):
        course_resources = [
            {
                "category": "Course: Professional Qualification",
                "link_title": "AAT (Association of Accounting Technicians) Qualifications",
                "url": "https://www.aat.org.uk/qualifications",
                "description": "Globally recognised practical qualifications for accounting and finance roles (bookkeeping, tax, ledger skills).",
            },
            {
                "category": "Course: Official Guide",
                "link_title": "HMRC Guidance and Online Training",
                "url": "https://www.gov.uk/government/organisations/hm-revenue-customs",
                "description": "Official resources and guidelines directly from HMRC regarding VAT, Corporation Tax, and Personal Taxation. Essential for compliance.",
            },
        ]
        event_resources = [
            {
                "category": "Event: Professional Body",
                "link_title": "ACCA/CIMA Local Events",
                "url": f"https://www.accaglobal.com/uk/en/member/member-support/events.html",
                "description": "Search events hosted by professional bodies like ACCA or CIMA for technical updates and networking.",
            },
            GENERIC_BUSINESS_RESOURCES["event_2"],
        ]
        volunteering_resources = [
            {
                "category": "Volunteering: Practical Finance",
                "link_title": "Treasurer or Bookkeeper for Local Charity",
                "url": "https://www.ncvo.org.uk/volunteer-opportunities/",
                "description": "Use NCVO to search for local charities needing finance or treasury support to practice real-world accounting.",
            },
            GENERIC_BUSINESS_RESOURCES["volunteer_1"],
        ]

    elif any(s in skill_lower for s in ["excel", "financial software usage", "sage", "xero", "quickbooks"]):
        course_resources = [
            {
                "category": "Course: Software Certification",
                "link_title": "Official Xero Advisor Certification",
                "url": "https://www.xero.com/uk/training/certification/",
                "description": "Free, certified training on modern cloud accounting software like Xero, essential for business accounting roles.",
            },
            {
                "category": "Course: Excel Mastery",
                "link_title": "Microsoft Excel: Basic to Advanced (Coursera Specialization)",
                "url": "https://www.coursera.org/specializations/excel-skills-business",
                "description": "Master advanced functions, Pivot Tables, Power Query, and VBA for automation and robust data analysis in Excel.",
            },
        ]
        event_resources = [
            {
                "category": "Event: Tech Workshop",
                "link_title": "Local Data/Excel Workshops",
                "url": f"https://www.eventbrite.co.uk/d/united-kingdom/business--events/?q=Excel+training",
                "description": "Search Eventbrite for local workshops or training sessions focused specifically on Excel or data tooling.",
            },
            GENERIC_BUSINESS_RESOURCES["event_2"],
        ]
        volunteering_resources = [
            {
                "category": "Volunteering: Data Admin",
                "link_title": "Data Entry or Admin roles using Spreadsheets",
                "url": "https://reachvolunteering.org.uk/",
                "description": "Find volunteer roles that involve managing data, reporting, or using spreadsheets for a local charity to practice Excel.",
            },
            GENERIC_BUSINESS_RESOURCES["volunteer_2"],
        ]

    else:
        course_resources = [
            GENERIC_BUSINESS_RESOURCES["course_1"],
            GENERIC_BUSINESS_RESOURCES["course_2"],
        ]
        event_resources = [
            GENERIC_BUSINESS_RESOURCES["event_1"],
            GENERIC_BUSINESS_RESOURCES["event_2"],
        ]
        volunteering_resources = [
            GENERIC_BUSINESS_RESOURCES["volunteer_1"],
            GENERIC_BUSINESS_RESOURCES["volunteer_2"],
        ]

    course_search = {
        "category": "Search: Courses/Certifications",
        "link_title": f"**Search Google** for **{skill_gap}** courses", 
        "url": f"https://www.google.com/search?q={url_safe_query(f'best accredited online course or certification for {skill_gap}')}", 
        "description": f"Click here to search Google for accredited professional certificates or mastery courses specific to **{skill_gap}**.",
    }
    
    event_search = {
        "category": "Search: Events/Conferences",
        "link_title": f"**Search Google** for **{skill_gap}** events", 
        "url": f"https://www.google.com/search?q={url_safe_query(f'local {skill_gap} networking events or conferences near {postcode_formatted}')}",
        "description": f"Search for local industry events, professional meetups, or conferences related to **{skill_gap}** near your area.",
    }

    volunteering_search = {
        "category": "Search: Volunteering/Practice",
        "link_title": f"**Search Google** for **{skill_gap}** practice", 
        "url": f"https://www.google.com/search?q={url_safe_query(f'skill-based volunteering or practical experience for {skill_gap} near {postcode_formatted}')}", 
        "description": f"Search for local volunteering, side projects, or practice opportunities to develop hands-on **{skill_gap}** experience.",
    }

    final_resources = []
    final_resources.extend(course_resources[:2])
    final_resources.append(course_search)
    final_resources.extend(event_resources[:2])
    final_resources.append(event_search)
    final_resources.extend(volunteering_resources[:2])
    final_resources.append(volunteering_search)
    
    return final_resources

# --- ANALYSIS LOGIC (check_for_evidence and extract_general_skills remain the same) ---
def check_for_evidence(skill_name, cv_text, force_semantic_only=False):
    if not STATE.is_initialized:
        return False, "Model not initialized."
        
    sentences = split_text_to_sentences(cv_text)
    if not sentences: return False, "No readable text."

    if not force_semantic_only and skill_name in SKILL_ALIASES:
        keywords = SKILL_ALIASES[skill_name]
        for sentence in sentences:
            if any(k.lower() in sentence.lower() for k in keywords):
                return True, sentence  

    try:
        skill_row = STATE.skills_df[STATE.skills_df['SkillName'] == skill_name]
        rich_skill_text = skill_row.iloc[0]['embed_text'] if not skill_row.empty else skill_name
        
        skill_emb = STATE.model.encode(rich_skill_text, convert_to_tensor=True, device=DEVICE)
        
        context_windows = create_context_windows(sentences, CONTEXT_WINDOW_SIZE)
        if not context_windows: context_windows = sentences 

        window_embs = STATE.model.encode(context_windows, convert_to_tensor=True, device=DEVICE)
        
        cosine_scores = util.cos_sim(skill_emb, window_embs)[0]
        top_result = torch.topk(cosine_scores, k=1)
        best_score = top_result.values[0].item()
        best_index = top_result.indices[0].item()
        
        best_quote = context_windows[best_index]

        if best_score >= EVIDENCE_VERIFICATION_THRESHOLD:
            return True, best_quote
        else:
            return False, f"{best_quote} (Score: {best_score:.2f})"

    except Exception as e:
        return False, "Processing error."

def extract_general_skills(cv_text, cv_hash):
    if not STATE.is_initialized:
        return {"technical": [], "soft": [], "other": []}, [], set()
        
    extracted = []
    extracted_names_set = set() 
    
    # NEW LOGIC 1: Check for user-verified skills (Overrides NLP/Keyword detection)
    verified_skills = STATE.get_verified_skills(cv_hash)
    
    if verified_skills:
        for skill_name, data in verified_skills.items():
            skill_row = STATE.skills_df[STATE.skills_df['SkillName'] == skill_name]
            cat = skill_row.iloc[0]['SkillCategory'] if not skill_row.empty else "User Verified"

            evidence_chunk = data.get("evidence_quote", "User explicitly verified.") 

            extracted.append({
                "skill_name": skill_name,
                "skill_category": cat,
                "match_score": 1.0, 
                "evidence_chunk": evidence_chunk 
            })
            extracted_names_set.add(skill_name.lower())
    
    # NEW LOGIC 2: Proceed with NLP analysis for unverified skills
    cv_chunks = chunk_text(cv_text, chunk_size=40) 
    if not cv_chunks: return {"technical": [], "soft": [], "other": []}, extracted, extracted_names_set

    cv_embeddings = STATE.model.encode(cv_chunks, convert_to_tensor=True, device=DEVICE)
    cosine_scores = util.cos_sim(STATE.skill_embeddings, cv_embeddings)
    max_scores_per_skill, _ = torch.max(cosine_scores, dim=1)
    
    for idx, score in enumerate(max_scores_per_skill):
        skill_name = STATE.skills_df.iloc[idx]['SkillName']
        skill_category = STATE.skills_df.iloc[idx]['SkillCategory']
        
        if skill_name.lower() in extracted_names_set: 
            continue
            
        is_keyword_match = False
        if skill_name in SKILL_ALIASES:
             is_keyword_match = any(alias.lower() in cv_text.lower() for alias in SKILL_ALIASES[skill_name])

        if score.item() >= MATCH_THRESHOLD or is_keyword_match:
            extracted.append({
                "skill_name": skill_name,
                "skill_category": skill_category,
                "match_score": float(score),
                "evidence_chunk": "Matched via conceptual semantic analysis" 
            })
            extracted_names_set.add(skill_name.lower())
            
    categorized = {"technical": [], "soft": [], "other": []}
    for item in extracted:
        cat_lower = str(item['skill_category']).lower()
        if any(x in cat_lower for x in ['tech', 'data', 'programming', 'software', 'design']):
            categorized['technical'].append(item['skill_name'])
        elif any(x in cat_lower for x in ['soft', 'communication', 'leadership']):
            categorized['soft'].append(item['skill_name'])
        else:
            categorized['other'].append(item['skill_name'])
            
    for k in categorized:
        categorized[k] = sorted(list(set(categorized[k])))
        
    return categorized, extracted, extracted_names_set

def perform_role_gap_analysis(cv_text, role_name, already_extracted_set, cv_hash):
    if not STATE.is_initialized:
        return None, [], []
        
    cleaned_role_name = role_name.strip()
    print(f"DEBUG: Role lookup requested for: '{cleaned_role_name}'")
    
    # 1. Try case-insensitive substring match 
    matched_row = STATE.roles_df[STATE.roles_df['RoleTitle'].str.contains(cleaned_role_name, case=False, na=False)]
    
    # 2. If substring match fails, try exact match
    if matched_row.empty: 
        matched_row = STATE.roles_df[STATE.roles_df['RoleTitle'] == cleaned_role_name]
        
        if matched_row.empty:
            print(f"DEBUG: Role lookup FAILED for '{cleaned_role_name}'. The name was not found in roles.csv.")
            return None, [], []
    
    target_role_title = matched_row.iloc[0]['RoleTitle']
    print(f"DEBUG: Role '{target_role_title}' successfully matched.")
    
    # Get and clean required skills list
    req_string = matched_row.iloc[0]['RequiredSkills']
    required_skills_list = [s.strip() for s in str(req_string).split(',') if s.strip()]
    
    print(f"DEBUG: Found {len(required_skills_list)} required skills for this role: {required_skills_list[:3]}...")

    if not required_skills_list:
        print("DEBUG: WARNING: RequiredSkills list is empty for this matched role.")
        return target_role_title, [], []

    matches = []
    gaps = []
    
    cv_chunks = chunk_text(cv_text, chunk_size=40) 
    if not cv_chunks: return target_role_title, [], required_skills_list

    cv_embeddings = STATE.model.encode(cv_chunks, convert_to_tensor=True, device=DEVICE)
    
    for req_skill in required_skills_list:
        # 1. Check if skill was already extracted (via general analysis or verification)
        if req_skill.lower() in already_extracted_set:
            matches.append(req_skill)
            continue 

        # 2. Check hardcoded keyword aliases
        if req_skill in SKILL_ALIASES:
            if any(alias.lower() in cv_text.lower() for alias in SKILL_ALIASES[req_skill]):
                matches.append(req_skill)
                continue
        
        # 3. Perform Semantic Match (Encapsulated in try/except)
        try:
            req_embed = STATE.model.encode(req_skill, convert_to_tensor=True, device=DEVICE)
            scores = util.cos_sim(req_embed, cv_embeddings)[0]
            best_score = torch.max(scores).item()
            
            if best_score >= MATCH_THRESHOLD:
                matches.append(req_skill)
            else:
                gaps.append(req_skill)
                
        except Exception as e:
            # Fallback: If the skill is so domain-specific it can't be encoded, assume it is a gap
            # unless a keyword match was found (handled above).
            print(f"DEBUG: Semantic analysis failed for skill '{req_skill}'. Assuming gap. Error: {e}")
            gaps.append(req_skill)
            
    return target_role_title, matches, gaps

# --- FLASK ROUTES START ---

@app.route("/")
def index():
    STATE.initialize()
    return render_template('index.html')

@app.route("/verify_skill_and_store", methods=["POST"])
def verify_skill_and_store():
    if not STATE.is_initialized: 
        return jsonify({"status": "error", "message": "NLP Model is not ready."}), 503
    
    data = request.json
    skill_name = data.get("skill_name")
    user_evidence = data.get("user_provided_evidence")
    cv_text = data.get("cv_text") 

    if not all([skill_name, user_evidence, cv_text]):
        return jsonify({"status": "error", "message": "Missing skill name, evidence, or CV text."}), 400
    
    cv_hash = get_cv_hash(cv_text)
    
    evidence_to_store = f"User explicitly verified: {user_evidence}"
    
    if STATE.save_verified_skill(cv_hash, skill_name, evidence_to_store):
        return jsonify({"status": "success", "message": f"Skill '{skill_name}' verified and stored in DB."})
    else:
        return jsonify({"status": "error", "message": "Failed to store verification in database."}), 500

@app.route("/get_evidence", methods=["POST"])
def get_evidence():
    if not STATE.is_initialized: return jsonify({"error": "NLP Model is not ready."}), 503
    data = request.json
    skill_name = data.get("skill_name", "")
    cv_text = data.get("cv_text", "")
    if not skill_name or not cv_text: return jsonify({"error": "Missing skill name or CV text."}), 400

    is_found, quote_or_error = check_for_evidence(skill_name, cv_text)
    
    is_inferred = not is_found or "(Score:" in quote_or_error

    if is_found:
        return jsonify({"skill": skill_name, "evidence": [quote_or_error], "is_inferred": is_inferred})
    else:
        return jsonify({"skill": skill_name, "evidence": [f"Could not find strong sentence evidence. NLP Result: {quote_or_error}"], "is_inferred": True})

@app.route("/get_learning_resources", methods=["POST"])
def get_learning_resources():
    if not STATE.is_initialized: return jsonify({"error": "NLP Model is not ready."}), 503
    data = request.json
    skill_gap = data.get("skill_name", "")
    cv_text = data.get("cv_text", "")
    if not skill_gap or not cv_text: return jsonify({"error": "Missing data."}), 400
    
    postcode = extract_uk_postcode(cv_text)
    search_postcode = postcode if postcode else 'LU1 2BQ'
    
    resources = google_search_grounding(skill_gap, search_postcode)
    
    return jsonify({"skill": skill_gap, "postcode_used": search_postcode, "resources": resources})

@app.route("/analyze", methods=["POST"])
def analyze():
    if not STATE.is_initialized: 
        return jsonify({"error": "NLP Model is not ready. Please try again."}), 503
    
    # 1. --- GET INPUTS (Using request.files and request.form for POST multipart/form-data) ---
    cv_file = request.files.get('cv_file')
    target_role_select = request.form.get('target_role_select')
    job_description = request.form.get('job_description', '').strip() # Not currently used in analysis, but received
    job_description_url = request.form.get('job_description_url', '').strip() # Not currently used in analysis, but received
    
    mock_cv_text = """
    GHULAM HANIF
    07533452638 G_hanif@hotmail.co.uk
    Luton LU1 2BQ
    INTRODUCTORY PROFILE
    Analytically minded Accounts Assistant with admin experience, whose enhanced ability to critically evaluate accounting concepts and principles and their application in solutions to practical accounting problems while preparing financial statements of entities, includ
    """
    
    cv_text = ""
    file_path = None
    cv_hash = ""
    
    try:
        # 2. --- HANDLE FILE UPLOAD AND TEXT EXTRACTION ---
        if cv_file and cv_file.filename:
            filename = secure_filename(cv_file.filename)
            unique_folder = str(uuid.uuid4())
            upload_dir = os.path.join(app.config["UPLOAD_ROOT"], unique_folder)
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            cv_file.save(file_path)
            
            cv_text = extract_text_from_file(file_path)
        
        if not cv_text and not job_description and not job_description_url:
            cv_text = mock_cv_text
            
        if not cv_text:
             return jsonify({
                "error": "File parsing failed or no readable content found.", 
                "cv_text_for_evidence_retrieval": "",
                "cv_hash": ""
            }), 400

        # 3. --- CORE ANALYSIS SETUP ---
        role_name = target_role_select if target_role_select else "General Skills Assessment"
        cv_hash = get_cv_hash(cv_text)
        
        # 4. --- EXTRACT GENERAL SKILLS ---
        categorized_skills, all_skill_details, extracted_skill_names_set = extract_general_skills(cv_text, cv_hash)

        # 5. --- ROLE GAP ANALYSIS ---
        skills_matched = []
        skills_gap = []
        target_role_title = "General Skills Assessment"
        match_score = 0
        
        if target_role_select:
             target_role_title, skills_matched, skills_gap = perform_role_gap_analysis(
                cv_text, 
                target_role_select, 
                extracted_skill_names_set,
                cv_hash
            )
             
             # If role lookup fails (target_role_title is None), ensure we don't crash
             if target_role_title is None:
                 target_role_title = target_role_select + " (Lookup Failed - Showing General Skills Only)"
                 skills_matched = []
                 skills_gap = []

             total_required = len(skills_matched) + len(skills_gap)
             if total_required > 0:
                 match_score = round((len(skills_matched) / total_required) * 100)
        
        # 6. --- COMPILE RESULTS ---
        category_counts = {k: len(v) for k, v in categorized_skills.items()}

        response_data = {
            "cv_text_for_evidence_retrieval": cv_text,
            "cv_hash": cv_hash, 
            "extracted_skills": categorized_skills,
            "all_skills_details": all_skill_details,
            "category_counts": category_counts,
            
            "role_match": target_role_title,
            "match_score": match_score,
            "skills_matched": skills_matched,
            "skills_gap": skills_gap,
            "error": None
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Server Error during analysis: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "An unexpected error occurred on the server.", 
            "cv_text_for_evidence_retrieval": cv_text,
            "cv_hash": cv_hash 
        }), 500
        
    finally:
        # 7. --- CLEANUP ---
        if file_path and os.path.exists(file_path):
            try:
                shutil.rmtree(os.path.dirname(file_path)) 
            except Exception as e:
                print(f"Cleanup failed: {e}")
                
if __name__ == "__main__":
    STATE.initialize() 
    app.run(debug=True, port=5001, use_reloader=False)