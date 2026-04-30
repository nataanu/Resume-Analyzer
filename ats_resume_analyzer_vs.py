import subprocess
subprocess.run(["pip", "install", "pdfplumber", "python-docx", "nltk"])
"""
╔══════════════════════════════════════════════════════════════════╗
║          ATS Resume Analyzer  —  Local / VS Code Edition        ║
║  Works on Windows, macOS, Linux with a display (not Colab)      ║
╚══════════════════════════════════════════════════════════════════╝

SETUP (run once in your terminal):
    pip install pdfplumber python-docx nltk

    Then in Python (one-time):
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')

RUN:
    python ats_resume_analyzer.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import re
import io

# ── Optional PDF / DOCX libraries ─────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── Optional NLP ──────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    HAS_NLTK   = True
except Exception:
    HAS_NLTK   = False
    LEMMATIZER = None
    STOP_WORDS = {
        "i","me","my","we","our","you","your","he","she","it","its","they","their",
        "what","which","who","this","that","these","those","am","is","are","was",
        "were","be","been","being","have","has","had","do","does","did","will",
        "would","shall","should","may","might","can","could","a","an","the","and",
        "but","or","nor","for","so","yet","both","either","neither","not","only",
        "own","same","than","too","very","just","because","as","until","while",
        "of","at","by","with","about","against","between","into","through",
        "during","before","after","above","below","to","from","in","out","on",
        "off","over","under","again","then","once","here","there","when","where",
        "why","how","all","each","every","more","most","other","some","such","no",
        "any","if","else","also","must","s","t","don","doesn","didn","won","isn",
        "aren","wasn","weren","haven","hadn","shouldn","wouldn","couldn","shan",
    }

# ══════════════════════════════════════════════════════════════════════════════
# SKILL TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════
SKILL_TAXONOMY = {
    "python","java","javascript","typescript","c++","c#","c","ruby","go","golang",
    "rust","swift","kotlin","scala","r","matlab","perl","php","bash","shell",
    "powershell","vba","dart","lua","haskell","elixir",
    "html","css","react","angular","vue","svelte","nextjs","nuxtjs","jquery",
    "bootstrap","tailwind","sass","less","webpack","vite","graphql","rest","soap",
    "api","ajax","json","xml",
    "django","flask","fastapi","spring","springboot","express","nodejs","laravel",
    "rails","asp.net","dotnet",".net","hibernate",
    "sql","mysql","postgresql","postgres","sqlite","mongodb","redis",
    "elasticsearch","cassandra","oracle","mssql","dynamodb","firebase","nosql",
    "mariadb","neo4j",
    "aws","azure","gcp","google cloud","docker","kubernetes","k8s","terraform",
    "ansible","jenkins","github actions","ci/cd","linux","unix","nginx","apache",
    "heroku","vercel","netlify",
    "machine learning","deep learning","nlp","computer vision","ai","tensorflow",
    "pytorch","keras","scikit-learn","sklearn","pandas","numpy","scipy",
    "matplotlib","seaborn","plotly","tableau","power bi","data analysis",
    "data science","data engineering","big data","spark","hadoop","kafka",
    "airflow","dbt","etl","mlops","llm",
    "git","github","gitlab","bitbucket","jira","confluence","notion","agile",
    "scrum","kanban","devops","tdd","bdd","microservices","oop","solid",
    "design patterns","mvc","rest api",
    "android","ios","react native","flutter","xamarin","ionic",
    "cybersecurity","penetration testing","owasp","ssl","oauth","jwt",
    "encryption","firewall","siem",
    "communication","leadership","teamwork","problem solving","critical thinking",
    "project management","time management","collaboration","analytical",
    "presentation","mentoring","documentation",
}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(filepath: str) -> str:
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(
                    p.extract_text() for p in pdf.pages if p.extract_text()
                )
            if text.strip():
                return text
        except Exception:
            pass

    if HAS_PYPDF2:
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
        except Exception as e:
            raise RuntimeError(f"PyPDF2 error: {e}")

    raise RuntimeError(
        "No PDF library found.\n"
        "Run:  pip install pdfplumber"
    )


def extract_text_from_docx(filepath: str) -> str:
    if not HAS_DOCX:
        raise RuntimeError(
            "python-docx not installed.\nRun:  pip install python-docx"
        )
    doc = DocxDocument(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Please use PDF or DOCX.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> list:
    text = text.lower()
    text = (text.replace("c++", "cplusplus")
                .replace("c#",  "csharp")
                .replace(".net","dotnet"))
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    if HAS_NLTK and LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — SKILL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_skills(raw_text: str) -> set:
    text_lower = raw_text.lower()
    return {
        skill for skill in SKILL_TAXONOMY
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_ats_score(resume_skills, jd_skills, resume_tokens, jd_tokens) -> dict:
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    extra   = resume_skills - jd_skills

    skill_score = (len(matched) / len(jd_skills) * 100) if jd_skills else 0.0

    rt = set(resume_tokens)
    jt = set(jd_tokens)
    common   = rt & jt
    kw_score = (len(common) / len(jt) * 100) if jt else 0.0

    composite = round(0.70 * skill_score + 0.30 * kw_score, 1)
    return {
        "ats_score":       composite,
        "skill_score":     round(skill_score, 1),
        "keyword_score":   round(kw_score, 1),
        "matched_skills":  sorted(matched),
        "missing_skills":  sorted(missing),
        "extra_skills":    sorted(extra),
        "common_tokens":   len(common),
        "total_jd_tokens": len(jt),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — SUGGESTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_suggestions(result: dict) -> list:
    score   = result["ats_score"]
    missing = result["missing_skills"]
    tips    = []

    if score >= 80:
        tips.append("✅ Excellent match! Mirror the JD's exact phrasing in your bullet points to maximise ATS pass-through.")
    elif score >= 60:
        tips.append("👍 Good match. A few targeted additions could push your score above 80. Add the missing skills you genuinely have.")
    elif score >= 40:
        tips.append("⚠ Moderate match. Consider a focused revision — use the JD's exact terminology throughout.")
    else:
        tips.append("❌ Low match. This role may require skills not yet listed. Consider upskilling or targeting better-aligned roles.")

    if missing:
        preview = ", ".join(missing[:12]) + ("…" if len(missing) > 12 else "")
        tips.append(f"📌 Add these {len(missing)} missing skill(s) if you have them:\n   {preview}")
    else:
        tips.append("🎯 You've covered all skills mentioned in the JD — great!")

    tips.append("📝 Use the exact keywords from the JD — many ATS engines match verbatim phrases.")
    tips.append("📊 Quantify achievements: 'Reduced latency by 35%' beats 'improved performance'.")
    tips.append("📄 Keep resume to 1–2 pages: Summary · Skills · Experience · Education · Projects.")
    if result["skill_score"] < result["keyword_score"]:
        tips.append("💡 Your general vocabulary overlaps well. Add a dedicated 'Technical Skills' section listing tools explicitly.")
    return tips


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_resume(filepath: str, job_description: str) -> dict:
    resume_raw = extract_text(filepath)
    if not resume_raw.strip():
        raise ValueError("Could not extract any text. Is the PDF scanned/image-only?")
    if not job_description.strip():
        raise ValueError("Job description is empty.")

    resume_tokens = preprocess_text(resume_raw)
    jd_tokens     = preprocess_text(job_description)
    resume_skills = extract_skills(resume_raw)
    jd_skills     = extract_skills(job_description)

    result = compute_ats_score(resume_skills, jd_skills, resume_tokens, jd_tokens)
    result["suggestions"]    = generate_suggestions(result)
    result["resume_word_ct"] = len(resume_raw.split())
    result["jd_word_ct"]     = len(job_description.split())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — TKINTER GUI  (works locally in VS Code)
# ══════════════════════════════════════════════════════════════════════════════

# ── Colour palette ─────────────────────────────────────────────────────────────
BG       = "#0F1117"
SURFACE  = "#1A1D27"
PANEL    = "#1E2235"
BORDER   = "#2A2F4A"
ACCENT   = "#4F8EF7"
PURPLE   = "#A78BFA"
GREEN    = "#34D399"
AMBER    = "#FBBF24"
RED      = "#F87171"
TEXT     = "#E2E8F0"
MUTED    = "#64748B"
CARD     = "#252A40"


def score_color(score: float) -> str:
    if score >= 75: return GREEN
    if score >= 50: return AMBER
    return RED


class Tooltip:
    """Simple hover tooltip for any widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text   = text
        self.tip    = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.tip, text=self.text,
            bg="#2E3350", fg=TEXT,
            font=("Segoe UI", 9),
            relief="flat", padx=8, pady=4
        ).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScoreRing(tk.Canvas):
    """Animated circular score ring drawn on a Canvas."""

    def __init__(self, master, size=130, **kw):
        super().__init__(master, width=size, height=size,
                         bg=PANEL, highlightthickness=0, **kw)
        self._size   = size
        self._score  = 0.0
        self._target = 0.0
        self._color  = BORDER
        self._draw()

    def _draw(self):
        s = self._size
        p = 12           # padding
        self.delete("all")

        # Background ring
        self.create_oval(p, p, s - p, s - p,
                         outline=BORDER, width=10, fill="")

        # Score arc
        if self._score > 0:
            extent = -(self._score / 100) * 359.9
            self.create_arc(p, p, s - p, s - p,
                            start=90, extent=extent,
                            outline=self._color, width=10,
                            style="arc")

        # Score text
        cx = cy = s // 2
        self.create_text(cx, cy - 8,
                         text=f"{self._score:.0f}%",
                         fill=self._color,
                         font=("Segoe UI", 18, "bold"))
        self.create_text(cx, cy + 14,
                         text="ATS Score",
                         fill=MUTED,
                         font=("Segoe UI", 8))

    def animate_to(self, target: float, color: str):
        self._target = target
        self._color  = color
        self._step()

    def _step(self):
        if self._score < self._target:
            self._score = min(self._score + 2, self._target)
            self._draw()
            self.after(16, self._step)
        else:
            self._score = self._target
            self._draw()

    def reset(self):
        self._score  = 0.0
        self._target = 0.0
        self._color  = BORDER
        self._draw()


class ATSApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ATS Resume Analyzer")
        self.geometry("1160x820")
        self.minsize(960, 700)
        self.configure(bg=BG)
        self.resizable(True, True)

        # Attempt to set a nice icon color on Windows taskbar
        try:
            self.iconbitmap(default="")
        except Exception:
            pass

        self._resume_path = ""
        self._analyzing   = False

        self._apply_styles()
        self._build_ui()
        self._check_deps()

    # ── ttk styles ─────────────────────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TFrame",        background=BG)
        s.configure("Card.TFrame",   background=PANEL)
        s.configure("Surface.TFrame",background=SURFACE)

        s.configure("TLabel",        background=BG, foreground=TEXT,
                    font=("Segoe UI", 10))
        s.configure("Muted.TLabel",  background=BG, foreground=MUTED,
                    font=("Segoe UI", 9))
        s.configure("Card.TLabel",   background=PANEL, foreground=TEXT,
                    font=("Segoe UI", 10))
        s.configure("Title.TLabel",  background=BG, foreground=TEXT,
                    font=("Segoe UI", 11, "bold"))

        # Primary action button
        s.configure("Primary.TButton",
                    background=ACCENT, foreground="#fff",
                    font=("Segoe UI", 10, "bold"),
                    relief="flat", padding=(18, 9))
        s.map("Primary.TButton",
              background=[("active", "#3B7AE0"), ("disabled", BORDER)],
              foreground=[("disabled", MUTED)])

        # Secondary / ghost button
        s.configure("Ghost.TButton",
                    background=PANEL, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"),
                    relief="flat", padding=(14, 9))
        s.map("Ghost.TButton",
              background=[("active", BORDER)])

        # Danger button
        s.configure("Danger.TButton",
                    background="#3B1515", foreground=RED,
                    font=("Segoe UI", 9),
                    relief="flat", padding=(10, 6))
        s.map("Danger.TButton",
              background=[("active", "#5a1e1e")])

        s.configure("TProgressbar",
                    troughcolor=PANEL, background=ACCENT,
                    bordercolor=PANEL, lightcolor=ACCENT,
                    darkcolor=ACCENT, thickness=6)

        s.configure("TNotebook",     background=BG, borderwidth=0)
        s.configure("TNotebook.Tab",
                    background=PANEL, foreground=MUTED,
                    font=("Segoe UI", 9, "bold"),
                    padding=(14, 7))
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT), ("active", BORDER)],
              foreground=[("selected", "#fff")])

    # ── Main layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        body = ttk.Frame(self, style="TFrame")
        body.pack(fill="both", expand=True, padx=14, pady=(10, 6))
        body.columnconfigure(0, weight=0, minsize=370)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=SURFACE, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⚡", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 20)).pack(side="left", padx=(18, 4), pady=10)
        tk.Label(hdr, text="ATS Resume Analyzer", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(side="left", pady=10)

        self._dep_lbl = tk.Label(hdr, text="", bg=SURFACE, fg=MUTED,
                                 font=("Segoe UI", 8))
        self._dep_lbl.pack(side="right", padx=16)

        # Reset button in header
        tk.Button(hdr, text="↺  Reset", bg=SURFACE, fg=MUTED,
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  activebackground=BORDER, activeforeground=TEXT,
                  command=self._reset).pack(side="right", padx=(0, 4), pady=12)

    def _build_left(self, parent):
        left = ttk.Frame(parent, style="TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(2, weight=1)

        # ── Upload card ────────────────────────────────────────────────────────
        self._section_label(left, "📄  Resume File").grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        upload_card = tk.Frame(left, bg=PANEL, bd=0)
        upload_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        inner = tk.Frame(upload_card, bg=PANEL)
        inner.pack(fill="x", padx=14, pady=10)

        self._file_icon = tk.Label(inner, text="📁", bg=PANEL, fg=MUTED,
                                   font=("Segoe UI", 22))
        self._file_icon.pack(side="left", padx=(0, 10))

        meta = tk.Frame(inner, bg=PANEL)
        meta.pack(side="left", fill="x", expand=True)

        self._file_name_lbl = tk.Label(meta, text="No file selected",
                                       bg=PANEL, fg=MUTED,
                                       font=("Segoe UI", 10, "bold"),
                                       anchor="w")
        self._file_name_lbl.pack(fill="x")

        self._file_size_lbl = tk.Label(meta, text="Click below to browse",
                                       bg=PANEL, fg=MUTED,
                                       font=("Segoe UI", 8),
                                       anchor="w")
        self._file_size_lbl.pack(fill="x")

        ttk.Button(upload_card, text="📂  Browse PDF / DOCX",
                   style="Ghost.TButton",
                   command=self._browse).pack(fill="x", padx=14, pady=(0, 10))

        # ── JD card ────────────────────────────────────────────────────────────
        self._section_label(left, "💼  Job Description").grid(
            row=2, column=0, sticky="nw", pady=(0, 4))

        jd_card = tk.Frame(left, bg=PANEL)
        jd_card.grid(row=3, column=0, sticky="nsew", pady=(0, 12))
        left.rowconfigure(3, weight=1)

        jd_frame = tk.Frame(jd_card, bg=PANEL)
        jd_frame.pack(fill="both", expand=True, padx=2, pady=2)

        self._jd = tk.Text(
            jd_frame, bg="#1A1D2E", fg=TEXT,
            insertbackground=ACCENT, selectbackground=ACCENT,
            font=("Segoe UI", 10), relief="flat", bd=0,
            padx=10, pady=10, wrap="word",
        )
        self._jd.pack(side="left", fill="both", expand=True)

        jd_sb = tk.Scrollbar(jd_frame, command=self._jd.yview,
                              bg=PANEL, troughcolor=PANEL,
                              activebackground=BORDER)
        jd_sb.pack(side="right", fill="y")
        self._jd.configure(yscrollcommand=jd_sb.set)

        # Placeholder
        self._PLACEHOLDER = "Paste the full job description here…"
        self._jd.insert("1.0", self._PLACEHOLDER)
        self._jd.config(fg=MUTED)
        self._jd.bind("<FocusIn>",  self._jd_focus_in)
        self._jd.bind("<FocusOut>", self._jd_focus_out)

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_row = ttk.Frame(left, style="TFrame")
        btn_row.grid(row=4, column=0, sticky="ew", pady=(0, 4))
        btn_row.columnconfigure(0, weight=1)

        self._analyze_btn = ttk.Button(btn_row, text="🔍  Analyze Resume",
                                       style="Primary.TButton",
                                       command=self._start)
        self._analyze_btn.grid(row=0, column=0, sticky="ew")
        Tooltip(self._analyze_btn, "Run the ATS skill-matching analysis")

        self._progress = ttk.Progressbar(left, mode="indeterminate",
                                         style="TProgressbar")
        # shown only during analysis

    def _build_right(self, parent):
        right = ttk.Frame(parent, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Score banner ───────────────────────────────────────────────────────
        banner = tk.Frame(right, bg=PANEL)
        banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Score ring
        self._ring = ScoreRing(banner, size=130)
        self._ring.pack(side="left", padx=16, pady=12)

        # Score meta
        meta = tk.Frame(banner, bg=PANEL)
        meta.pack(side="left", fill="both", expand=True, pady=12)

        self._score_var = tk.StringVar(value="—")
        self._band_var  = tk.StringVar(value="Upload a resume and paste a job description, then click Analyze.")

        tk.Label(meta, textvariable=self._score_var,
                 bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 30, "bold")).pack(anchor="w")
        tk.Label(meta, textvariable=self._band_var,
                 bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10),
                 wraplength=320, justify="left").pack(anchor="w", pady=(4, 0))

        # Sub-scores on right
        sub = tk.Frame(banner, bg=PANEL)
        sub.pack(side="right", padx=20, pady=12)

        self._skill_var = tk.StringVar(value="Skill match:    —")
        self._kw_var    = tk.StringVar(value="Keyword match:  —")
        self._wc_var    = tk.StringVar(value="")

        for var, col in [(self._skill_var, PURPLE),
                         (self._kw_var,    AMBER),
                         (self._wc_var,    MUTED)]:
            tk.Label(sub, textvariable=var, bg=PANEL, fg=col,
                     font=("Segoe UI", 10, "bold"),
                     anchor="e").pack(anchor="e", pady=2)

        # ── Notebook tabs ──────────────────────────────────────────────────────
        self._nb = ttk.Notebook(right)
        self._nb.grid(row=1, column=0, sticky="nsew")

        self._tab_matched  = self._make_tab("✅  Matched")
        self._tab_missing  = self._make_tab("❌  Missing")
        self._tab_suggest  = self._make_tab("💡  Suggestions")
        self._tab_detail   = self._make_tab("🔍  Full Report")

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready — load a resume to begin.")
        tk.Label(bar, textvariable=self._status_var,
                 bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w").pack(side="left", padx=12)

        # Dependency badge on right of status bar
        self._dep_badge = tk.Label(bar, text="", bg=SURFACE, fg=MUTED,
                                   font=("Segoe UI", 8))
        self._dep_badge.pack(side="right", padx=12)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _section_label(self, parent, text):
        return tk.Label(parent, text=text, bg=BG, fg=PURPLE,
                        font=("Segoe UI", 10, "bold"), anchor="w")

    def _make_tab(self, title: str) -> tk.Text:
        frame = tk.Frame(self._nb, bg=PANEL)
        self._nb.add(frame, text=title)

        t = tk.Text(frame, bg="#1A1D2E", fg=TEXT,
                    font=("Consolas", 10),
                    relief="flat", bd=0,
                    padx=14, pady=12, wrap="word",
                    state="disabled")
        t.pack(side="left", fill="both", expand=True)

        sb = tk.Scrollbar(frame, command=t.yview,
                          bg=PANEL, troughcolor=PANEL)
        sb.pack(side="right", fill="y")
        t.configure(yscrollcommand=sb.set)

        # Configure coloured tags
        t.tag_configure("heading",  foreground=ACCENT,  font=("Consolas", 10, "bold"))
        t.tag_configure("green",    foreground=GREEN)
        t.tag_configure("red",      foreground=RED)
        t.tag_configure("amber",    foreground=AMBER)
        t.tag_configure("purple",   foreground=PURPLE)
        t.tag_configure("muted",    foreground=MUTED)
        t.tag_configure("bold",     font=("Consolas", 10, "bold"))
        return t

    def _tab_write(self, widget: tk.Text, segments: list):
        """
        Write coloured segments to a tab.
        segments = list of (text, tag_or_None)
        """
        widget.config(state="normal")
        widget.delete("1.0", "end")
        for text, tag in segments:
            if tag:
                widget.insert("end", text, tag)
            else:
                widget.insert("end", text)
        widget.config(state="disabled")

    def _check_deps(self):
        parts = []
        if HAS_PDFPLUMBER:  parts.append("pdfplumber ✓")
        elif HAS_PYPDF2:    parts.append("PyPDF2 ✓")
        else:               parts.append("⚠ no PDF lib")
        if HAS_DOCX:        parts.append("python-docx ✓")
        else:               parts.append("⚠ no DOCX lib")
        parts.append("NLTK ✓" if HAS_NLTK else "NLTK ✗ (basic mode)")
        badge = "  |  ".join(parts)
        self._dep_badge.config(text=badge)

        if not HAS_PDFPLUMBER and not HAS_PYPDF2:
            messagebox.showwarning(
                "Missing Library",
                "No PDF library detected.\n\n"
                "Run in your terminal:\n"
                "   pip install pdfplumber\n\n"
                "Then restart this application."
            )

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _jd_focus_in(self, _):
        if self._jd.get("1.0", "end-1c") == self._PLACEHOLDER:
            self._jd.delete("1.0", "end")
            self._jd.config(fg=TEXT)

    def _jd_focus_out(self, _):
        if not self._jd.get("1.0", "end-1c").strip():
            self._jd.insert("1.0", self._PLACEHOLDER)
            self._jd.config(fg=MUTED)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Resume",
            filetypes=[
                ("Supported", "*.pdf *.docx *.doc"),
                ("PDF",       "*.pdf"),
                ("Word",      "*.docx *.doc"),
            ]
        )
        if path:
            self._resume_path = path
            fname    = os.path.basename(path)
            size_kb  = os.path.getsize(path) // 1024
            ext      = os.path.splitext(fname)[1].upper().lstrip(".")
            self._file_icon.config(text="📄" if ext == "PDF" else "📝", fg=GREEN)
            self._file_name_lbl.config(text=fname, fg=GREEN)
            self._file_size_lbl.config(text=f"{ext}  •  {size_kb} KB", fg=MUTED)
            self._status_var.set(f"Resume loaded: {fname}")

    def _reset(self):
        self._resume_path = ""
        self._file_icon.config(text="📁", fg=MUTED)
        self._file_name_lbl.config(text="No file selected", fg=MUTED)
        self._file_size_lbl.config(text="Click below to browse", fg=MUTED)
        self._jd.config(state="normal", fg=MUTED)
        self._jd.delete("1.0", "end")
        self._jd.insert("1.0", self._PLACEHOLDER)
        self._ring.reset()
        self._score_var.set("—")
        self._band_var.set("Upload a resume and paste a job description, then click Analyze.")
        self._skill_var.set("Skill match:    —")
        self._kw_var.set("Keyword match:  —")
        self._wc_var.set("")
        for tab in (self._tab_matched, self._tab_missing,
                    self._tab_suggest, self._tab_detail):
            self._tab_write(tab, [])
        self._status_var.set("Reset — ready for a new analysis.")

    def _start(self):
        if self._analyzing:
            return
        if not self._resume_path:
            messagebox.showwarning("No Resume", "Please browse and select a resume file.")
            return
        jd = self._jd.get("1.0", "end-1c").strip()
        if not jd or jd == self._PLACEHOLDER:
            messagebox.showwarning("No JD", "Please paste the job description.")
            return

        self._analyzing = True
        self._analyze_btn.state(["disabled"])
        self._progress.grid(row=5, column=0, sticky="ew",
                            in_=self._analyze_btn.master, pady=(6, 0))
        self._progress.start(10)
        self._status_var.set("Analyzing… please wait.")

        threading.Thread(
            target=self._worker,
            args=(self._resume_path, jd),
            daemon=True
        ).start()

    def _worker(self, path, jd):
        try:
            result = analyze_resume(path, jd)
            self.after(0, self._show, result)
        except Exception as exc:
            self.after(0, self._err, str(exc))

    # ── Results display ────────────────────────────────────────────────────────

    def _show(self, r: dict):
        score = r["ats_score"]
        col   = score_color(score)

        # Score ring animation
        self._ring.animate_to(score, col)

        # Score labels
        self._score_var.set(f"{score}%")
        if score >= 80:   band = "Excellent match 🏆"
        elif score >= 60: band = "Good match 👍"
        elif score >= 40: band = "Moderate match 🤔"
        else:             band = "Low match ⚠"
        self._band_var.set(band)

        self._skill_var.set(f"Skill match:     {r['skill_score']}%")
        self._kw_var.set(   f"Keyword match:  {r['keyword_score']}%")
        self._wc_var.set(
            f"Resume: {r['resume_word_ct']} words  |  JD: {r['jd_word_ct']} words"
        )

        # ── Tab: Matched ───────────────────────────────────────────────────────
        matched = r["matched_skills"]
        seg = [
            (f"✅  Matched Skills  ({len(matched)})\n", "heading"),
            ("─" * 52 + "\n\n", "muted"),
        ]
        if matched:
            for i, sk in enumerate(matched, 1):
                seg += [(f"  {i:>2}.  ", "muted"), (f"{sk}\n", "green")]
        else:
            seg += [("  No taxonomy skills matched. Add a dedicated Skills section.\n", "muted")]
        self._tab_write(self._tab_matched, seg)

        # ── Tab: Missing ───────────────────────────────────────────────────────
        missing = r["missing_skills"]
        seg = [
            (f"❌  Missing Skills  ({len(missing)})\n", "heading"),
            ("─" * 52 + "\n\n", "muted"),
        ]
        if missing:
            for i, sk in enumerate(missing, 1):
                seg += [(f"  {i:>2}.  ", "muted"), (f"{sk}\n", "red")]
        else:
            seg += [("  🎯  None — you've covered all JD skills!\n", "green")]
        self._tab_write(self._tab_missing, seg)

        # ── Tab: Suggestions ──────────────────────────────────────────────────
        seg = [("💡  Improvement Suggestions\n", "heading"),
               ("─" * 52 + "\n\n", "muted")]
        for i, tip in enumerate(r["suggestions"], 1):
            seg += [(f"{i}.  ", "purple"), (f"{tip}\n\n", None)]

        if missing:
            seg += [("\n🔑  Keywords to add:\n", "heading"),
                    ("─" * 52 + "\n", "muted")]
            kw_line = "  " + ",  ".join(missing[:20])
            if len(missing) > 20:
                kw_line += f"\n  … and {len(missing)-20} more"
            seg += [(kw_line + "\n", "amber")]
        self._tab_write(self._tab_suggest, seg)

        # ── Tab: Full Report ───────────────────────────────────────────────────
        extra = r.get("extra_skills", [])
        seg = [
            ("═" * 54 + "\n", "muted"),
            ("  ATS ANALYSIS REPORT\n", "heading"),
            ("═" * 54 + "\n\n", "muted"),
            ("  Composite ATS Score : ", "muted"), (f"{r['ats_score']}%\n", "bold"),
            ("  Skill match score   : ", "muted"), (f"{r['skill_score']}%\n", "purple"),
            ("  Keyword match score : ", "muted"), (f"{r['keyword_score']}%\n", "amber"),
            ("\n", None),
            ("  Resume words        : ", "muted"), (f"{r['resume_word_ct']}\n", None),
            ("  JD words            : ", "muted"), (f"{r['jd_word_ct']}\n", None),
            ("  JD unique tokens    : ", "muted"), (f"{r['total_jd_tokens']}\n", None),
            ("  Common tokens       : ", "muted"), (f"{r['common_tokens']}\n", None),
            ("\n", None),
            ("  Matched skills  : ", "muted"),
            ((", ".join(matched) if matched else "none") + "\n", "green"),
            ("\n", None),
            ("  Missing skills  : ", "muted"),
            ((", ".join(missing) if missing else "none") + "\n", "red"),
            ("\n", None),
            ("  Extra skills    : ", "muted"),
            ((", ".join(extra)   if extra   else "none") + "\n", "purple"),
            ("\n", None),
            ("─" * 54 + "\n", "muted"),
            ("  NLP  : ", "muted"),
            (("NLTK (lemmatization enabled)" if HAS_NLTK else "Built-in (basic mode)") + "\n", None),
            ("  PDF  : ", "muted"),
            (("pdfplumber" if HAS_PDFPLUMBER else "PyPDF2" if HAS_PYPDF2 else "⚠ none") + "\n", None),
            ("═" * 54 + "\n", "muted"),
        ]
        self._tab_write(self._tab_detail, seg)

        # Switch to suggestions tab automatically
        self._nb.select(2)

        # Unlock
        self._done(
            f"✅  Done — ATS Score: {score}%  |  "
            f"{len(matched)} matched, {len(missing)} missing skills."
        )

    def _err(self, msg: str):
        self._done("❌  Analysis failed.")
        messagebox.showerror("Analysis Error", msg)

    def _done(self, status: str):
        self._progress.stop()
        self._progress.grid_remove()
        self._analyze_btn.state(["!disabled"])
        self._analyzing = False
        self._status_var.set(status)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ATSApp()
    app.mainloop()
ś