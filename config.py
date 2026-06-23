"""
Redrob AI Candidate Ranking — Central Configuration
====================================================
All tunable weights, thresholds, skill taxonomies, company lists,
and default values live here. Nothing is hardcoded elsewhere.
"""

# ============================================================================
# Final Score Composition
# ============================================================================
SEMANTIC_WEIGHT = 0.30    # Captures plain-language strong candidates
RULE_BASED_WEIGHT = 0.50  # Enforces JD's explicit requirements
BEHAVIORAL_WEIGHT = 0.20  # Platform engagement & reachability modifier

# ============================================================================
# Behavioral Signal Multiplier
# ============================================================================
BEHAVIORAL_FLOOR = 0.5     # Don't crush strong candidates who are inactive
BEHAVIORAL_CEILING = 1.2   # Reward highly engaged candidates
BEHAVIORAL_RANGE = BEHAVIORAL_CEILING - BEHAVIORAL_FLOOR  # 0.7

# Individual signal weights within the behavioral composite (sum → 1.0)
BEHAVIORAL_SIGNAL_WEIGHTS = {
    "recruiter_response_rate":   0.20,
    "avg_response_time":         0.10,
    "last_active_recency":       0.15,
    "open_to_work":              0.10,
    "profile_completeness":      0.05,
    "interview_completion_rate": 0.10,
    "notice_period":             0.10,
    "github_activity":           0.10,
    "verification":              0.05,
    "saved_by_recruiters":       0.05,
}

# ============================================================================
# Experience Thresholds
# ============================================================================
MIN_EXPERIENCE_YEARS = 2.0   # Hard filter — below this, eliminate
MAX_EXPERIENCE_YEARS = 20.0  # Above this + no ML titles → eliminate
IDEAL_EXPERIENCE_MIN = 5.0
IDEAL_EXPERIENCE_MAX = 9.0

# ============================================================================
# Embedding Model  (dense retrieval — precomputed offline, loaded on CPU)
# ============================================================================
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 256
# BGE retrieval instruction prefix (applied to the JD query, NOT to documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ============================================================================
# Cross-Encoder Re-ranking  (Stage 5.5 — top-of-funnel precision)
# ============================================================================
# Re-ranks only a small shortlist, so it fits the CPU/5-min ranking budget.
# Runs only if the model weights are available locally (no network at rank
# time); otherwise the pipeline gracefully falls back to the dense+rule order.
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_SHORTLIST = 200          # How many top candidates to re-rank
RERANK_MAX_LENGTH = 512         # Cross-encoder truncation length
RERANK_BATCH_SIZE = 32
RERANK_WEIGHT = 0.6             # Blend: 0.6*cross-encoder + 0.4*base score

# ============================================================================
# Skill Categories
# ============================================================================

# --- Core skills the JD explicitly requires (highest weight) ---
CORE_AI_SKILLS = {
    # Embeddings & retrieval
    "sentence-transformers", "sentence transformers", "embeddings",
    "semantic search", "vector search", "dense retrieval", "hybrid search",
    "bm25", "information retrieval", "search systems",
    # Vector databases
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "vector database",
    # ML/AI fundamentals
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "nlp", "natural language processing",
    "machine learning", "deep learning", "neural networks",
    "ranking", "ranking systems", "recommendation systems",
    "recommendations", "search ranking", "learning to rank",
    "learning-to-rank",
    # Python
    "python",
    # Evaluation
    "ndcg", "mrr", "a/b testing", "evaluation",
    "evaluation frameworks", "offline evaluation",
}

# --- Nice-to-have skills (medium weight) ---
NICE_TO_HAVE_SKILLS = {
    # LLM & fine-tuning
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms",
    "llm", "large language models", "gpt", "bert", "transformers",
    "hugging face", "huggingface",
    # ML tools
    "xgboost", "lightgbm", "gradient boosting",
    "weights & biases", "wandb", "mlflow", "experiment tracking",
    "bentoml", "torchserve", "model serving",
    "feature engineering", "statistical modeling",
    # Data engineering
    "spark", "pyspark", "airflow", "data pipeline", "data pipelines",
    "kafka", "redis", "postgresql", "mongodb",
    "apache beam", "apache flink", "databricks",
    # Infrastructure
    "distributed systems", "kubernetes", "docker",
    "aws", "gcp", "azure", "cloud",
    # Adjacent AI
    "rag", "retrieval augmented generation",
    "object detection", "image classification", "computer vision",
    "gans", "generative ai", "tts", "speech recognition", "asr",
    # General engineering
    "sql", "data analysis", "data science",
    "flask", "fastapi", "django",
    "git", "ci/cd", "mlops",
}

# --- Red-flag skills indicating non-tech / irrelevant career ---
RED_FLAG_SKILLS = {
    "photoshop", "illustrator", "indesign", "figma",
    "seo", "content writing", "marketing", "brand design",
    "accounting", "tally", "quickbooks", "sap",
    "six sigma", "supply chain", "lean manufacturing",
    "solidworks", "autocad", "creo", "ansys", "catia",
    "powerpoint",
}

# ============================================================================
# Title Relevance Scores  (0 → 15)
# ============================================================================
TITLE_RELEVANCE = {
    # Highly relevant (12-15)
    "ml engineer": 15,
    "machine learning engineer": 15,
    "senior machine learning engineer": 15,
    "junior ml engineer": 12,
    "ai engineer": 15,
    "senior ai engineer": 15,
    "data scientist": 13,
    "senior data scientist": 14,
    "lead data scientist": 14,
    "nlp engineer": 14,
    "research engineer": 11,
    "applied scientist": 13,
    "research scientist": 10,
    # Medium relevance (4-10)
    "data engineer": 8,
    "senior data engineer": 9,
    "backend engineer": 7,
    "senior backend engineer": 8,
    "software engineer": 6,
    "senior software engineer": 7,
    "full stack engineer": 5,
    "platform engineer": 6,
    "devops engineer": 4,
    "analytics engineer": 7,
    "data analyst": 5,
    # Low / irrelevant — trap profiles (0-2)
    "marketing manager": 0,
    "hr manager": 0,
    "accountant": 0,
    "sales executive": 0,
    "operations manager": 0,
    "customer support": 0,
    "content writer": 1,
    "graphic designer": 0,
    "civil engineer": 1,
    "mechanical engineer": 1,
    "business analyst": 2,
    "project manager": 2,
}

# ============================================================================
# Consulting Firms  (entire-career-only = JD disqualifier)
# ============================================================================
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture",
    "hcl technologies", "hcl", "tech mahindra", "mphasis",
    "ltimindtree", "lti", "mindtree", "hexaware",
    "deloitte", "ey", "ernst & young", "kpmg", "pwc",
    "pricewaterhousecoopers",
}

# ============================================================================
# Location Preferences
# ============================================================================
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "new delhi", "delhi ncr", "ncr",
    "gurgaon", "gurugram", "ghaziabad", "faridabad",
}
ACCEPTABLE_INDIAN_CITIES = {
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata",
}
INDIA_COUNTRY = "india"

# ============================================================================
# Notice Period
# ============================================================================
IDEAL_NOTICE_DAYS = 30
MAX_ACCEPTABLE_NOTICE_DAYS = 90

# ============================================================================
# Default Values for Missing / Null Fields
# ============================================================================
DEFAULTS = {
    "years_of_experience":        0.0,
    "current_title":              "Unknown",
    "current_company":            "Unknown",
    "current_industry":           "Unknown",
    "country":                    "Unknown",
    "location":                   "Unknown",
    "willing_to_relocate":        False,
    "open_to_work_flag":          False,
    "recruiter_response_rate":    0.0,
    "avg_response_time_hours":    168.0,   # 1 week
    "profile_completeness_score": 50.0,
    "interview_completion_rate":  0.5,
    "notice_period_days":         90,
    "github_activity_score":      -1,
    "last_active_date":           "2025-01-01",
    "signup_date":                "2025-01-01",
    "connection_count":           0,
    "endorsements_received":      0,
    "saved_by_recruiters_30d":    0,
    "profile_views_received_30d": 0,
    "search_appearance_30d":      0,
    "applications_submitted_30d": 0,
    "verified_email":             False,
    "verified_phone":             False,
    "linkedin_connected":         False,
    "preferred_work_mode":        "flexible",
    "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
    "skill_assessment_scores":    {},
}

# ============================================================================
# Score Normalization
# ============================================================================
MIN_SCORE_SPREAD = 0.15  # Apply normalization if range < this
NORMALIZATION_TARGET_MIN = 0.10
NORMALIZATION_TARGET_MAX = 0.99

# ============================================================================
# Rule-Based Sub-Score Ceilings
# ============================================================================
TITLE_CAREER_MAX_PTS = 30
SKILL_RELEVANCE_MAX_PTS = 25
EXPERIENCE_DEPTH_MAX_PTS = 15
EDUCATION_MAX_PTS = 10
LOCATION_MAX_PTS = 10
CAREER_PATTERN_PENALTY_MAX = -10   # Negative — deductions only
RULE_BASED_TOTAL_MAX = 100         # Normaliser denominator

# ============================================================================
# Parallelism
# ============================================================================
N_JOBS = -1         # All CPU cores for joblib
BATCH_SIZE = 500    # Progress-bar granularity

# ============================================================================
# Non-Technical Titles  (used for hard filtering)
# ============================================================================
NON_TECH_TITLES = {
    "accountant", "hr manager", "sales executive",
    "operations manager", "customer support",
    "marketing manager", "graphic designer",
    "content writer", "civil engineer", "mechanical engineer",
}

# Titles that indicate AI/ML relevance (used in the MAX_EXPERIENCE filter)
ML_ADJACENT_TITLES = {
    "ml engineer", "machine learning engineer",
    "senior machine learning engineer", "junior ml engineer",
    "ai engineer", "senior ai engineer",
    "data scientist", "senior data scientist", "lead data scientist",
    "nlp engineer", "research engineer", "applied scientist",
    "research scientist", "data engineer", "senior data engineer",
    "backend engineer", "senior backend engineer",
    "software engineer", "senior software engineer",
    "analytics engineer", "data analyst",
}

# ============================================================================
# JD Text for Semantic Embedding  (condensed core requirements)
# ============================================================================
JD_EMBEDDING_TEXT = (
    "Senior AI Engineer for a Series A AI-native talent intelligence platform. "
    "Building ranking retrieval and matching systems for recruiter search. "
    "Production experience with embeddings-based retrieval systems "
    "sentence-transformers vector databases Pinecone Weaviate Qdrant Milvus "
    "FAISS Elasticsearch hybrid search. "
    "Strong Python NLP machine learning deep learning. "
    "Evaluation frameworks for ranking systems NDCG MRR MAP A/B testing. "
    "Hybrid search combining BM25 with dense retrieval. "
    "LLM fine-tuning LoRA learning-to-rank models. "
    "5-9 years applied ML AI experience at product companies shipping to "
    "real users at meaningful scale. "
    "Located in India Pune or Noida preferred hybrid flexible work."
)
