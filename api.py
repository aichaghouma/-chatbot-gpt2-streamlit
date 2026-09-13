import re
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

from knowledge_base import KNOWLEDGE_BASE

# ============================================================
# (Reprise de la logique de app.py, sans Streamlit, sans PyTorch local)
# ============================================================

MOTS_FRANCAIS = {
    "quelle", "quel", "quels", "quelles", "qu'est-ce", "qu est ce",
    "comment", "pourquoi", "combien", "où", "ou est", "qui est",
    "est-ce", "peux-tu", "peux tu", "explique", "explique-moi",
    "définis", "définir", "c'est quoi", "c est quoi", "quelles sont",
    "quels sont", "je veux savoir", "dis-moi", "dis moi",
}


def est_francais(question):
    q = question.lower()
    if any(c in q for c in "éèêàùçôî"):
        return True
    if any(mot in q for mot in MOTS_FRANCAIS):
        return True
    return False


def traduire_en_francais(texte_anglais):
    try:
        return GoogleTranslator(source="en", target="fr").translate(texte_anglais)
    except Exception:
        return texte_anglais + "\n\n(Traduction indisponible, réponse affichée en anglais)"


def calculer_expression(question):
    q = question.replace(" ", "")
    match = re.search(r'[\d\.\+\-\*x×/\(\)]{3,}', q)
    if not match:
        return None
    expr = match.group().rstrip("+-*/x×.")
    if not expr:
        return None
    expr_calc = expr.replace("x", "*").replace("×", "*")
    if not re.fullmatch(r'[\d\.\+\-\*/\(\)]+', expr_calc):
        return None
    if expr_calc.count("(") != expr_calc.count(")"):
        return None
    try:
        resultat = eval(expr_calc, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return "Division by zero is undefined."
    except Exception:
        return None
    if isinstance(resultat, float) and resultat == int(resultat):
        resultat = int(resultat)
    expr_affichage = expr.replace("x", "×").replace("*", "×")
    return f"{expr_affichage} = {resultat}"


CAPITALES = {
    "france": "Paris", "tunisia": "Tunis", "tunisie": "Tunis",
    "morocco": "Rabat", "maroc": "Rabat", "algeria": "Algiers",
    "algerie": "Algiers", "libya": "Tripoli", "egypt": "Cairo",
    "egypte": "Cairo", "germany": "Berlin", "allemagne": "Berlin",
    "italy": "Rome", "italie": "Rome", "spain": "Madrid",
    "espagne": "Madrid", "portugal": "Lisbon", "uk": "London",
    "united kingdom": "London", "england": "London",
    "angleterre": "London", "usa": "Washington, D.C.",
    "united states": "Washington, D.C.", "etats-unis": "Washington, D.C.",
    "canada": "Ottawa", "china": "Beijing", "chine": "Beijing",
    "japan": "Tokyo", "japon": "Tokyo", "india": "New Delhi",
    "inde": "New Delhi", "brazil": "Brasilia", "bresil": "Brasilia",
    "russia": "Moscow", "russie": "Moscow", "turkey": "Ankara",
    "turquie": "Ankara", "greece": "Athens", "grece": "Athens",
    "switzerland": "Bern", "suisse": "Bern", "belgium": "Brussels",
    "belgique": "Brussels", "netherlands": "Amsterdam",
    "pays-bas": "Amsterdam", "senegal": "Dakar", "mali": "Bamako",
    "ivory coast": "Yamoussoukro", "cote d'ivoire": "Yamoussoukro",
    "saudi arabia": "Riyadh", "arabie saoudite": "Riyadh",
    "qatar": "Doha", "uae": "Abu Dhabi",
    "emirats arabes unis": "Abu Dhabi",
}


def chercher_capitale(question):
    q = question.lower()
    if "capital" not in q:
        return None
    for pays, capitale in CAPITALES.items():
        if pays in q:
            return f"The capital of {pays.title()} is {capitale}."
    return None


SEUIL_SIMILARITE = 0.15

EXPANSIONS_SYNONYMES = {
    "ml": "machine learning", "ai": "artificial intelligence", "dl": "deep learning",
    "db": "database", "os": "operating system", "ui": "user interface",
    "ux": "user experience", "api": "application programming interface",
    "nn": "neural network", "nlp": "natural language processing",
    "cs": "computer science", "iot": "internet of things",
}


def etendre_synonymes(texte):
    mots = texte.split()
    mots_etendus = []
    for m in mots:
        m_propre = re.sub(r"[^a-zA-Z]", "", m).lower()
        mots_etendus.append(EXPANSIONS_SYNONYMES.get(m_propre, m))
    return " ".join(mots_etendus)


def chercher_dans_base(question, vectorizer, matrix):
    question_etendue = etendre_synonymes(question)
    q_vec = vectorizer.transform([question_etendue])
    scores = cosine_similarity(q_vec, matrix)[0]
    idx = scores.argmax()
    if scores[idx] >= SEUIL_SIMILARITE:
        return KNOWLEDGE_BASE[idx], scores[idx]
    return None, scores[idx]


# ============================================================
# GENERATION VIA L'API D'INFERENCE HUGGING FACE (à distance)
# Remplace le chargement local de PyTorch/Transformers, trop lourd
# en mémoire pour un hébergement gratuit (Render free = 512 Mo).
# ============================================================

MODEL_NAME = "Aicha83/chatbot-gpt2-finetuned"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"


def generer_reponse(question, max_length=80, temperature=0.4):
    prompt = f"Question: {question}\nAnswer:"
    try:
        response = requests.post(
            HF_API_URL,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.85,
                    "repetition_penalty": 1.15,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
            timeout=30,
        )
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            reponse = data[0]["generated_text"].strip()
        else:
            return "Le modèle est momentanément indisponible, réessaie dans quelques instants."
    except Exception:
        return "Le modèle est momentanément indisponible, réessaie dans quelques instants."

    phrases = re.split(r'(?<=[.!?])\s+', reponse)
    reponse_courte = " ".join(phrases[:2]).strip()
    return reponse_courte if reponse_courte else reponse


# ============================================================
# CONSTRUCTION DE L'INDEX RAG (léger, tourne bien en local)
# ============================================================

print("Construction de l'index RAG...")
_textes = [f"{doc['title']} {doc['content']}" for doc in KNOWLEDGE_BASE]
_stop_words_etendus = list(ENGLISH_STOP_WORDS) + [
    "define", "defined", "defining", "definition",
    "explain", "explains", "explained", "explaining",
    "describe", "describes", "described", "describing",
    "tell", "tells", "telling",
]
vectorizer = TfidfVectorizer(stop_words=_stop_words_etendus)
matrix = vectorizer.fit_transform(_textes)
print("Index RAG prêt.")


# ============================================================
# API FASTAPI
# ============================================================

app = FastAPI(title="Chatbot Éducatif API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class ReponseAPI(BaseModel):
    reponse: str
    badge: str


@app.post("/chat", response_model=ReponseAPI)
def chat(payload: QuestionRequest):
    question = payload.question
    francais = est_francais(question)

    question_recherche = question
    if francais:
        try:
            question_recherche = GoogleTranslator(source="fr", target="en").translate(question)
        except Exception:
            question_recherche = question

    reponse_calcul = calculer_expression(question)

    if reponse_calcul:
        reponse = reponse_calcul
        badge = "Calcul exact (Python)"
    elif chercher_capitale(question_recherche):
        reponse = chercher_capitale(question_recherche)
        if francais:
            reponse = traduire_en_francais(reponse)
        badge = "Réponse vérifiée (base de capitales)"
    else:
        doc_trad, score_trad = chercher_dans_base(question_recherche, vectorizer, matrix)
        if francais:
            doc_brut, score_brut = chercher_dans_base(question, vectorizer, matrix)
            if doc_brut and doc_brut["subject"] in ("French", "English"):
                doc_brut, score_brut = None, 0
            if doc_brut and score_brut > score_trad:
                doc_trouve, score = doc_brut, score_brut
            else:
                doc_trouve, score = doc_trad, score_trad
        else:
            doc_trouve, score = doc_trad, score_trad

        if doc_trouve:
            reponse = doc_trouve["content"]
            if francais:
                reponse = traduire_en_francais(reponse)
            badge = f"Réponse vérifiée : {doc_trouve['title']} ({doc_trouve['subject']}) — RAG"
        else:
            reponse = generer_reponse(question_recherche)
            if francais:
                reponse = traduire_en_francais(reponse)
            badge = "Réponse générée par GPT-2 (non vérifiée)"

    return ReponseAPI(reponse=reponse, badge=badge)


# Sert les fichiers de l'app Flutter (dossier "static") sur le même port.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
