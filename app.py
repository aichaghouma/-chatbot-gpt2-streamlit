import streamlit as st
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================================================
# MINI BASE DE FAITS VÉRIFIÉS (capitales du monde)
# ============================================================
# Approche "RAG léger" : pour les questions de capitales, on répond
# depuis une source fiable plutôt que de laisser GPT-2 inventer.

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
    """Vérifie si la question porte sur une capitale connue et fiable."""
    q = question.lower()
    if "capital" not in q:
        return None
    for pays, capitale in CAPITALES.items():
        if pays in q:
            return f"The capital of {pays.title()} is {capitale}."
    return None


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(page_title="Chatbot ENET'Com - GPT-2 Fine-tuné", page_icon="🤖")

# ============================================================
# CHARGEMENT DU MODÈLE (mis en cache pour la rapidité)
# ============================================================

MODEL_NAME = "Aicha83/chatbot-gpt2-finetuned"  # ton modèle hébergé sur Hugging Face

@st.cache_resource
def charger_modele():
    device = torch.device("cpu")  # Streamlit Cloud n'a pas de GPU

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    return model, tokenizer, device


# ============================================================
# FONCTION DE GÉNÉRATION DE RÉPONSE
# ============================================================

def generer_reponse(model, tokenizer, question, device, max_length=80, temperature=0.4):
    model.eval()
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.15,     # pénalise légèrement les mots déjà utilisés
            no_repeat_ngram_size=3,      # interdit de répéter une séquence de 3 mots
            pad_token_id=tokenizer.eos_token_id
        )

    reponse = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reponse = reponse.split("Answer:")[-1].strip()

    # Garder seulement les 2 premières phrases (le modèle dérive souvent après)
    phrases = re.split(r'(?<=[.!?])\s+', reponse)
    reponse_courte = " ".join(phrases[:2]).strip()

    return reponse_courte if reponse_courte else reponse


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.title("🤖 Chatbot GPT-2 Fine-tuné")
st.caption("Projet de stage — ESSE Lab, ENET'Com Sfax — GPT-2 base fine-tuné sur Databricks Dolly 15k")

with st.expander("ℹ️ À propos de ce chatbot"):
    st.markdown("""
    Ce chatbot utilise **GPT-2 base**, un modèle pré-entraîné par OpenAI (124M paramètres),
    **fine-tuné** (ajusté) sur le dataset Databricks Dolly 15k pour ce projet de stage.

    **Caractéristiques techniques :**
    - Modèle de base : GPT-2 (124M paramètres, pré-entraîné sur du texte anglais varié)
    - Fine-tuning : 8 époques sur ~8500 exemples de questions/réponses filtrées
    - Dataset : Databricks Dolly 15k
    - Génération : anti-répétition activée (repetition_penalty, no_repeat_ngram_size)
    - **Mini-RAG** : pour les questions de capitales, une petite base de faits vérifiés
      est consultée en priorité, avant de laisser GPT-2 générer une réponse libre

    **Comparaison avec la version précédente (Transformer from scratch) :**
    Ce projet a d'abord exploré un Transformer codé et entraîné entièrement from scratch
    (~3.9M paramètres). Face aux limites de précision observées avec des ressources aussi
    limitées, cette version utilise un modèle pré-entraîné fine-tuné, illustrant l'écart
    de performance entre les deux approches — un point de comparaison clé pour l'analyse
    du rapport de stage.

    ⚠️ **Limites connues** : malgré l'amélioration nette, le modèle peut encore produire
    des inexactitudes factuelles occasionnelles (ex : mauvaise capitale, faits inventés),
    ce qui illustre l'intérêt d'une approche LLM + RAG pour un chatbot plus fiable.
    """)

# Charger le modèle (une seule fois, mis en cache)
try:
    model, tokenizer, device = charger_modele()
    modele_charge = True
except Exception as e:
    modele_charge = False
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.info(f"Vérifie que le modèle `{MODEL_NAME}` est bien public sur Hugging Face et accessible.")

if modele_charge:
    # Initialiser l'historique de conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie utilisateur
    if question := st.chat_input("Pose ta question ici (en anglais)..."):
        # Afficher la question de l'utilisateur
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("Génération de la réponse..."):
                reponse_fiable = chercher_capitale(question)
                if reponse_fiable:
                    reponse = reponse_fiable
                    st.caption("✅ Réponse vérifiée (base de faits)")
                else:
                    reponse = generer_reponse(model, tokenizer, question, device)
                    st.caption("🤖 Réponse générée par GPT-2 (non vérifiée)")
            st.markdown(reponse)
        st.session_state.messages.append({"role": "assistant", "content": reponse})

    # Bouton pour réinitialiser la conversation
    if st.session_state.messages:
        if st.button("🗑️ Réinitialiser la conversation"):
            st.session_state.messages = []
            st.rerun()
