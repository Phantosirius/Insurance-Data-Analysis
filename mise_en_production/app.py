"""
Dashboard — Assistance Automobile 2021-2022

Application Streamlit de mise en production du projet d'analyse de donnees.
Elle centralise les resultats des quatre etapes du pipeline :
nettoyage, exploration descriptive, econometrie et machine learning.

Structure : 7 pages navigables depuis la barre laterale.
Donnees attendues dans ./data/ et les rapports pre-calcules dans ./rapports/
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
#  Librairies standard du projet :
#  - streamlit  : framework de l'interface web interactive
#  - pandas     : manipulation des DataFrames (lecture CSV, agregations)
#  - numpy      : calculs numeriques (percentiles, distributions)
#  - plotly     : visualisations interactives (barres, aires, scatter, pie...)
#  - pathlib    : gestion des chemins de fichiers independante de l'OS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION DE LA PAGE
#  Doit etre le premier appel Streamlit du script.
#  layout="wide" utilise toute la largeur de l'ecran pour les graphiques.
#  initial_sidebar_state="expanded" ouvre la barre de navigation par defaut.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistance Auto · Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
#  THEME VISUEL — CSS GLOBAL
#
#  On injecte du CSS personnalise via st.markdown(unsafe_allow_html=True).
#  Objectif : remplacer le theme Streamlit par defaut par un design
#  "dark tech" coherent avec l'univers data/analytics.
#
#  Variables CSS definies dans :root pour centraliser les couleurs :
#    --bg / --bg2 / --bg3  : niveaux de fond (noir profond a gris anthracite)
#    --accent              : cyan #00d4ff  → couleur principale
#    --accent2             : violet #7c3aed → couleur secondaire
#    --accent3             : vert #10b981  → succes / positif
#    --warn / --danger     : orange / rouge → alertes
#    --text / --muted      : texte principal / texte secondaire
#    --font-h / --font-m   : Syne (titres) / JetBrains Mono (valeurs, code)
#
#  Classes utilitaires definies ici et reutilisees dans les helpers Python :
#    .prose       → bloc de texte analytique (gris clair, interligne 1.75)
#    .badge-*     → pastilles colorees pour les badges de significativite
#    .insight-box → encadre lateral avec bordure coloree (conclusions cles)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  :root {
    --bg:     #0a0d14; --bg2: #111520; --bg3: #181d2c;
    --border: #1e2740; --accent: #00d4ff; --accent2: #7c3aed;
    --accent3: #10b981; --warn: #f59e0b; --danger: #ef4444;
    --text:   #e2e8f0; --muted: #64748b;
    --font-h: 'Syne', sans-serif; --font-m: 'JetBrains Mono', monospace;
  }

  /* Fond global et police par defaut */
  .stApp { background: var(--bg); color: var(--text); font-family: var(--font-h); }
  .stApp > header { display: none; }

  /* Barre laterale */
  [data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--border); }
  [data-testid="stSidebar"] .stRadio label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span { color: var(--text) !important; }

  /* Cartes metriques (st.metric) */
  [data-testid="stMetric"] {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 20px !important; transition: border-color .2s;
  }
  [data-testid="stMetric"]:hover { border-color: var(--accent); }
  [data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: var(--font-m); font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--font-h); font-size: 28px !important; font-weight: 800; }
  [data-testid="stMetricDelta"] { font-family: var(--font-m); font-size: 12px !important; }

  /* Onglets (st.tabs) */
  .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; border-bottom: 1px solid var(--border); }
  .stTabs [data-baseweb="tab"] { background: transparent; color: var(--muted); font-family: var(--font-h); font-weight: 600; font-size: 13px; padding: 10px 20px; border-radius: 8px 8px 0 0; border: none; }
  .stTabs [aria-selected="true"] { background: var(--bg2) !important; color: var(--accent) !important; border-bottom: 2px solid var(--accent); }

  /* En-tete de section (tag + titre) */
  .section-header { font-family: var(--font-h); font-size: 11px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: var(--accent); margin-bottom: 4px; }
  .section-title  { font-family: var(--font-h); font-size: 26px; font-weight: 800; color: var(--text); margin-bottom: 8px; }

  /* Blocs de texte analytique */
  .prose { font-family: var(--font-h); font-size: 14px; color: #94a3b8; line-height: 1.75; margin-bottom: 24px; }
  .prose b { color: var(--text); }

  /* Pastilles colorees (badges) — utilisees pour significativite, statut */
  .badge { display: inline-block; padding: 3px 10px; border-radius: 99px; font-family: var(--font-m); font-size: 11px; font-weight: 700; }
  .badge-green  { background: #10b98122; color: #10b981; border: 1px solid #10b98144; }
  .badge-blue   { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff44; }
  .badge-purple { background: #7c3aed22; color: #a78bfa; border: 1px solid #7c3aed44; }
  .badge-orange { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }
  .badge-red    { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }

  /* Encadre de conclusion / point cle (bordure gauche coloree) */
  .insight-box { background: #0a0d14; border-left: 3px solid var(--accent); padding: 16px 20px; border-radius: 0 10px 10px 0; font-family: var(--font-m); font-size: 12px; color: #94a3b8; line-height: 1.8; margin-top: 12px; }
  .insight-box b { color: var(--text); }

  /* Separateurs et scrollbar */
  hr { border-color: var(--border) !important; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  THEME PLOTLY — CONSTANTES GLOBALES
#
#  COLORS : palette de 10 couleurs utilisee pour toutes les visualisations.
#           Ordonnee par priorite visuelle (cyan > violet > vert > orange > rouge).
#
#  BASE_LAYOUT : dictionnaire de parametres communs a tous les graphiques Plotly.
#    - Fond transparent (paper_bgcolor / plot_bgcolor) pour s'integrer au CSS sombre
#    - Polices et couleurs coherentes avec le reste de l'interface
#    - Grilles discretes (#1e2740) pour ne pas concurrencer les donnees
#    - Marges standardisees
#
#  Ces constantes sont passees a chaque figure via la fonction helper apply_layout().
# ─────────────────────────────────────────────────────────────────────────────
COLORS = ["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444",
          "#06b6d4","#8b5cf6","#34d399","#fbbf24","#f87171"]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",   # fond du graphique transparent
    plot_bgcolor="rgba(0,0,0,0)",    # fond de la zone de trace transparent
    font=dict(family="Syne, sans-serif", color="#e2e8f0", size=12),
    title_font=dict(family="Syne, sans-serif", size=16, color="#e2e8f0"),
    legend=dict(bgcolor="rgba(17,21,32,.8)", bordercolor="#1e2740", borderwidth=1),
    xaxis=dict(gridcolor="#1e2740", linecolor="#1e2740", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1e2740", linecolor="#1e2740", tickfont=dict(size=11)),
    margin=dict(t=50, b=40, l=40, r=20),
)


# ─────────────────────────────────────────────────────────────────────────────
#  CHEMINS DES DONNEES
#  On utilise pathlib.Path pour que les chemins fonctionnent sur tous les OS.
#  DATA_DIR    : dossier contenant les CSV nettoyees et enrichies
#  REPORTS_DIR : dossier contenant les rapports pre-calcules par les notebooks
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
REPORTS_DIR = Path("rapports")


# ─────────────────────────────────────────────────────────────────────────────
#  CHARGEMENT DES DONNEES — FONCTIONS CACHEES
#
#  @st.cache_data : decorateur Streamlit qui met les resultats en cache.
#  Sans ce decorateur, chaque interaction utilisateur rechargerait tous les CSV
#  depuis le disque, ce qui rendrait l'application trop lente.
#  Le cache est invalide automatiquement si les fichiers ou le code changent.
#
#  load_data()    : charge les 4 tables nettoyees issues du pipeline
#    - dos (dossiers_clean)    : table principale, une ligne par dossier
#    - enr (dossiers_enrichis) : table enrichie avec temps_total_min, nb_intervenants
#    - res (ressources_clean)  : presences agents (1 ligne par agent par jour)
#    - tps (temps_clean)       : durees de traitement (1 ligne par agent par dossier)
#
#  load_reports() : charge les 7 rapports CSV pre-calcules dans les notebooks
#    Ces rapports contiennent des agregats qui seraient trop longs a recalculer
#    en direct (indicateurs RH, feature importance ML, bilan de completude, etc.)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des donnees...")
def load_data():
    dos = pd.read_csv(DATA_DIR / "dossiers_clean.csv",    low_memory=False)
    enr = pd.read_csv(DATA_DIR / "dossiers_enrichis.csv", low_memory=False)
    res = pd.read_csv(DATA_DIR / "ressources_clean.csv",  low_memory=False)
    tps = pd.read_csv(DATA_DIR / "temps_clean.csv",       low_memory=False)
    return dos, enr, res, tps

@st.cache_data(show_spinner=False)
def load_reports():
    r = {}
    # Bilan de completude : taux de remplissage par variable (issu du nettoyage)
    r["completude"] = pd.read_csv(REPORTS_DIR / "bilan_completude.csv",        index_col=0)
    # Indicateurs d'activite globaux : volume, taux d'activation, temps median
    r["activite"]   = pd.read_csv(REPORTS_DIR / "indicateurs_activite.csv")
    # Indicateurs RH : nb agents, teletravail, presentiel, experience moyenne
    r["rh"]         = pd.read_csv(REPORTS_DIR / "indicateurs_rh.csv")
    # Comparaison des modeles ML : RMSE, MAE, R2 pour les 4 modeles
    r["ml_models"]  = pd.read_csv(REPORTS_DIR / "ml_comparaison_modeles.csv",   index_col=0)
    # Importance des variables du Random Forest (top features)
    r["ml_feat"]    = pd.read_csv(REPORTS_DIR / "ml_feature_importance.csv",    index_col=0)
    # Tableau des anomalies detectees et traitees lors du nettoyage
    r["anomalies"]  = pd.read_csv(REPORTS_DIR / "tableau_anomalies.csv")
    # Taux d'activation de chaque service (D/R, VR, Rapatriement, etc.)
    r["services"]   = pd.read_csv(REPORTS_DIR / "taux_activation_services.csv")
    return r


# ─────────────────────────────────────────────────────────────────────────────
#  BARRE LATERALE — NAVIGATION
#
#  La sidebar contient :
#    1. Un en-tete HTML avec le nom du projet et la periode
#    2. Un st.radio() qui definit la variable `page` controlant l'affichage
#    3. Un rappel des etapes du pipeline en bas de sidebar
#
#  La variable `page` est lue plus bas dans les blocs if/elif pour
#  afficher le contenu correspondant a la page selectionnee.
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 8px'>
      <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#00d4ff'>
        Assistance Auto
      </div>
      <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#64748b;margin-top:2px'>
        Donnees 2021 - 2022
      </div>
    </div>
    <hr style='margin:0 0 20px'>
    """, unsafe_allow_html=True)

    # Menu de navigation — chaque valeur correspond a une page du dashboard
    page = st.radio("Navigation", [
        "Vue d'ensemble",        # KPIs globaux + graphiques introductifs
        "Qualite des donnees",   # Completude + anomalies detectees
        "Activite et Services",  # Volumes, services, causes, clients
        "Ressources Humaines",   # Profil agents, contrats, experience
        "Temps de traitement",   # Distribution des durees, outliers, scatter
        "Machine Learning",      # Comparaison modeles + feature importance
        "Econometrie",           # OLS (temps) + Logit (vehicule de remplacement)
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    # Rappel des etapes du pipeline pour contextualiser la navigation
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#64748b;line-height:1.8'>
      <b style='color:#e2e8f0'>Pipeline</b><br>
      Nettoyage<br>Exploration<br>Econometrie<br>Machine Learning
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CHARGEMENT EFFECTIF DES DONNEES
#
#  On appelle les fonctions cachees ici, a la racine du script.
#  Si le chargement des CSV principaux echoue (fichiers absents ou mal nommes),
#  on affiche une erreur explicite et on stoppe l'execution avec st.stop().
#  Les rapports sont optionnels : s'ils sont absents, reports_ok = False
#  et les sections correspondantes affichent un avertissement plutot que de planter.
# ─────────────────────────────────────────────────────────────────────────────
try:
    dos, enr, res, tps = load_data()
except Exception as e:
    st.error(f"Impossible de charger les donnees : {e}")
    st.info("Verifiez que les fichiers CSV sont dans le dossier `data/`.")
    st.stop()

try:
    reports = load_reports()
    reports_ok = True
except Exception as e:
    reports = {}
    reports_ok = False
    st.warning(f"Rapports non charges : {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  FONCTIONS HELPER — COMPOSANTS REUTILISABLES
#
#  Ces fonctions encapsulent des patterns HTML/Plotly repetes dans chaque page.
#  Les centraliser ici evite la duplication et facilite les modifications globales.
#
#  section(tag, title, subtitle)
#    → Affiche l'en-tete d'une page : tag uppercase + grand titre + sous-titre optionnel
#    → Utilise les classes CSS .section-header et .section-title
#
#  prose(text)
#    → Affiche un paragraphe de texte analytique dans la classe CSS .prose
#    → Supporte les balises <b> pour mettre en evidence les chiffres cles
#
#  insight(text, color)
#    → Affiche un encadre de conclusion avec bordure coloree a gauche
#    → color accepte : "accent" (cyan), "purple" (violet), "green" (vert)
#    → Utilise la classe CSS .insight-box
#
#  badge(text, color)
#    → Retourne un span HTML avec la classe .badge-{color}
#    → Utilise pour les niveaux de significativite (***/**/*) et les statuts
#
#  apply_layout(fig, **kw)
#    → Applique BASE_LAYOUT a une figure Plotly + parametres additionnels optionnels
#    → Centralise l'application du theme sombre a tous les graphiques
#
#  row_item(label, value, color, note)
#    → Retourne un div HTML formatant une ligne label/valeur pour les tableaux de bord
# ─────────────────────────────────────────────────────────────────────────────
def section(tag, title, subtitle=""):
    """Affiche l'en-tete d'une page avec tag, titre et sous-titre optionnel."""
    sub = f'<div class="prose" style="margin-top:4px;margin-bottom:16px">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">{tag}</div>
    <div class="section-title">{title}</div>
    {sub}""", unsafe_allow_html=True)

def prose(text):
    """Affiche un bloc de texte analytique avec la mise en forme .prose."""
    st.markdown(f'<div class="prose">{text}</div>', unsafe_allow_html=True)

def insight(text, color="accent"):
    """Affiche un encadre de conclusion avec bordure coloree a gauche."""
    border = {"accent":"#00d4ff","purple":"#7c3aed","green":"#10b981"}.get(color,"#00d4ff")
    st.markdown(f"""
    <div class="insight-box" style="border-left-color:{border}">
      {text}
    </div>""", unsafe_allow_html=True)

def badge(text, color="blue"):
    """Retourne un span HTML representant un badge colore (pour significativite, statuts)."""
    return f'<span class="badge badge-{color}">{text}</span>'

def apply_layout(fig, **kw):
    """Applique le theme sombre BASE_LAYOUT a une figure Plotly."""
    fig.update_layout(**{**BASE_LAYOUT, **kw})
    return fig

def row_item(label, value, color="#00d4ff", note=""):
    """Retourne le HTML d'une ligne label/valeur pour les listes d'indicateurs."""
    note_html = f'<span style="color:#64748b;font-size:11px">{note}</span>' if note else ""
    return f"""
    <div style='display:flex;justify-content:space-between;align-items:center;
    padding:10px 14px;margin-bottom:5px;background:#0a0d14;border:1px solid #1e2740;
    border-radius:8px;font-family:JetBrains Mono,monospace;font-size:12px'>
      <span style='color:{color};font-weight:700'>{label}</span>
      <span style='color:#e2e8f0'>{value}&nbsp;{note_html}</span>
    </div>"""


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — VUE D'ENSEMBLE
#
#  Objectif : donner une vision synthetique du projet et des donnees.
#  Contenu :
#    - 5 metriques cles (KPIs) issues des rapports et du corpus
#    - Texte d'introduction sur la structure multi-table du projet
#    - Graphique d'evolution mensuelle des dossiers (serie temporelle agregee)
#    - Donut de repartition par service (depuis taux_activation_services.csv)
#    - Barres horizontales Top 10 causes d'intervention
#    - Barres verticales repartition par type d'energie
# ═════════════════════════════════════════════════════════════════════════════
if page == "Vue d'ensemble":
    section("DASHBOARD", "Vue d'ensemble du projet",
            "Ce tableau de bord centralise l'ensemble des resultats issus du pipeline d'analyse des donnees "
            "d'assistance automobile pour la periode 2021-2022. Il couvre 101 206 dossiers traites, "
            "2 507 agents actifs et quatre niveaux d'analyse : nettoyage, exploration descriptive, "
            "econometrie et machine learning.")

    # ── KPIs globaux ─────────────────────────────────────────────────────────
    # Valeurs issues des rapports pre-calcules (indicateurs_activite.csv et ml_comparaison_modeles.csv)
    # Affiches en dur pour garantir la coherence avec les rapports meme si les CSV changent
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Dossiers total",   "101 206",   "2021-2022")
    col2.metric("Agents uniques",   "2 507",     "RH actifs")
    col3.metric("Temps median",     "9.4 min",   "par dossier")
    col4.metric("Taux D/R",         "92.3 %",    "Depannage/Remorquage")
    col5.metric("Meilleur modele",  "R2 = 0.733","Random Forest")

    st.markdown("<br>", unsafe_allow_html=True)

    prose("""Le corpus de donnees brutes se decompose en trois tables sources complementaires :
    <b>dossiers.csv</b> (historique des interventions), <b>ressources.csv</b> (presences agents)
    et <b>temps.csv</b> (chronometrie des traitements). Apres nettoyage, les versions
    <i>_clean</i> et <i>_enrichis</i> constituent la base de travail stable sur laquelle
    reposent toutes les analyses presentees ici. La nature multi-table du projet impose une
    rigueur particuliere dans les jointures : le <b>Numero_dossier_ID</b> fait le lien entre
    dossiers et temps, tandis que le <b>Matricule</b> relie agents et interventions.""")

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        # ── Serie temporelle mensuelle ────────────────────────────────────────
        # On parse date.ouverture avec dayfirst=True (format FR : JJ/MM/AAAA)
        # resample("MS") = agregation au debut de chaque mois (Month Start)
        # Utilise dossiers_enrichis car la colonne date.ouverture y est fiable
        try:
            enr_copy = enr.copy()
            enr_copy["date.ouverture"] = pd.to_datetime(
                enr_copy["date.ouverture"], dayfirst=True, errors="coerce")
            monthly = (enr_copy.dropna(subset=["date.ouverture"])
                       .set_index("date.ouverture")
                       .resample("MS").size()          # compte de dossiers par mois
                       .reset_index(name="Dossiers"))
            monthly.columns = ["Mois", "Dossiers"]
            fig = px.area(monthly, x="Mois", y="Dossiers",
                          title="Evolution mensuelle des dossiers ouverts",
                          color_discrete_sequence=[COLORS[0]])
            fig.update_traces(fill="tozeroy", line_width=2,
                              fillcolor="rgba(0,212,255,.12)")   # remplissage transparent
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as ex:
            st.info(f"Graphique mensuel indisponible : {ex}")

    with col_b:
        # ── Donut des services ────────────────────────────────────────────────
        # Source : taux_activation_services.csv (pre-calcule dans le notebook d'exploration)
        # hole=0.55 pour un donut (vs pie plein)
        if reports_ok and "services" in reports:
            df_srv = reports["services"]
            fig = go.Figure(go.Pie(
                labels=df_srv["Service"], values=df_srv["Taux (%)"],
                hole=0.55, marker_colors=COLORS,
                textinfo="label+percent", textfont_size=11,
            ))
            fig.update_layout(title="Taux d'activation par service",
                showlegend=False,
                **{k:v for k,v in BASE_LAYOUT.items() if k not in ["xaxis","yaxis"]})
            st.plotly_chart(fig, use_container_width=True)

    prose("""La repartition par service revele une concentration majeure de l'activite sur le
    <b>Depannage/Remorquage (92.3 %)</b> des dossiers, ce qui en fait le flux operationnel
    central du plateau. Le Vehicule de Remplacement, bien que minoritaire (10.1 %), implique
    des delais de traitement significativement plus longs et mobilise des ressources specifiques,
    comme le confirme l'analyse econometrique. Le Rapatriement (8.4 %) et la Poursuite de voyage
    (1.4 %) representent des interventions rares mais complexes, souvent declenchees en cumul
    avec d'autres garanties.""")

    col_c, col_d = st.columns(2)
    with col_c:
        # ── Top 10 causes d'intervention ─────────────────────────────────────
        # Calcul en direct depuis dossiers_clean via value_counts()
        # Tri croissant pour que la barre la plus longue soit en haut du graphique horizontal
        if "Cause.intervention" in dos.columns:
            top_causes = dos["Cause.intervention"].value_counts().head(10).reset_index()
            top_causes.columns = ["Cause", "Dossiers"]
            fig = px.bar(top_causes.sort_values("Dossiers"),
                         x="Dossiers", y="Cause", orientation="h",
                         title="Top 10 — Causes d'intervention",
                         color="Dossiers",
                         color_continuous_scale=[[0,"#1e2740"],[1,"#00d4ff"]])
            fig.update_coloraxes(showscale=False)
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # ── Repartition par type d'energie ───────────────────────────────────
        # Segmentation technique du parc traite (essence, diesel, electrique, hybride...)
        if "Type.d.energie" in dos.columns:
            nrj = dos["Type.d.energie"].value_counts().reset_index()
            nrj.columns = ["Energie", "n"]
            fig = px.bar(nrj, x="Energie", y="n",
                         title="Repartition par type d'energie",
                         color="Energie", color_discrete_sequence=COLORS)
            fig.update_layout(showlegend=False)
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    prose("""La <b>Panne mecanique</b> domine largement le classement des causes d'intervention,
    suivie des accidents et des incidents lies aux pneumatiques. Cette predominance conditionne
    directement la nature des ressources a mobiliser : les interventions mecaniques requierent
    en general un technicien specifique, ce qui explique la mediane de 2 intervenants par dossier.
    La segmentation energetique montre une preponderance des vehicules <b>essence et diesel</b>,
    les motorisations electriques et hybrides restant marginales sur cette periode, ce qui
    pourrait evoluer significativement dans les prochains milllesimes de donnees.""")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — QUALITE DES DONNEES
#
#  Objectif : presenter les resultats du nettoyage realise dans traitement_donnees.ipynb.
#  Deux onglets :
#    Tab 1 — Completude : taux de remplissage par variable (bilan_completude.csv)
#      - Graphique a barres horizontales avec code couleur (vert/orange/rouge)
#      - Liste detaillee colonne par colonne avec badge de statut
#    Tab 2 — Anomalies : types d'anomalies, volumetrie et traitement applique
#      - 3 compteurs par table source (Dossiers / Temps / Ressources)
#      - Expanders par table avec detail ligne par ligne
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Qualite des donnees":
    section("DATA QUALITY", "Qualite et Nettoyage des donnees",
            "Avant toute modelisation, une etape rigoureuse de diagnostic et de traitement "
            "des anomalies a ete realisee sur les trois tables sources. Cette page en "
            "presente les resultats quantifies.")

    prose("""Le processus de nettoyage a porte sur <b>101 206 dossiers</b>, auxquels
    s'ajoutent plusieurs centaines de milliers de lignes dans la table temps (une ligne
    par intervenant par dossier) et la table ressources (une ligne par agent par jour de
    presence). La methodologie adoptee suit un principe de <b>conservation maximale de
    l'information</b> : les valeurs aberrantes ou manquantes sont isolees par des flags
    plutot que supprimees, ce qui permet de les inclure ou les exclure selon le besoin
    analytique. Trois categories d'anomalies ont ete identifiees et traitees :
    les <b>valeurs sentinelles</b> (notamment '???' representant 14 995 occurrences),
    les <b>decalages de colonnes</b> (28 lignes corrigees par shift) et les
    <b>formats de dates invalides</b> (38 lignes sur date.ouverture).""")

    # Deux onglets pour separer completude et anomalies
    tab1, tab2 = st.tabs(["  Completude  ", "  Anomalies detectees  "])

    with tab1:
        # ── Graphique de completude ───────────────────────────────────────────
        # Source : bilan_completude.csv (genere par le notebook de traitement)
        # Code couleur : vert >= 99%, orange >= 95%, rouge < 95%
        if reports_ok and "completude" in reports:
            df_comp = reports["completude"].copy()
            df_comp.index.name = "Colonne"
            df_comp = df_comp.reset_index()
            df_comp.columns = ["Colonne","Nb renseignes","Nb manquants","pct"]

            col1, col2 = st.columns([1, 1.2])
            with col1:
                df_s = df_comp.sort_values("pct")
                # Attribution des couleurs selon les seuils de qualite
                bar_colors = ["#ef4444" if v < 95 else "#f59e0b" if v < 99 else "#10b981"
                              for v in df_s["pct"]]
                fig = go.Figure(go.Bar(
                    x=df_s["pct"], y=df_s["Colonne"], orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:.1f}%" for v in df_s["pct"]],
                    textposition="outside", textfont_color="#e2e8f0",
                ))
                # Ligne de seuil a 95% pour materialiser le niveau d'acceptabilite
                fig.add_vline(x=95, line_dash="dash", line_color="#f59e0b", opacity=0.6,
                              annotation_text="Seuil 95%", annotation_font_color="#f59e0b")
                fig.update_layout(title="Taux de completude par variable", xaxis_range=[85,103],
                                  **BASE_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # ── Liste detaillee colonne par colonne ───────────────────────
                # Chaque ligne affiche : nom de la colonne, badge % completude, nb manquants
                st.markdown("<br>", unsafe_allow_html=True)
                for _, row in df_comp.iterrows():
                    pct   = row["pct"]
                    color = "green" if pct >= 99 else "orange" if pct >= 95 else "red"
                    manq  = int(row["Nb manquants"])
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:10px 14px;margin-bottom:6px;background:#111520;border:1px solid #1e2740;
                    border-radius:8px;font-family:JetBrains Mono,monospace;font-size:12px'>
                      <span style='color:#e2e8f0'>{row['Colonne']}</span>
                      <span>{badge(f'{pct:.1f} %', color)}&nbsp;
                      <span style='color:#64748b;font-size:11px'>{manq:,} manquants</span></span>
                    </div>""", unsafe_allow_html=True)

        insight("""<b>Lecture des resultats :</b> Quatre variables atteignent 100% de completude —
        <b>Outil.d.assistance</b>, <b>Client</b>, <b>date.ouverture</b> et
        <b>date.de.survenance</b> (arrondie a 100% avec 38 manquants sur 101 206).
        Ces variables pivot etant integralement renseignees, elles securisent les analyses
        temporelles et la granularite client. La <b>Formule</b> affiche 94.1% de completude,
        soit 5 945 valeurs manquantes conservees sous forme de NaN — une deperdition acceptable
        qui n'invalide pas les modeles mais doit etre prise en compte dans l'interpretation
        des segments contrat. Le point de vigilance principal reste
        <b>Assistance.ou.Administratif</b> a 90.1% (10 021 manquants) : cette variable cle
        pour distinguer les dossiers d'assistance des dossiers administratifs constitue le
        principal plafond de precision des modeles de classification futurs.""")

    with tab2:
        # ── Compteurs d'anomalies par table ──────────────────────────────────
        # Source : tableau_anomalies.csv (genere par le notebook de traitement)
        # Agregation par table (Dossiers / Temps / Ressources) via groupby implicite
        if reports_ok and "anomalies" in reports:
            df_anom = reports["anomalies"]

            col1, col2, col3 = st.columns(3)
            for tbl, col in [("Dossiers", col1), ("Temps", col2), ("Ressources", col3)]:
                n = int(df_anom[df_anom["Table"] == tbl]["Nb lignes"].sum())
                col.markdown(f"""
                <div style='background:#111520;border:1px solid #1e2740;border-radius:12px;
                padding:16px;text-align:center'>
                  <div style='font-family:JetBrains Mono,monospace;font-size:10px;
                  color:#64748b;letter-spacing:2px;text-transform:uppercase'>{tbl}</div>
                  <div style='font-family:Syne,sans-serif;font-size:28px;font-weight:800;
                  color:#e2e8f0;margin:4px 0'>{n:,}</div>
                  <div style='font-size:11px;color:#64748b'>lignes concernees</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            prose("""Le tableau ci-dessous detaille l'ensemble des anomalies detectees,
            leur volumetrie et le traitement applique. Chaque correction est tracable et
            reproductible via le pipeline de nettoyage. Les anomalies de la table
            <b>Temps</b> sont les plus volumineuses en nombre de lignes (263 702 dates
            manquantes sur Date.debut.traitement), ce qui reflete la nature de cette table :
            une ligne par agent par dossier, avec des interventions parfois non horodatees.
            Ces valeurs ont ete conservees avec un flag plutot que supprimees pour ne pas
            fausser les distributions de duree.""")

            # ── Detail par table via expanders ───────────────────────────────
            # Expander ouvert par defaut pour la table Dossiers (la plus importante)
            # Badge colore selon le type de retraitement (Corrige / Conserve / autre)
            for tbl in df_anom["Table"].unique():
                sub = df_anom[df_anom["Table"] == tbl]
                with st.expander(f"  Table {tbl} — {len(sub)} types d'anomalies detectees",
                                 expanded=(tbl == "Dossiers")):
                    for _, row in sub.iterrows():
                        retr  = str(row["Retraitement"])
                        col_a = "green" if "Corrig" in retr else "orange" if "Conserv" in retr else "blue"
                        st.markdown(f"""
                        <div style='display:flex;gap:12px;align-items:flex-start;
                        padding:10px 14px;margin-bottom:5px;background:#0a0d14;
                        border:1px solid #1e2740;border-radius:8px;
                        font-size:12px;font-family:JetBrains Mono,monospace'>
                          <div style='min-width:160px;color:#00d4ff'>{row['Colonne']}</div>
                          <div style='flex:1;color:#94a3b8'>{row['Type']}</div>
                          <div style='min-width:70px;text-align:right;color:#f59e0b'>
                            {int(row['Nb lignes']):,} lignes</div>
                          <div>{badge(retr[:40], col_a)}</div>
                        </div>""", unsafe_allow_html=True)

            insight("""<b>Anomalie marquante — valeur '???'</b> : 14 995 cellules contenaient
            la chaine de caracteres '???' repartie sur plusieurs colonnes, probablement issue
            d'un export systeme defaillant ou d'une valeur par defaut non nettoyee en amont.
            Ces valeurs ont ete systematiquement remplaces par NaN. Par ailleurs, 4 087
            dossiers ont ete crees automatiquement par le systeme (matricule 171) sans
            intervention humaine directe — ils ont ete flaggues via <b>flag_auto_creation</b>
            pour pouvoir les isoler dans les analyses de charge reelle du plateau.""",
            color="purple")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ACTIVITE ET SERVICES
#
#  Objectif : explorer les patterns d'activite du plateau sur 2021-2022.
#  Contenu :
#    - 4 KPIs d'activite (depuis indicateurs_activite.csv)
#    - Barres horizontales des taux d'activation par service
#    - Barres horizontales Top 10 clients par volume
#    - Texte analytique sur la hierarchie des services et la structure B2B
#    - Evolution mensuelle des 3 premières causes (serie temporelle multi-lignes)
#    - Note sur la migration MCS → Higgins (pas de graphique, texte uniquement)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Activite et Services":
    section("ACTIVITE", "Activite et Services",
            "Analyse descriptive des 101 206 dossiers traites sur la periode 2021-2022 : "
            "repartition par service, par client, par cause et par outil d'assistance.")

    prose("""Sur l'ensemble de la periode etudiee, le plateau d'assistance a traite
    <b>101 206 dossiers</b>, dont <b>90 %</b> relèvent de l'assistance directe et
    seulement <b>0.1 %</b> d'une nature purement administrative — ce desequilibre confirme
    que la quasi-totalite de l'activite est operationnelle. Le temps median de traitement
    s'etablit a <b>9.4 minutes</b> par dossier avec une mediane de <b>2 intervenants</b>,
    ce qui reflete un flux majoritairement standardise. Cependant, comme le montre
    l'analyse des temps, la distribution est fortement asymetrique : une minorite de
    dossiers complexes concentre l'essentiel des heures consommees.""")

    # ── KPIs d'activite depuis le rapport pre-calcule ────────────────────────
    # Les cles correspondent aux valeurs de la colonne "Indicateur" du CSV
    if reports_ok and "activite" in reports:
        df_act = reports["activite"].set_index("Indicateur")["Valeur"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dossiers total",       f'{int(df_act.get("Nombre total de dossiers", 101206)):,}')
        col2.metric("Part Assistance",      f'{df_act.get("Part dossiers Assistance", 90):.0f} %')
        col3.metric("Temps median",         f'{df_act.get("Temps médian de traitement", 9.4)} min')
        col4.metric("Intervenants (med.)",  f'{df_act.get("Nb médian intervenants/dossier", 2):.0f} agents')

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # ── Taux d'activation des services ───────────────────────────────────
        # Tri croissant pour que Depannage/Remorquage (le plus eleve) soit en haut
        # Echelle de couleur continue : gris fonce → violet → cyan
        if reports_ok and "services" in reports:
            df_srv = reports["services"].sort_values("Taux (%)", ascending=True)
            fig = px.bar(df_srv, x="Taux (%)", y="Service", orientation="h",
                         title="Taux d'activation des services (%)",
                         color="Taux (%)",
                         color_continuous_scale=[[0,"#1e2740"],[0.5,"#7c3aed"],[1,"#00d4ff"]])
            fig.update_coloraxes(showscale=False)
            fig.update_traces(
                text=[f"{v:.1f}%" for v in df_srv["Taux (%)"]],
                textposition="outside", textfont_color="#e2e8f0")
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ── Top 10 clients par volume ─────────────────────────────────────────
        # Calcul en direct depuis dossiers_clean : value_counts sur la colonne Client
        if "Client" in dos.columns:
            top_clients = dos["Client"].value_counts().head(10).reset_index()
            top_clients.columns = ["Client", "Dossiers"]
            fig = px.bar(top_clients.sort_values("Dossiers"),
                         x="Dossiers", y="Client", orientation="h",
                         title="Top 10 clients par volume de dossiers",
                         color_discrete_sequence=[COLORS[1]])
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    prose("""Le graphique des taux d'activation illustre la hierarchie des services :
    le <b>Depannage/Remorquage</b> est quasi-systematique (92.3%), ce qui en fait
    le service de reference du plateau. Le <b>Vehicule de Remplacement</b> (10.1%)
    et le <b>Rapatriement</b> (8.4%) sont des services complementaires actives
    en cas d'immobilisation prolongee du vehicule. La <b>Recuperation du vehicule</b>
    (4.3%) et les <b>Autres garanties</b> (3.2%) representent des services de niche
    fortement dependants de la formule souscrite par le client.
    La concentration des dossiers chez quelques clients majeurs confirme une structure
    B2B : les 10 premiers clients representent une part disproportionnee du volume,
    ce qui implique des SLA differencies et une gestion de compte specifique.""")

    # ── Evolution mensuelle par cause (Top 3) ────────────────────────────────
    # On filtre sur les 3 causes les plus frequentes pour limiter la charge visuelle
    # La variable mois est reconstituee depuis date.ouverture de dossiers_enrichis
    try:
        enr_copy = enr.copy()
        enr_copy["date.ouverture"] = pd.to_datetime(
            enr_copy["date.ouverture"], dayfirst=True, errors="coerce")
        enr_valid = enr_copy.dropna(subset=["date.ouverture","Cause.intervention"])
        top3 = enr_valid["Cause.intervention"].value_counts().head(3).index.tolist()
        enr_top = enr_valid[enr_valid["Cause.intervention"].isin(top3)].copy()
        enr_top["Mois"] = enr_top["date.ouverture"].dt.to_period("M").dt.to_timestamp()
        monthly_cause = enr_top.groupby(["Mois","Cause.intervention"]).size().reset_index(name="n")
        fig = px.line(monthly_cause, x="Mois", y="n", color="Cause.intervention",
                      title="Evolution mensuelle — Top 3 causes d'intervention",
                      color_discrete_sequence=COLORS, markers=True)
        fig.update_traces(line_width=2)
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        prose("""L'evolution mensuelle confirme la presence d'une <b>saisonnalite marquee</b>
        dans les causes d'intervention. Les pannes mecaniques connaissent des pics hivernaux
        (demarrage difficile, batterie) et estivaux (surchauffe, climatisation). Cette
        saisonnalite est capturee dans le modele ML via la variable <b>mois</b>, qui ressort
        comme huitieme variable en importance (1.24%). Anticiper ces cycles permet d'adapter
        les niveaux de staffing avant les periodes de tension.""")
    except Exception:
        pass  # Si la date n'est pas parsable, on saute ce graphique sans faire planter l'app

    # ── Note sur les outils (texte uniquement, pas de graphique) ─────────────
    # Le graphique de repartition MCS/Higgins a ete supprime car il ne s'affichait pas
    # correctement. L'information est preservee sous forme textuelle.
    prose("""Sur les outils d'assistance, deux systemes coexistent sur la periode : <b>MCS</b>
    (ancien outil) et <b>Higgins</b> (nouvel outil). Une migration progressive de MCS vers
    Higgins est observable au fil des mois de 2021 a 2022. Cette transition est
    operationnellement importante : l'outil MCS ressort comme variable significative dans
    le modele Logit, son association negative avec l'activation du vehicule de remplacement
    suggerant que les dossiers traites sous MCS correspondent a des typologies d'interventions
    differentes de ceux traites sous Higgins.""")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RESSOURCES HUMAINES
#
#  Objectif : caracteriser le profil des 2 507 agents actifs sur la periode.
#  Contenu :
#    - 4 KPIs RH (depuis indicateurs_rh.csv)
#    - Donut teletravail / presentiel (valeurs issues du rapport)
#    - Donut type de contrat CDI / CDD-CDS (valeurs issues du rapport)
#    - Histogramme de la distribution de l'experience (depuis ressources_clean)
#    - Barres horizontales repartition par lieu de travail
#    - Barres verticales composition par type de contrat
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Ressources Humaines":
    section("RESSOURCES HUMAINES", "Analyse des ressources humaines",
            "Profil des 2 507 agents actifs sur la periode : contrats, lieux de travail, "
            "modes de presence et distribution de l'experience.")

    prose("""La table <b>ressources_clean.csv</b> recense l'ensemble des jours de presence
    des agents, avec une ligne par agent et par jour. Cette structure permet de retracer
    finement les conditions de travail : lieu (teletravail ou site physique), type de
    contrat, duree contractuelle et experience cumulee. Sur la periode 2021-2022,
    <b>2 507 matricules uniques</b> ont ete identifies, avec une experience moyenne
    de <b>787 jours</b> (environ 2.2 ans). Cette anciennete moderee reflete un plateau
    en phase de montee en competence, avec une part significative de profils recents
    (CDD et CDS representent 64.2% des contrats).""")

    # ── KPIs RH depuis le rapport pre-calcule ────────────────────────────────
    # Les indicateurs sont lus par position (iloc) car les noms peuvent varier avec les accents
    if reports_ok and "rh" in reports:
        df_rh = reports["rh"]
        vals  = df_rh.set_index("Indicateur")["Valeur"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Agents uniques",    f'{int(vals.iloc[0]):,}')
        col2.metric("Teletravail",       f'{vals.iloc[1]:.1f} %')
        col3.metric("Presentiel",        f'{vals.iloc[2]:.1f} %')
        col4.metric("Experience moy.",   f'{int(vals.iloc[3])} jours')

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # ── Donut teletravail vs presentiel ──────────────────────────────────
        # Valeurs issues du rapport indicateurs_rh.csv (49.4% / 50.6%)
        fig = go.Figure(go.Pie(
            labels=["Teletravail","Presentiel"], values=[49.4, 50.6],
            hole=0.55, marker_colors=[COLORS[0], COLORS[1]],
            textinfo="label+percent",
        ))
        fig.update_layout(title="Teletravail / Presentiel",
            showlegend=False,
            **{k:v for k,v in BASE_LAYOUT.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ── Donut type de contrat ─────────────────────────────────────────────
        # Valeurs issues du rapport : CDI = 35.8%, CDD+CDS+autres = 64.2%
        fig2 = go.Figure(go.Pie(
            labels=["CDI","CDD / CDS / Autres"], values=[35.8, 64.2],
            hole=0.55, marker_colors=[COLORS[2], COLORS[3]],
            textinfo="label+percent",
        ))
        fig2.update_layout(title="Types de contrat",
            showlegend=False,
            **{k:v for k,v in BASE_LAYOUT.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig2, use_container_width=True)

    prose("""La repartition teletravail/presentiel est quasi-equiparee : <b>49.4%</b> des
    jours de presence sont effectues en teletravail contre <b>50.6%</b> en presentiel.
    Cette quasi-parite, observee sur l'ensemble de la periode, reflete l'adaptation du
    plateau aux contraintes sanitaires de 2021 puis la montee en puissance du travail hybride
    en 2022. Elle souleve une question operationnelle importante : le mode de presence
    influence-t-il la productivite et le temps de traitement ? Cette piste n'a pas ete
    modelisee dans le scope actuel mais constitue un axe d'analyse pertinent pour la suite.
    La predominance des <b>CDD et CDS</b> (64.2%) indique une politique RH flexible,
    adaptee aux variations saisonnieres d'activite, mais qui peut generer une volatilite
    dans le niveau d'experience moyen du plateau selon la periode.""")

    # ── Histogramme de la distribution de l'experience ───────────────────────
    # Colonne Experience dans ressources_clean.csv (en jours)
    # On filtre les valeurs nulles ou negatives (absences non renseignees)
    if "Experience" in res.columns:
        exp_valid = res["Experience"].dropna()
        exp_valid = exp_valid[exp_valid > 0]
        fig = px.histogram(exp_valid, nbins=60,
                           title="Distribution de l'experience des agents (jours)",
                           color_discrete_sequence=[COLORS[0]])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_title="Experience (jours)", yaxis_title="Nb agents")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        prose("""La distribution de l'experience presente une forme bimodale caracteristique
        des plateaux d'assistance : un premier pic concentre les agents recents (moins de
        200 jours), correspondant aux nouvelles recrues en CDD, et un second pic autour
        des agents experimentes (800 a 1 500 jours) formant le socle stable du plateau.
        Cette structure bimodale a une implication directe sur les temps de traitement :
        les agents experimentes traitent les dossiers complexes plus rapidement et
        avec moins d'intervenants. Croiser l'experience avec le temps de traitement
        constitue une piste d'optimisation RH concrete.""")

    # ── Repartition par lieu de travail ──────────────────────────────────────
    if "Lieu.travail" in res.columns:
        lieu = res["Lieu.travail"].value_counts().head(8).reset_index()
        lieu.columns = ["Lieu", "n"]
        fig = px.bar(lieu, x="n", y="Lieu", orientation="h",
                     title="Repartition par lieu de travail",
                     color_discrete_sequence=[COLORS[4]])
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Composition par type de contrat (calcul direct depuis ressources_clean) ─
    if "Type.de.contrat" in res.columns:
        c_count = res["Type.de.contrat"].value_counts().reset_index()
        c_count.columns = ["Type contrat", "n"]
        fig = px.bar(c_count, x="Type contrat", y="n",
                     title="Composition par type de contrat",
                     color="Type contrat", color_discrete_sequence=COLORS)
        fig.update_layout(showlegend=False)
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    insight("""<b>Point de vigilance :</b> 51 833 lignes de la table ressources presentent
    une valeur de <b>Temps.travail</b> hors du referentiel attendu (30, 50, 70, 80 ou 100%).
    Ces lignes ont ete conservees avec une note de documentation. Avant d'utiliser cette
    variable dans un modele, une verification aupres de la source RH s'impose pour determiner
    si ces valeurs correspondent a des temps partiels atypiques ou a des erreurs de saisie.""",
    color="purple")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — TEMPS DE TRAITEMENT
#
#  Objectif : analyser la distribution des durees de traitement et identifier
#             les facteurs associes aux dossiers longs.
#  Source : temps_clean.csv (colonne duree.corrigee en secondes)
#           dossiers_enrichis.csv (colonne temps_total_min pour le scatter)
#  Contenu :
#    - 4 metriques : mediane, moyenne, P90, P99 (calcules depuis duree.corrigee)
#    - Histogramme de la distribution (clippe au P99 pour lisibilite)
#    - Boite a moustaches (avec affichage de la moyenne via boxmean="sd")
#    - Repartition des flags de duree extreme
#    - Scatter nb_intervenants vs temps_total_min avec tendance LOWESS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Temps de traitement":
    section("TEMPS", "Analyse des temps de traitement",
            "Distribution et caracteristiques de la duree de traitement par dossier, "
            "calculee depuis la table temps_clean.csv (colonne duree.corrigee, en secondes).")

    prose("""La table <b>temps_clean.csv</b> contient une ligne par intervenant par dossier.
    La colonne <b>duree.corrigee</b> represente la duree effective de traitement en secondes,
    apres correction des valeurs aberrantes (une seule duree superieure a 8 heures a ete
    conservee avec flag). Pour obtenir le temps total par dossier, les durees sont agreges
    au niveau du dossier dans <b>dossiers_enrichis.csv</b> (colonne <b>temps_total_min</b>),
    qui constitue la variable cible des modeles predictifs. Il est important de noter que
    263 702 lignes n'ont pas de date de debut de traitement renseignee — elles ont ete
    conservees mais exclues du calcul des distributions temporelles presentees ici.""")

    if "duree.corrigee" in tps.columns:
        # ── Calcul des statistiques descriptives ─────────────────────────────
        # Conversion secondes → minutes pour une lecture metier directe
        valid_sec = tps["duree.corrigee"].dropna()
        valid_sec = valid_sec[valid_sec > 0]        # on exclut les durees nulles
        valid_min = valid_sec / 60.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mediane",  f"{valid_min.median():.1f} min")
        col2.metric("Moyenne",  f"{valid_min.mean():.1f} min")
        col3.metric("P90",      f"{valid_min.quantile(.9):.1f} min")   # seuil SLA
        col4.metric("P99",      f"{valid_min.quantile(.99):.1f} min")  # cas extremes

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            # ── Histogramme clippe au P99 ────────────────────────────────────
            # On clippe au P99 pour eviter que quelques valeurs extremes
            # ne rendent illisible la majorite de la distribution
            clipped = valid_min.clip(upper=valid_min.quantile(.99))
            fig = px.histogram(clipped, nbins=80,
                               title="Distribution des durees de traitement (min)",
                               color_discrete_sequence=[COLORS[0]])
            fig.update_traces(marker_line_width=0)
            fig.update_layout(xaxis_title="Duree (min)", yaxis_title="Frequence")
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # ── Boite a moustaches ───────────────────────────────────────────
            # boxmean="sd" affiche la moyenne et l'ecart-type en plus de la mediane
            # Permet de visualiser l'asymetrie (moyenne > mediane → queue droite)
            fig = go.Figure(go.Box(
                y=clipped, name="Duree (min)",
                marker_color=COLORS[1], line_color=COLORS[1], boxmean="sd",
            ))
            fig.update_layout(title="Boite a moustaches — duree par intervention", **BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        prose("""La distribution des durees est fortement <b>asymetrique a droite</b> :
        la majorite des interventions se concentrent sous les 20 minutes (flux standardises),
        mais une queue epaisse s'etend jusqu'a plusieurs heures pour les dossiers complexes.
        L'ecart entre la <b>mediane</b> et la <b>moyenne</b> illustre cet effet : la
        moyenne est systematiquement tiree vers le haut par les cas extremes.
        Le <b>P90</b> donne le seuil au-dela duquel une intervention est consideree comme
        longue — ce seuil est directement exploitable pour la definition d'alertes SLA.
        Le <b>P99</b> identifie les cas exceptionnels qui necessitent un traitement
        specifique, potentiellement une escalade vers un gestionnaire senior.""")

        # ── Distribution du flag duree extreme ───────────────────────────────
        # flag_duree_extreme = 1 si la duree depasse le seuil defini lors du nettoyage
        if "flag_duree_extreme" in tps.columns:
            flag_counts = tps["flag_duree_extreme"].value_counts().reset_index()
            flag_counts.columns = ["flag", "n"]
            fig = px.bar(flag_counts, x="flag", y="n",
                         title="Repartition des dossiers par flag duree extreme",
                         color_discrete_sequence=[COLORS[3]])
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── Scatter nb_intervenants vs temps_total_min ───────────────────────
        # Source : dossiers_enrichis (jointure implicite via Numero_dossier_ID)
        # Echantillon de 5 000 points pour la lisibilite (random_state fixe pour reproductibilite)
        # trendline="lowess" : tendance lissee non parametrique (meilleure que lineaire ici)
        if "nb_intervenants" in enr.columns and "temps_total_min" in enr.columns:
            sample = enr[["nb_intervenants","temps_total_min"]].dropna()
            sample = sample[(sample["temps_total_min"] > 0)]
            sample = sample[sample["temps_total_min"] < sample["temps_total_min"].quantile(.99)]
            sample = sample.sample(min(5000, len(sample)), random_state=42)
            fig = px.scatter(sample, x="nb_intervenants", y="temps_total_min",
                             title="Temps total vs Nombre d'intervenants (echantillon 5 000 dossiers)",
                             opacity=0.3, color_discrete_sequence=[COLORS[0]],
                             trendline="lowess",             # courbe LOWESS en jaune
                             trendline_color_override="#f59e0b")
            fig.update_layout(xaxis_title="Nb intervenants", yaxis_title="Temps total (min)")
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            prose("""Le nuage de points confirme la relation <b>croissante et non lineaire</b>
            entre le nombre d'intervenants et le temps total de traitement. En deca de 3
            intervenants, la relation est quasi-lineaire et moderee. Au-dela de 3, le temps
            s'emballe : chaque intervenant supplementaire induit une coordination qui multiplie
            les delais. C'est pourquoi <b>nb_intervenants</b> ressort comme variable dominante
            dans le modele Random Forest avec 72.9% d'importance — c'est le meilleur proxy
            observable de la complexite d'un dossier des son ouverture.""")

        insight("""<b>Recommandation operationnelle :</b> Mettre en place une alerte automatique
        des qu'un dossier franchit le seuil de 3 intervenants. Ce signal precoce permettrait
        au superviseur de prioriser la resolution avant que le temps de traitement ne devienne
        critique. Les dossiers P99 (au-dela du 99eme percentile) pourraient faire l'objet d'un
        suivi specifique avec consignation des causes de depassement.""", color="green")

    else:
        st.warning("Colonne `duree.corrigee` absente de `temps_clean.csv`.")
        st.write("Colonnes disponibles :", list(tps.columns))


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — MACHINE LEARNING
#
#  Objectif : presenter les resultats du volet ML realise dans machine_learning.ipynb.
#  Variable cible : temps_total_min (regression)
#  Modeles compares : Regression lineaire, Arbre de decision, Random Forest, Gradient Boosting
#  Partition : 80% apprentissage / 20% test (stratification par mois)
#  Contenu :
#    - Contexte et objectif de la prediction
#    - 4 cartes de modeles avec metriques (RMSE, MAE, R2) — meilleur modele surligné
#    - Comparaison visuelle des metriques (subplots 3 colonnes)
#    - Feature importance du Random Forest (Top 12 variables)
#    - Encadre de conclusion et pistes d'amelioration
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Machine Learning":
    section("MACHINE LEARNING", "Modeles predictifs",
            "Prediction du temps total de traitement (temps_total_min) a partir des "
            "caracteristiques observables a l'ouverture du dossier.")

    prose("""L'objectif du volet Machine Learning est de predire, <b>des l'ouverture d'un
    dossier</b>, le temps total qu'il necessittera. Cette prediction en temps reel permettrait
    aux superviseurs d'allouer les ressources de facon proactive : orienter les dossiers
    prevus courts vers les agents en periode de forte charge, et reserver les dossiers complexes
    aux profils experimentes disponibles. Quatre modeles ont ete entraines et evalues sur la
    partition <b>Test</b> (20% des donnees, apres stratification par mois) :
    la <b>Regression lineaire</b>, l'<b>Arbre de decision</b>, le <b>Random Forest</b>
    et le <b>Gradient Boosting</b>. Les metriques retenues sont le RMSE (erreur quadratique
    moyenne), le MAE (erreur absolue moyenne) et le R2 (coefficient de determination).""")

    if reports_ok and "ml_models" in reports:
        df_ml   = reports["ml_models"]
        # Detection du nom de la colonne R2 (peut etre "R2" ou "R²" selon l'encodage)
        r2_col  = "R2" if "R2" in df_ml.columns else "R²"
        best_model = df_ml[r2_col].idxmax()   # modele avec le R2 le plus eleve

        # ── Cartes de modeles ─────────────────────────────────────────────────
        # Le meilleur modele est surligné avec une bordure cyan (#00d4ff)
        # Les autres ont une bordure neutre (#1e2740)
        cols = st.columns(len(df_ml))
        for i, (name, row) in enumerate(df_ml.iterrows()):
            is_best      = (name == best_model)
            border_color = "#00d4ff" if is_best else "#1e2740"
            best_badge   = badge("MEILLEUR", "blue") if is_best else ""
            cols[i].markdown(f"""
            <div style='background:#111520;border:2px solid {border_color};border-radius:12px;
            padding:16px;text-align:center;min-height:170px'>
              <div style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;
              color:#e2e8f0;margin-bottom:10px'>{name}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#64748b'>RMSE</div>
              <div style='font-size:22px;font-weight:800;color:#e2e8f0;margin-bottom:8px'>{row['RMSE']}</div>
              <div style='display:flex;justify-content:space-around;margin-bottom:10px'>
                <div>
                  <div style='font-size:10px;color:#64748b'>MAE</div>
                  <div style='font-size:14px;color:#e2e8f0'>{row['MAE']}</div>
                </div>
                <div>
                  <div style='font-size:10px;color:#64748b'>R2</div>
                  <div style='font-size:14px;color:#10b981'>{row[r2_col]}</div>
                </div>
              </div>
              {best_badge}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        prose("""Le <b>Random Forest</b> est le modele le plus performant avec un <b>R2 = 0.733</b>,
        un RMSE de <b>18.05 minutes</b> et un MAE de <b>8.55 minutes</b>. Cela signifie que
        le modele explique <b>73.3%</b> de la variance du temps de traitement, et que son
        erreur mediane (MAE) est d'environ 8 minutes et demi. Le Gradient Boosting est tres
        proche (R2 = 0.729, RMSE = 18.17), confirmant que les methodes ensemblistes sont
        adaptees a ce type de donnees. La Regression lineaire, bien qu'inferieure (R2 = 0.701),
        reste competitive, ce qui suggere que la relation est en partie lineaire — c'est
        d'ailleurs ce que l'econometrie confirme avec la variable nb_intervenants.""")

        col1, col2 = st.columns(2)
        with col1:
            # ── Comparaison visuelle des metriques ────────────────────────────
            # Subplots 3 colonnes : RMSE, MAE (plus court = meilleur), R2 (plus haut = meilleur)
            # On trie dans le sens croissant pour RMSE/MAE, decroissant pour R2
            fig = make_subplots(rows=1, cols=3, subplot_titles=["RMSE (min)", "MAE (min)", "R2"])
            for i, metric in enumerate(["RMSE","MAE",r2_col], 1):
                asc = (metric != r2_col)   # tri croissant pour RMSE et MAE, decroissant pour R2
                sd  = df_ml.sort_values(metric, ascending=asc)
                fig.add_trace(go.Bar(
                    y=sd.index, x=sd[metric], orientation="h",
                    marker_color=COLORS[:len(df_ml)], showlegend=False,
                    text=[f"{v:.3f}" for v in sd[metric]], textposition="outside"),
                    row=1, col=i)
            fig.update_layout(title="Comparaison des metriques de performance", **BASE_LAYOUT)
            fig.update_xaxes(showgrid=True, gridcolor="#1e2740")
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # ── Feature importance — Random Forest ────────────────────────────
            # Source : ml_feature_importance.csv (genere par le notebook ML)
            # On nettoie les noms des variables encodees (one-hot) pour la lisibilite
            # On affiche les 12 variables les plus importantes (tail apres tri croissant)
            if reports_ok and "ml_feat" in reports:
                df_feat = reports["ml_feat"].sort_values("importance").tail(12).copy()
                df_feat.index = (df_feat.index
                    .str.replace("Cause.intervention_","Cause: ",regex=False)
                    .str.replace("Outil.d.assistance_","Outil: ",regex=False)
                    .str.replace("Formule_","Formule: ",regex=False)
                    .str.replace("Client_","Client: ",regex=False))
                fig = px.bar(df_feat, x="importance", y=df_feat.index, orientation="h",
                             title="Feature Importance — Random Forest (Top 12)",
                             color="importance",
                             color_continuous_scale=[[0,"#1e2740"],[0.3,"#7c3aed"],[1,"#00d4ff"]])
                fig.update_coloraxes(showscale=False)
                apply_layout(fig)
                st.plotly_chart(fig, use_container_width=True)

        prose("""L'analyse des importances de variables revele une hierarchie tres marquee.
        <b>nb_intervenants</b> concentre a lui seul <b>72.9%</b> de l'importance totale —
        c'est de loin le signal le plus puissant. Vient ensuite la <b>Formule F154</b>
        (8.8%), suggerant que ce contrat specifique est associe a des dossiers
        structurellement differents. La <b>Panne mecanique</b> (3.9%) et la categorie
        <b>Autres causes</b> (2.4%) apportent une information incrementale utile.
        <b>TOP.VR</b> (1.7%), la saisonnalite <b>mois</b> (1.2%) et <b>TOP.Rappat.valide</b>
        (0.7%) completent le tableau. La faible importance des variables de client et
        d'outil (moins de 2% chacune) indique que ces dimensions n'ont qu'un impact marginal
        sur la duree une fois le nombre d'intervenants connu — ce qui est coherent avec
        l'interpretation econometrique.""")

        insight("""<b>Interpretation operationnelle du R2 = 0.733 :</b> Le modele predit
        correctement la fourchette de duree dans environ 3 cas sur 4. Les 26.7% de variance
        non expliquee correspondent probablement a des facteurs non observes dans les donnees
        actuelles : disponibilite reelle des prestataires, difficulte geographique de
        l'intervention, qualite de la communication client. L'amelioration de la variable
        <b>Assistance.ou.Administratif</b> (actuellement a 90.1% de completude) constitue
        le levier le plus accessible pour faire progresser le R2 sans changer l'architecture
        du modele.""", color="green")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — ECONOMETRIE
#
#  Objectif : presenter les resultats econometriques realises dans econometrie.ipynb.
#  Deux modeles en deux onglets :
#
#  Tab 1 — OLS (Regression lineaire sur temps_total_min)
#    - Liste des variables retenues avec niveau de significativite
#    - Interpretation des effets (signe + niveau de significativite ***)
#    - Trois tests de validite : Shapiro-Wilk, Breusch-Pagan, VIF
#    - Encadre d'aide a la lecture des coefficients
#
#  Tab 2 — Logit (Probabilite d'activation du vehicule de remplacement TOP.VR)
#    - KPIs : variable cible et pseudo R2 McFadden
#    - Tableau des effets marginaux (direction + et - par variable)
#    - Graphique distribution de TOP.VR (pour illustrer le desequilibre de classes)
#    - Encadre de synthese sur la convergence OLS / Logit / ML
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Econometrie":
    section("ECONOMETRIE", "Modeles econometriques",
            "Deux modeles sont presentes : une regression OLS sur le temps de traitement "
            "et un modele Logit sur la probabilite d'activation du vehicule de remplacement.")

    prose("""Le volet econometrique vient completer l'analyse Machine Learning en apportant
    une <b>lecture causale et statistiquement rigoureuse</b> des determinants du temps de
    traitement et de l'activation des services. Contrairement aux modeles ML qui optimisent
    la prediction, l'econometrie permet d'interpreter les <b>coefficients</b> et de tester
    leur significativite. Les deux modeles ont ete estimes sur la partition
    <b>Apprentissage</b> du fichier dossiers_enrichis.csv, apres exclusion des dossiers
    flaggues auto-crees (flag_auto_creation = 1) et des valeurs manquantes sur les
    variables retenues.""")

    tab1, tab2 = st.tabs(["  OLS — Temps de traitement  ",
                           "  Logit — Vehicule de remplacement  "])

    with tab1:
        prose("""Le modele OLS (Moindres Carres Ordinaires) vise a expliquer la variable
        <b>temps_total_min</b> a partir des caracteristiques observables a l'ouverture du
        dossier. Les variables continues (nb_intervenants) sont incluses directement,
        les variables categorielles (Outil, Client, Formule, Cause) sont encodees en
        variables indicatrices (dummies). La variable <b>mois</b> est traitee comme
        variable continue pour capturer la tendance intra-annuelle, apres verification
        qu'une specification lineaire est suffisante sur ce corpus.""")

        # ── Variables retenues dans le modele OLS ────────────────────────────
        # Liste manuelle des variables significatives issues du notebook econometrie.ipynb
        # Champ : (nom_variable, niveau_significativite, couleur, interpretation_metier)
        ols_vars = [
            ("nb_intervenants",   "Positif ***", "#10b981",
             "Chaque intervenant supplementaire allonge significativement la duree"),
            ("TOP.D.R",           "Positif ***", "#10b981",
             "Les dossiers Depannage/Remorquage durent plus longtemps"),
            ("TOP.VR",            "Positif ***", "#10b981",
             "L'activation du VR allonge mecaniquement le traitement"),
            ("TOP.Rappat.valide", "Positif **",  "#f59e0b",
             "Le rapatriement valide implique une coordination plus longue"),
            ("Outil.d.assistance","Significatif","#00d4ff",
             "L'outil utilise influence la duree selon ses specificites fonctionnelles"),
            ("mois",              "Saisonnier",  "#7c3aed",
             "La saisonnalite module le temps de traitement sur l'annee"),
        ]

        st.markdown("#### Variables explicatives retenues dans le modele OLS")
        for var, sign, color, interp in ols_vars:
            # Code couleur du badge selon le niveau de significativite
            sig_color = "green" if "***" in sign else "orange" if "**" in sign else "blue"
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
            padding:10px 14px;margin-bottom:5px;background:#0a0d14;border:1px solid #1e2740;
            border-radius:8px;font-family:JetBrains Mono,monospace;font-size:12px'>
              <span style='color:{color};font-weight:700;min-width:160px'>{var}</span>
              <span style='color:#94a3b8;flex:1;padding:0 12px'>{interp}</span>
              <span>{badge(sign, sig_color)}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>#### Tests de validite du modele OLS", unsafe_allow_html=True)
        prose("""Trois tests classiques ont ete appliques pour valider les hypotheses du
        modele OLS. L'heteroscedasticite et la non-normalite des residus sont des
        resultats attendus sur des donnees operationnelles de ce type : la distribution
        du temps de traitement etant fortement asymetrique, les residus le sont
        mecaniquement. Ces violations n'invalident pas les coefficients mais invitent
        a interpreter les intervalles de confiance avec prudence ou a recourir a des
        ecarts-types robustes (HC3).""")

        # ── Resultats des tests de validite ──────────────────────────────────
        # Valeurs issues du notebook econometrie.ipynb, presentees en dur
        # Format : (nom_test, valeur_seuil, interpretation, couleur_badge)
        tests = [
            ("Shapiro-Wilk (normalite residus)",          "p < 0.05",
             "Rejet de la normalite asymptotique", "orange"),
            ("Breusch-Pagan (heteroscedasticite)",        "p < 0.05",
             "Heteroscedasticite presente — IC a corriger", "orange"),
            ("VIF (multicolinearite)",                     "< 5 pour toutes les vars",
             "Pas de colinearite critique", "green"),
        ]
        for test, val, interp, c in tests:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
            padding:9px 14px;margin-bottom:5px;background:#0a0d14;border:1px solid #1e2740;
            border-radius:8px;font-family:JetBrains Mono,monospace;font-size:12px'>
              <span style='color:#e2e8f0;flex:1'>{test}</span>
              <span style='color:#64748b;margin:0 12px;font-size:11px'>{val}</span>
              <span>{badge(interp, c)}</span>
            </div>""", unsafe_allow_html=True)

        insight("""<b>Lecture des coefficients :</b> Dans un modele OLS bien specifie,
        chaque coefficient s'interprete toutes choses egales par ailleurs. Si le coefficient
        de nb_intervenants est par exemple de +8.5, cela signifie que chaque intervenant
        supplementaire ajoute en moyenne 8.5 minutes au traitement, independamment du type
        d'intervention ou de l'outil utilise. Cette lecture directe est l'avantage principal
        de l'econometrie sur le ML : elle produit des estimations interpretables et
        communicables aupres de la direction metier.""")

    with tab2:
        # ── KPIs du modele Logit ──────────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("Variable cible",       "TOP.VR (0/1)",    "Vehicule de remplacement")
        col2.metric("Pseudo R2 (McFadden)", "0.15 — 0.25",     "Logit binomial")

        prose("""Le modele Logit modelise la <b>probabilite qu'un vehicule de remplacement
        soit active</b> pour un dossier donne. Le vehicule de remplacement (TOP.VR) est
        active dans <b>10.1%</b> des dossiers, ce qui en fait un evenement relativement
        rare mais a fort impact operationnel et financier. Le Pseudo-R2 de McFadden
        compris entre 0.15 et 0.25 indique un modele de qualite satisfaisante pour des
        donnees comportementales de ce type. Les coefficients sont exprimes en log-odds
        et les effets marginaux ci-dessous indiquent la direction et la significativite
        de chaque variable sur la probabilite d'activation.""")

        # ── Effets marginaux du Logit ─────────────────────────────────────────
        # Format : (nom_variable, direction "+"/"-", interpretation_metier)
        # Couleur verte pour effet positif, rouge pour effet negatif
        logit_vars = [
            ("nb_intervenants",   "+",
             "Chaque intervenant supplementaire augmente la probabilite de VR"),
            ("TOP.D.R",           "+",
             "Un dossier avec depannage/remorquage est plus susceptible de generer un VR"),
            ("TOP.Rappat.valide", "–",
             "Si rapatriement valide, le VR est moins probable (vehicule recupre)"),
            ("Outil MCS",         "–",
             "L'outil MCS est associe a des dossiers sans VR"),
            ("Mois ete",          "+",
             "Les mois estivaux augmentent la probabilite de VR (deplacements vacances)"),
        ]

        st.markdown("#### Effets marginaux du modele Logit")
        for var, direction, interp in logit_vars:
            color = "#10b981" if direction == "+" else "#ef4444"
            st.markdown(f"""
            <div style='display:flex;gap:14px;align-items:center;
            padding:9px 14px;margin-bottom:5px;background:#0a0d14;
            border:1px solid #1e2740;border-radius:8px;
            font-family:JetBrains Mono,monospace;font-size:12px'>
              <span style='color:{color};font-size:18px;font-weight:900;min-width:20px'>{direction}</span>
              <span style='color:#00d4ff;min-width:165px'>{var}</span>
              <span style='color:#94a3b8'>{interp}</span>
            </div>""", unsafe_allow_html=True)

        # ── Distribution de TOP.VR ────────────────────────────────────────────
        # Calcul en direct depuis dossiers_clean
        # Illustre le fort desequilibre de classes : ~10% de 1 vs ~90% de 0
        if "TOP.VR" in dos.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            vr_counts = dos["TOP.VR"].value_counts().reset_index()
            vr_counts.columns = ["TOP.VR", "n"]
            fig = px.bar(vr_counts, x="TOP.VR", y="n",
                         title="Distribution de TOP.VR dans les dossiers",
                         color_discrete_sequence=[COLORS[1]])
            apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        prose("""La distribution de TOP.VR confirme le fort desequilibre de classes :
        environ <b>10% de dossiers positifs</b> contre 90% de negatifs. Ce desequilibre
        a ete pris en compte dans l'estimation du Logit. L'effet positif du <b>mois d'ete</b>
        est particulierement interessant d'un point de vue metier : les deplacements de
        vacances impliquent des immobilisations plus longues loin du domicile, ce qui rend
        le vehicule de remplacement quasi-indispensable. Cette information est actionnable
        pour dimensionner le parc de vehicules de remplacement en anticipation des pics estivaux.""")

        insight("""<b>Synthese econometrique :</b> Les deux modeles convergent vers les memes
        conclusions. Le <b>nb_intervenants</b> est le determinant central du temps de
        traitement (OLS) et est aussi un predicteur significatif de l'activation du vehicule
        de remplacement (Logit). Le type d'intervention (TOP.D.R, TOP.VR, TOP.Rappat.valide)
        joue un role structurant dans les deux modeles. Ces resultats valident la coherence
        interne du projet : les signaux identifies par l'econometrie sont exactement ceux
        que le Machine Learning retrouve de facon independante via l'importance des variables.
        Cette double convergence renforce la fiabilite des conclusions.""", color="purple")


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
#  Barre de credit en bas de chaque page.
#  Rendu via du HTML inline avec une couleur tres discrete (#1e2740).
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:40px 0 20px;
font-family:JetBrains Mono,monospace;font-size:11px;color:#1e2740'>
  Assistance Automobile · Donnees 2021-2022 · Pipeline Python · Streamlit
</div>""", unsafe_allow_html=True)