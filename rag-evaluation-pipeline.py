"""
Pipeline d'évaluation RAGAS complet
=====================================
Hypothèse : votre agent RAG expose une fonction answerFaq(question: str)
qui retourne un dict :
    {
        "answer":   str,          # réponse générée par le LLM
        "contexts": list[str],    # chunks récupérés par le retriever
    }

Ce fichier gère tout le reste :
  1. Chargement du golden dataset
  2. Exécution de answerFaq sur chaque question
  3. Construction du Dataset RAGAS
  4. Calcul des 4 métriques principales
  5. Rapport détaillé + diagnostic automatique
  6. Export CSV + JSON
  7. Variantes : monitoring continu, CI/CD gate, métriques avancées

Dépendances :
    pip install ragas datasets openai langchain-openai pandas
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Any
from datasets import Dataset


# ════════════════════════════════════════════════════════════════════════════
# IMPORT DE VOTRE AGENT EXISTANT
# Adaptez cet import à votre structure de projet
# ════════════════════════════════════════════════════════════════════════════

# from my_rag_agent import answerFaq          # ← votre import réel

# ── Stub de démonstration (remplacez par votre import) ────────────────────
def answerFaq(question: str) -> dict[str, Any]:
    """
    STUB — remplacez par votre implémentation réelle.
    Votre fonction doit retourner :
        {
            "answer":   str,           # réponse du LLM
            "contexts": list[str],     # chunks récupérés (texte brut)
        }
    """
    return {
        "answer": f"[STUB] Réponse simulée pour : {question}",
        "contexts": [
            f"[STUB] Chunk pertinent A pour : {question}",
            f"[STUB] Chunk pertinent B pour : {question}",
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET
# Minimum 50–100 paires en production.
# Format : question + ground_truth (réponse attendue annotée par un expert).
# ════════════════════════════════════════════════════════════════════════════

GOLDEN_DATASET = [
    {
        "question": "Quel est le délai de remboursement après retour d'un produit ?",
        "ground_truth": "Le remboursement est effectué sous 5 à 7 jours ouvrés après réception du retour.",
    },
    {
        "question": "Comment contacter le support client en dehors des heures d'ouverture ?",
        "ground_truth": "En dehors des heures d'ouverture, vous pouvez envoyer un email à support@example.com ou utiliser le formulaire de contact disponible 24h/24.",
    },
    {
        "question": "Quels sont les modes de livraison disponibles ?",
        "ground_truth": "Trois modes de livraison sont disponibles : standard (3-5 jours), express (24h) et retrait en point relais.",
    },
    {
        "question": "Le service est-il disponible dans les pays de l'UE ?",
        "ground_truth": "Oui, le service est disponible dans l'ensemble des 27 pays membres de l'Union Européenne.",
    },
    {
        "question": "Comment modifier une commande déjà passée ?",
        "ground_truth": "Une commande peut être modifiée dans les 2 heures suivant sa validation, en contactant le support ou via l'espace client.",
    },
    {
        "question": "Quelle est la durée minimale d'engagement pour un abonnement ?",
        "ground_truth": "L'abonnement est sans engagement minimum. Vous pouvez résilier à tout moment depuis votre espace client.",
    },
    {
        "question": "Les données personnelles sont-elles partagées avec des tiers ?",
        "ground_truth": "Les données personnelles ne sont jamais vendues à des tiers. Elles peuvent être partagées avec des sous-traitants techniques dans le cadre strict de l'exécution du service.",
    },
    {
        "question": "Comment réinitialiser mon mot de passe ?",
        "ground_truth": "Cliquez sur 'Mot de passe oublié' sur la page de connexion, entrez votre email et suivez le lien reçu valable 24 heures.",
    },
    {
        "question": "Quelles garanties couvre le SAV produit ?",
        "ground_truth": "Le SAV couvre les défauts de fabrication pendant 2 ans. Les dommages accidentels et l'usure normale sont exclus.",
    },
    {
        "question": "Peut-on utiliser plusieurs codes promo sur une même commande ?",
        "ground_truth": "Non, un seul code promotionnel peut être appliqué par commande.",
    },
]


# ════════════════════════════════════════════════════════════════════════════
# SEUILS ET DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    "faithfulness":      0.90,   # Priorité 1 — hallucination = danger prod
    "answer_relevancy":  0.85,
    "context_precision": 0.80,
    "context_recall":    0.85,
}

DIAGNOSTICS = {
    "faithfulness": {
        "cause": "Le LLM génère des affirmations non vérifiables dans les chunks (hallucination).",
        "actions": [
            "Renforcer le prompt : 'Réponds UNIQUEMENT depuis le contexte fourni'",
            "Réduire temperature → 0.0",
            "Changer de LLM (GPT-4o, Claude Sonnet 4.5)",
            "Réduire le nombre de chunks injectés (moins de distraction)",
            "Ajouter une étape de vérification post-génération",
        ],
    },
    "answer_relevancy": {
        "cause": "La réponse ne répond pas directement à la question posée.",
        "actions": [
            "Retravailler le prompt template",
            "Tester query expansion ou HyDE",
            "Nettoyer les chunks parasites qui dévient la réponse",
            "Vérifier la cohérence du golden dataset (questions ambiguës ?)",
            "Réduire top_k pour moins de bruit contextuel",
        ],
    },
    "context_precision": {
        "cause": "Trop de chunks hors-sujet récupérés — bruit dans le contexte.",
        "actions": [
            "Réduire similarity_top_k (ex : 6 → 4)",
            "Augmenter similarity_cutoff (ex : 0.70 → 0.78)",
            "Ajouter un reranker (Cohere Rerank recommandé)",
            "Passer en hybrid retrieval (dense + BM25)",
            "Revoir la stratégie de chunking (chunks trop larges ?)",
        ],
    },
    "context_recall": {
        "cause": "Informations nécessaires absentes des chunks récupérés.",
        "actions": [
            "Augmenter similarity_top_k (ex : 4 → 7)",
            "Passer en hybrid retrieval (dense + BM25)",
            "Revoir le chunking (chunks trop petits/fragmentés ?)",
            "Essayer small-to-big / AutoMergingRetriever",
            "Améliorer le modèle d'embedding (BGE-M3, Cohere embed-v3)",
        ],
    },
}


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — EXÉCUTION DE ANSWERFAQ SUR TOUT LE DATASET
# ════════════════════════════════════════════════════════════════════════════

def run_rag_on_dataset(
    dataset: list[dict],
    delay_between_calls: float = 0.5,
) -> list[dict]:
    """
    Appelle answerFaq pour chaque question et collecte :
    question, answer, contexts, ground_truth.
    """
    results = []
    total = len(dataset)
    print(f"\n[1/4] Exécution de answerFaq sur {total} questions...")

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"  [{i:02d}/{total}] {question[:65]}…")

        try:
            response = answerFaq(question)

            # Validation du contrat de la fonction
            if not isinstance(response, dict):
                raise TypeError(f"answerFaq doit retourner un dict, reçu {type(response)}")
            if "answer" not in response:
                raise KeyError("Clé 'answer' manquante dans la réponse de answerFaq")
            if "contexts" not in response:
                raise KeyError("Clé 'contexts' manquante dans la réponse de answerFaq")
            if not isinstance(response["contexts"], list):
                raise TypeError("'contexts' doit être une list[str]")

            results.append({
                "question":     question,
                "answer":       str(response["answer"]),
                "contexts":     [str(c) for c in response["contexts"]],
                "ground_truth": item["ground_truth"],
            })

        except Exception as e:
            print(f"  ⚠ ERREUR question {i} : {e}")
            results.append({
                "question":     question,
                "answer":       "ERREUR — appel answerFaq échoué",
                "contexts":     ["ERREUR"],
                "ground_truth": item["ground_truth"],
            })

        if delay_between_calls > 0 and i < total:
            time.sleep(delay_between_calls)

    success = sum(1 for r in results if r["answer"] != "ERREUR — appel answerFaq échoué")
    print(f"  ✓ {success}/{total} appels réussis")
    return results


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — CONSTRUCTION DU DATASET RAGAS
# RAGAS attend un objet datasets.Dataset avec les colonnes exactes :
#   question | contexts | answer | ground_truth
# ════════════════════════════════════════════════════════════════════════════

def build_ragas_dataset(results: list[dict]) -> Dataset:
    """
    Convertit la liste de résultats en Dataset HuggingFace
    compatible avec ragas.evaluate().
    """
    data = {
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],  # list[list[str]]
        "ground_truth": [r["ground_truth"] for r in results],
    }
    dataset = Dataset.from_dict(data)
    print(f"\n[2/4] Dataset RAGAS construit : {len(dataset)} exemples")
    return dataset


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — CALCUL DES MÉTRIQUES RAGAS
# ════════════════════════════════════════════════════════════════════════════

def run_ragas_evaluation(
    dataset: Dataset,
    llm_model: str = "gpt-4o",
) -> tuple[dict, Any]:
    """
    Lance l'évaluation RAGAS avec les 4 métriques principales.

    RAGAS utilise un LLM juge pour noter chaque exemple.
    GPT-4o est recommandé pour la cohérence des notes.

    Temps estimé : ~1–3 min pour 50 exemples avec GPT-4o.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    print(f"\n[3/4] Calcul des métriques RAGAS (juge : {llm_model})…")
    print("      Chaque métrique est notée par le LLM — patientez 1–3 min.")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        raise_exceptions=False,     # continue même si un exemple échoue
    )

    scores = {
        "faithfulness":      round(float(result["faithfulness"]),      3),
        "answer_relevancy":  round(float(result["answer_relevancy"]),  3),
        "context_precision": round(float(result["context_precision"]), 3),
        "context_recall":    round(float(result["context_recall"]),    3),
    }
    scores["overall"] = round(sum(scores.values()) / 4, 3)

    print("  ✓ Métriques calculées")
    return scores, result


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — RAPPORT + DIAGNOSTIC AUTOMATIQUE
# ════════════════════════════════════════════════════════════════════════════

def print_report(scores: dict) -> None:
    """Affiche le rapport complet avec diagnostics dans la console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    W = 64

    print("\n" + "═" * W)
    print(f"  RAPPORT D'ÉVALUATION RAGAS — {ts}")
    print("═" * W)

    STATUS_ICON = {"OK": "✓", "Attention": "~", "Critique": "✗"}

    for metric, score in scores.items():
        if metric == "overall":
            continue
        thr = THRESHOLDS[metric]
        if score >= thr:
            status = "OK"
        elif score >= thr - 0.15:
            status = "Attention"
        else:
            status = "Critique"

        bar_filled = int(score * 32)
        bar = "█" * bar_filled + "░" * (32 - bar_filled)
        icon = STATUS_ICON[status]
        print(f"  {icon} {metric:<22} {bar}  {score:.3f}  (seuil {thr})")

    # Score global
    ov = scores["overall"]
    print(f"\n  Score global : {ov:.3f}")
    if   ov >= 0.85: verdict = "RAG PERFORMANT — monitorer en continu"
    elif ov >= 0.70: verdict = "RAG ACCEPTABLE — améliorations ciblées requises"
    elif ov >= 0.55: verdict = "RAG DÉGRADÉ — actions correctives urgentes"
    else:            verdict = "RAG CASSÉ — refactoring complet nécessaire"
    print(f"  Statut       : {verdict}")

    # Diagnostics uniquement pour les métriques sous seuil
    alerts = {
        k: v for k, v in scores.items()
        if k != "overall" and v < THRESHOLDS.get(k, 1.0)
    }
    if alerts:
        print("\n" + "─" * W)
        print("  DIAGNOSTICS ET ACTIONS CORRECTIVES")
        print("─" * W)
        for metric, score in alerts.items():
            d = DIAGNOSTICS[metric]
            print(f"\n  [{metric}]  score {score:.3f} < seuil {THRESHOLDS[metric]}")
            print(f"  Cause : {d['cause']}")
            print("  Actions correctives :")
            for action in d["actions"]:
                print(f"    → {action}")
    else:
        print("\n  Aucune alerte — tous les seuils sont respectés.")

    print("\n" + "═" * W)


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — EXPORT CSV + JSON
# ════════════════════════════════════════════════════════════════════════════

def export_results(
    scores: dict,
    ragas_result: Any,
    output_dir: str = "./ragas_output",
) -> None:
    """
    Exporte :
    - scores_TIMESTAMP.json      → scores agrégés + alertes + timestamp
    - details_TIMESTAMP.csv      → scores par question (debug)
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Résumé JSON ────────────────────────────────────────────────────────
    summary = {
        "timestamp":  ts,
        "scores":     scores,
        "thresholds": THRESHOLDS,
        "alerts": {
            k: {"score": v, "delta": round(v - THRESHOLDS[k], 3)}
            for k, v in scores.items()
            if k != "overall" and v < THRESHOLDS.get(k, 1.0)
        },
        "verdict": (
            "PERFORMANT" if scores["overall"] >= 0.85 else
            "ACCEPTABLE" if scores["overall"] >= 0.70 else
            "DÉGRADÉ"    if scores["overall"] >= 0.55 else
            "CASSÉ"
        ),
    }
    json_path = os.path.join(output_dir, f"scores_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] Résumé JSON → {json_path}")

    # ── Détails par question CSV ────────────────────────────────────────────
    try:
        df = ragas_result.to_pandas()
        csv_path = os.path.join(output_dir, f"details_{ts}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[Export] Détails CSV  → {csv_path}")

        # Top 5 questions les plus problématiques
        if "faithfulness" in df.columns:
            print("\n  Top 5 questions à problème (faithfulness le plus bas) :")
            worst = df.nsmallest(5, "faithfulness")[
                ["question", "faithfulness", "answer_relevancy", "context_precision"]
            ]
            for _, row in worst.iterrows():
                q = str(row["question"])[:55]
                f = f"{row['faithfulness']:.2f}"
                r = f"{row['answer_relevancy']:.2f}"
                print(f"  faith={f}  rel={r}  | {q}…")
    except Exception as e:
        print(f"[Export] Détails CSV non disponible : {e}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    print("╔═══════════════════════════════════════════════╗")
    print("║   Pipeline d'évaluation RAGAS                 ║")
    print("╚═══════════════════════════════════════════════╝")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY manquant. Exportez : export OPENAI_API_KEY=sk-..."
        )

    results      = run_rag_on_dataset(GOLDEN_DATASET, delay_between_calls=0.3)
    ragas_ds     = build_ragas_dataset(results)
    scores, raw  = run_ragas_evaluation(ragas_ds, llm_model="gpt-4o")
    print_report(scores)
    export_results(scores, raw, output_dir="./ragas_output")

    return scores


if __name__ == "__main__":
    final_scores = main()


# ════════════════════════════════════════════════════════════════════════════
# VARIANTE A : MONITORING CONTINU (production)
# Lance RAGAS sur un sous-échantillon aléatoire chaque jour
# ════════════════════════════════════════════════════════════════════════════
#
#   import random, schedule
#
#   def daily_eval():
#       sample  = random.sample(GOLDEN_DATASET, k=20)
#       results = run_rag_on_dataset(sample)
#       ds      = build_ragas_dataset(results)
#       scores, raw = run_ragas_evaluation(ds)
#       print_report(scores)
#       export_results(scores, raw)
#
#       # Alerte si faithfulness dégradé
#       if scores["faithfulness"] < THRESHOLDS["faithfulness"]:
#           send_slack_alert(f"ALERTE RAG — faithfulness = {scores['faithfulness']:.3f}")
#
#   schedule.every().day.at("06:00").do(daily_eval)
#   while True: schedule.run_pending(); time.sleep(60)
#
# ════════════════════════════════════════════════════════════════════════════
# VARIANTE B : CI/CD GATE (GitHub Actions / GitLab CI)
# Bloque le déploiement si les métriques passent sous un seuil dur
# ════════════════════════════════════════════════════════════════════════════
#
#   import sys
#
#   def ci_gate(scores: dict, hard_threshold: float = 0.75) -> None:
#       failing = {
#           k: v for k, v in scores.items()
#           if k != "overall" and v < hard_threshold
#       }
#       if failing:
#           print(f"\n[CI GATE] ÉCHEC — métriques sous {hard_threshold}:")
#           for k, v in failing.items():
#               print(f"  {k}: {v:.3f}")
#           sys.exit(1)    # bloque le pipeline CI
#       print(f"\n[CI GATE] OK — toutes les métriques >= {hard_threshold}")
#
#   # Après main() : ci_gate(final_scores, hard_threshold=0.75)
#
# ════════════════════════════════════════════════════════════════════════════
# VARIANTE C : MÉTRIQUES AVANCÉES
# ════════════════════════════════════════════════════════════════════════════
#
#   from ragas.metrics import (
#       answer_correctness,      # compare answer vs ground_truth (end-to-end)
#       answer_similarity,       # similarité sémantique answer / ground_truth
#       context_entity_recall,   # entités nommées dans les chunks
#   )
#   # Ajoutez-les dans la liste metrics=[...] de run_ragas_evaluation()
#   # answer_correctness est la métrique la plus complète côté utilisateur final
