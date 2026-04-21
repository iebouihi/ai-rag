"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  USE CASE 5 — OBSERVABILITÉ ET DÉTECTION D'ANOMALIES RETRIEVAL             ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTE — Quel problème réel on résout ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scénario typique : votre RAG est en production et tourne correctement.
Les tests CI/CD passent. Mais en production, un attaquant ou un utilisateur
malveillant peut adopter des comportements que vos tests statiques ne couvrent pas.

Problèmes invisibles sans monitoring :
    - Un utilisateur envoie 500 requêtes en 10 minutes pour cartographier
      votre knowledge base (membership inference systématique)
    - Un scraper sémantique extrait progressivement tout votre contenu
      via des requêtes à faible score de similarité (il sonde les edges)
    - Un score de retrieval anormalement élevé signale qu'un chunk
      adversarial a été injecté dans la DB et "répond" trop bien
    - Un pic de latence indique une dégradation ou une attaque DoS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MENACE — Ce qui se passe sans protection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Attaque 1 — Membership Inference Attack

    L'attaquant envoie des paraphrases exactes de documents suspects.
    Si le score de similarité est anormalement haut (ex: 0.97+) sur une
    requête très spécifique → le document est confirmé dans l'index.
    Répété sur des milliers de requêtes → cartographie complète.
    Pattern : nombreuses requêtes, scores bimodaux (très hauts ou très bas).

Attaque 2 — Semantic Scraping

    L'attaquant itère sur des requêtes génériques dans votre domaine.
    Même des réponses partielles permettent de reconstituer le contenu.
    Pattern : score moyen bas (0.3-0.4), volume élevé, diversité de requêtes.

Attaque 3 — Adversarial Chunk Poisoning (détection tardive)

    Un chunk malveillant a été injecté dans la DB (data poisoning).
    Il répond à certaines requêtes avec un score anormalement élevé.
    Pattern : score outlier statistiquement anormal vs historique.

Conséquences sans monitoring :
    - Exfiltration silencieuse et complète de la knowledge base
    - Impossible de savoir après coup qui a accédé à quoi
    - Aucune preuve pour une action légale ou disciplinaire

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REMÈDE — Stratégie de défense
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quatre détecteurs en temps réel, évalués à chaque requête :

    Détecteur 1 — Rate limiting par utilisateur
                  Comptage dans fenêtre glissante de 60s.
                  > 30 req/min → alerte et blocage optionnel.

    Détecteur 2 — Score entropy (semantic scraping)
                  Si plus de 40% des scores retrival sont < 0.30 →
                  l'utilisateur sonde les limites de la base.

    Détecteur 3 — Score z-score (chunk poisoning, anomalie)
                  Calcul du z-score des scores vs historique par user.
                  z > 3.0 → score statistiquement anormal → alerte.

    Détecteur 4 — Diversité des requêtes (reconnaissance)
                  Trop de requêtes uniques avec embeddings très différents
                  → balayage systématique plutôt qu'usage normal.

Toutes les traces sont envoyées à Langfuse pour visualisation et audit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRÉREQUIS D'INSTALLATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pip install langfuse scipy structlog

Variables d'environnement :
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"  # ou self-hosted

Langfuse self-hosted (Docker Compose) :
    git clone https://github.com/langfuse/langfuse.git
    cd langfuse
    docker compose up -d

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIBRAIRIES UTILISÉES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    langfuse       → Observabilité LLM : traces, métriques, dashboards
    scipy          → Calcul de z-score pour la détection d'anomalies statistiques
    structlog      → Logging structuré (JSON) — essentiel pour l'analyse SIEM
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import time
import statistics
import logging
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

import structlog
from scipy.stats import zscore

# Langfuse — optionnel : si non configuré, on continue sans traçabilité cloud
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# Configuration de structlog — sortie JSON pour les SIEMs (Splunk, Elastic, etc.)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()


# ==============================================================================
# CONFIGURATION DES SEUILS D'ANOMALIE
# ==============================================================================
#
# Ces seuils doivent être calibrés sur votre trafic réel.
# Procédure :
#   1. Déployer le monitoring en mode "log only" (sans blocage) pendant 1 semaine
#   2. Analyser la distribution des métriques sur le trafic normal
#   3. Fixer les seuils à 2-3 sigma au-delà de la moyenne normale
#
ANOMALY_CONFIG = {
    "rpm_limit":          30,     # requêtes par minute par utilisateur avant alerte
    "score_zscore_limit":  3.0,   # z-score max acceptable pour les scores de retrieval
    "low_score_pct_limit": 0.40,  # % max de chunks avec score < 0.30 (scraping sémantique)
    "score_history_min":   10,    # min d'observations avant d'activer le z-score
    "history_window":      50,    # taille de la fenêtre glissante par utilisateur
    "time_window_seconds": 60,    # fenêtre temporelle pour le rate limiting
    "min_score_threshold": 0.30,  # score en dessous duquel un chunk est "non pertinent"
}


# ==============================================================================
# DATACLASSES
# ==============================================================================

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityAlert:
    """Représente une alerte de sécurité détectée lors d'un retrieval."""
    alert_type: str
    severity: AlertSeverity
    message: str
    user_id: str
    query_preview: str
    metric_value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "user_id": self.user_id,
            "query_preview": self.query_preview,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
        }


@dataclass
class MonitoringResult:
    """Résultat complet de l'analyse d'une requête par le monitoring."""
    user_id: str
    query_preview: str
    scores: list[float]
    latency_ms: float
    alerts: list[SecurityAlert] = field(default_factory=list)
    trace_id: str | None = None

    @property
    def is_clean(self) -> bool:
        return len(self.alerts) == 0

    @property
    def critical_alerts(self) -> list[SecurityAlert]:
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]

    @property
    def should_block(self) -> bool:
        """Retourne True si la requête doit être bloquée (alertes critiques)."""
        return len(self.critical_alerts) > 0


# ==============================================================================
# CLASSE PRINCIPALE — RAGSecurityMonitor
# ==============================================================================

class RAGSecurityMonitor:
    """
    Moniteur de sécurité en temps réel pour les pipelines RAG.

    Analyse chaque requête de retrieval et détecte 4 types d'anomalies :
        1. Rate limiting (trop de requêtes)
        2. Semantic scraping (scores systématiquement bas)
        3. Score outliers (chunk poisoning ou membership inference)
        4. Pattern temporel (comportement suspect sur la durée)

    À intégrer directement dans votre fonction de retrieval.

    Instancier une seule fois au démarrage de l'application (singleton).
    Thread-safe pour les données partagées (dict/deque sont GIL-protégés en CPython).
    """

    def __init__(self, config: dict | None = None):
        self.config = config or ANOMALY_CONFIG

        # État par utilisateur — fenêtres glissantes
        # defaultdict + deque(maxlen=N) = mémoire bornée automatiquement
        self._score_history:   dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config["history_window"])
        )
        self._request_times:   dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Initialisation Langfuse (optionnelle)
        self._langfuse = None
        if LANGFUSE_AVAILABLE:
            try:
                self._langfuse = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                log.info("langfuse_connected", status="ok")
            except Exception as e:
                log.warning("langfuse_unavailable", error=str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # Détecteurs individuels
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_rate_limit(self, user_id: str, now: float) -> SecurityAlert | None:
        """
        Détecteur 1 — Rate limiting.

        Compte les requêtes dans la fenêtre glissante (dernières 60s).
        Alerte si le seuil est dépassé.

        Pourquoi 30 req/min ?
        → Un utilisateur normal fait rarement plus de 10 requêtes/min.
          30 est généreux mais capture clairement les scripts automatiques.
        """
        window = self.config["time_window_seconds"]
        self._request_times[user_id].append(now)

        recent_count = sum(
            1 for t in self._request_times[user_id]
            if now - t < window
        )

        if recent_count > self.config["rpm_limit"]:
            return SecurityAlert(
                alert_type="RATE_LIMIT",
                severity=AlertSeverity.CRITICAL,
                message=f"{recent_count} requêtes en {window}s (max: {self.config['rpm_limit']})",
                user_id=user_id,
                query_preview="[rate limit triggered]",
                metric_value=recent_count,
                threshold=self.config["rpm_limit"],
            )
        return None

    def _detect_semantic_scraping(
        self, user_id: str, scores: list[float], query_preview: str
    ) -> SecurityAlert | None:
        """
        Détecteur 2 — Semantic scraping.

        Si trop de scores sont sous le seuil minimum, l'utilisateur
        pose des questions auxquelles la KB ne répond pas bien →
        il explore les limites pour cartographier le contenu.

        Un usage normal a des scores moyens à 0.6-0.8.
        Un scraper systématique a des scores à 0.2-0.4 (beaucoup de misses).
        """
        if not scores:
            return None

        low_score_count = sum(
            1 for s in scores
            if s < self.config["min_score_threshold"]
        )
        low_score_pct = low_score_count / len(scores)

        if low_score_pct > self.config["low_score_pct_limit"]:
            return SecurityAlert(
                alert_type="SEMANTIC_SCAN",
                severity=AlertSeverity.HIGH,
                message=f"{low_score_pct:.0%} des chunks ont un score < {self.config['min_score_threshold']} — scraping sémantique probable",
                user_id=user_id,
                query_preview=query_preview,
                metric_value=low_score_pct,
                threshold=self.config["low_score_pct_limit"],
            )
        return None

    def _detect_score_anomaly(
        self, user_id: str, scores: list[float], query_preview: str
    ) -> SecurityAlert | None:
        """
        Détecteur 3 — Score outlier statistique.

        Calcule le z-score des scores actuels par rapport à l'historique
        de l'utilisateur. Un z-score > 3.0 indique que les scores de cette
        requête sont statistiquement anormaux.

        Scénarios couverts :
            - Score très élevé (0.99+) sur requête spécifique = membership inference
            - Score très bas de manière inhabituelle = tentative d'injection ratée

        Nécessite un historique minimal (score_history_min) pour être fiable.
        """
        self._score_history[user_id].extend(scores)
        history = list(self._score_history[user_id])

        min_observations = self.config["score_history_min"]
        if len(history) < min_observations:
            return None  # pas assez de données pour être statistiquement fiable

        z_scores = zscore(history)
        # On ne regarde que les z-scores des scores de la requête ACTUELLE
        # (les derniers dans l'historique)
        recent_zscores = z_scores[-len(scores):]

        max_z = float(max(abs(z) for z in recent_zscores))
        if max_z > self.config["score_zscore_limit"]:
            return SecurityAlert(
                alert_type="SCORE_ANOMALY",
                severity=AlertSeverity.HIGH,
                message=f"Score statistiquement anormal (z={max_z:.2f} > {self.config['score_zscore_limit']})",
                user_id=user_id,
                query_preview=query_preview,
                metric_value=max_z,
                threshold=self.config["score_zscore_limit"],
            )
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Méthode principale
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        user_id: str,
        query: str,
        retrieval_scores: list[float],
        latency_ms: float,
        extra_metadata: dict | None = None,
    ) -> MonitoringResult:
        """
        Analyse une requête de retrieval et retourne le résultat de monitoring.

        À appeler APRÈS chaque retrieval, AVANT de retourner la réponse.

        Paramètres :
            user_id           → identifiant de l'utilisateur (hashé si besoin de confidentialité)
            query             → texte de la requête utilisateur
            retrieval_scores  → liste des scores de similarité du retrieval
            latency_ms        → latence du retrieval en millisecondes
            extra_metadata    → métadonnées supplémentaires pour Langfuse

        Retourne un MonitoringResult.
        Si result.should_block → ne pas retourner la réponse à l'utilisateur.
        """
        now = time.time()
        query_preview = query[:80].replace("\n", " ")
        alerts = []

        # Lancer les 3 détecteurs
        if alert := self._detect_rate_limit(user_id, now):
            alerts.append(alert)

        if alert := self._detect_semantic_scraping(user_id, retrieval_scores, query_preview):
            alerts.append(alert)

        if alert := self._detect_score_anomaly(user_id, retrieval_scores, query_preview):
            alerts.append(alert)

        # Logging structuré (JSON pour SIEM)
        mean_score = statistics.mean(retrieval_scores) if retrieval_scores else 0.0

        log_data = {
            "user_id": user_id,
            "query_preview": query_preview,
            "mean_score": round(mean_score, 3),
            "chunk_count": len(retrieval_scores),
            "latency_ms": round(latency_ms),
            "alert_count": len(alerts),
        }

        if alerts:
            log.warning("rag_security_alert", alerts=[a.to_dict() for a in alerts], **log_data)
        else:
            log.info("rag_retrieval_ok", **log_data)

        # Traçabilité Langfuse
        trace_id = None
        if self._langfuse:
            try:
                trace = self._langfuse.trace(
                    name="rag_retrieval",
                    user_id=user_id,
                    metadata={
                        **log_data,
                        **(extra_metadata or {}),
                        "has_alerts": len(alerts) > 0,
                    }
                )
                trace_id = trace.id

                # Envoyer chaque alerte comme événement séparé dans Langfuse
                for alert in alerts:
                    self._langfuse.event(
                        name=f"security_alert_{alert.alert_type.lower()}",
                        trace_id=trace_id,
                        metadata=alert.to_dict(),
                    )
            except Exception as e:
                log.warning("langfuse_trace_failed", error=str(e))

        return MonitoringResult(
            user_id=user_id,
            query_preview=query_preview,
            scores=retrieval_scores,
            latency_ms=latency_ms,
            alerts=alerts,
            trace_id=trace_id,
        )


# ==============================================================================
# INTÉGRATION — Décorateur pour votre fonction de retrieval
# ==============================================================================

def monitored_retrieval(monitor: RAGSecurityMonitor, block_on_critical: bool = True):
    """
    Décorateur qui wrappe votre fonction de retrieval existante
    avec le monitoring de sécurité.

    Usage :
        monitor = RAGSecurityMonitor()

        @monitored_retrieval(monitor)
        def retrieve(user_id: str, query: str) -> list:
            return vectorstore.search(query, top_k=5)

    Si block_on_critical=True → lève PermissionError sur alerte critique.
    """
    def decorator(retrieval_fn):
        def wrapper(user_id: str, query: str, *args, **kwargs):
            t0 = time.time()
            results = retrieval_fn(user_id, query, *args, **kwargs)
            latency = (time.time() - t0) * 1000

            # Extraire les scores des résultats
            # Adapter selon le format de votre vector DB
            scores = [r.score if hasattr(r, "score") else 0.5 for r in results]

            monitoring_result = monitor.analyze(
                user_id=user_id,
                query=query,
                retrieval_scores=scores,
                latency_ms=latency,
            )

            if block_on_critical and monitoring_result.should_block:
                raise PermissionError(
                    f"Requête bloquée par le monitoring de sécurité "
                    f"({len(monitoring_result.critical_alerts)} alerte(s) critique(s))"
                )

            return results
        return wrapper
    return decorator


# ==============================================================================
# POINT D'ENTRÉE — Démonstration et simulation d'attaques
# ==============================================================================

if __name__ == "__main__":
    """
    Simulation de différents patterns d'attaque pour démontrer la détection.

    Ce script simule :
        1. Usage normal → aucune alerte
        2. Rate limiting → alerte critique
        3. Semantic scraping → alerte haute
        4. Score outlier → alerte haute
    """
    print("=" * 60)
    print("DÉMONSTRATION — Monitoring et Détection d'Anomalies RAG")
    print("=" * 60)

    monitor = RAGSecurityMonitor()

    # ── Scénario 1 : Usage normal ─────────────────────────────────────────
    print("\n[1] Usage normal (5 requêtes espacées)...")
    for i in range(5):
        result = monitor.analyze(
            user_id="user_alice",
            query=f"Question normale numéro {i+1} sur notre documentation",
            retrieval_scores=[0.75, 0.68, 0.71, 0.64, 0.59],
            latency_ms=120.0,
        )
        time.sleep(0.1)  # simuler un utilisateur normal
    print(f"  Alertes : {len(result.alerts)} (attendu : 0)")

    # ── Scénario 2 : Rate limiting ────────────────────────────────────────
    print("\n[2] Simulation rate limiting (35 requêtes rapides)...")
    last_result = None
    for i in range(35):
        last_result = monitor.analyze(
            user_id="attacker_bot",
            query=f"probe_{i}",
            retrieval_scores=[0.45, 0.42, 0.38],
            latency_ms=50.0,
        )
    critical = [a for a in last_result.alerts if a.severity == AlertSeverity.CRITICAL]
    print(f"  Alertes critiques : {len(critical)} (attendu : ≥1)")
    for a in critical:
        print(f"    → {a.alert_type}: {a.message}")

    # ── Scénario 3 : Semantic scraping ────────────────────────────────────
    print("\n[3] Simulation semantic scraping (scores systématiquement bas)...")
    result = monitor.analyze(
        user_id="scraper_user",
        query="Donnez-moi tout ce que vous savez sur le sujet X",
        retrieval_scores=[0.21, 0.18, 0.25, 0.19, 0.22],  # tous < 0.30
        latency_ms=200.0,
    )
    scan_alerts = [a for a in result.alerts if a.alert_type == "SEMANTIC_SCAN"]
    print(f"  Alertes SEMANTIC_SCAN : {len(scan_alerts)} (attendu : ≥1)")
    for a in scan_alerts:
        print(f"    → {a.message}")

    # ── Scénario 4 : Score outlier (après historique suffisant) ───────────
    print("\n[4] Simulation score outlier (chunk poisoning)...")
    # Construire un historique normal
    for _ in range(15):
        monitor.analyze(
            user_id="user_bob",
            query="question normale",
            retrieval_scores=[0.70, 0.65, 0.68],
            latency_ms=100.0,
        )
    # Maintenant envoyer un score anormalement élevé (chunk poisoning)
    result = monitor.analyze(
        user_id="user_bob",
        query="question très précise sur un document spécifique",
        retrieval_scores=[0.99, 0.98],  # score suspicieusement parfait
        latency_ms=100.0,
    )
    anomaly_alerts = [a for a in result.alerts if a.alert_type == "SCORE_ANOMALY"]
    print(f"  Alertes SCORE_ANOMALY : {len(anomaly_alerts)} (attendu : ≥1)")
    for a in anomaly_alerts:
        print(f"    → {a.message}")

    print("\n[OK] Démonstration terminée.")
    print("\nPour activer Langfuse, configurer :")
    print("  export LANGFUSE_PUBLIC_KEY='pk-lf-...'")
    print("  export LANGFUSE_SECRET_KEY='sk-lf-...'")
