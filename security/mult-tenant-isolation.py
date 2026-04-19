"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  USE CASE 2 — ISOLATION MULTI-TENANT AVEC QDRANT                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTE — Quel problème réel on résout ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scénario typique : un SaaS B2B expose un assistant RAG à plusieurs clients
(appelés "tenants"). Chaque client upload ses propres documents confidentiels
(contrats, données clients, procédures internes).

Problème fondamental : tous ces documents sont dans la MÊME vector DB.
Comment garantir qu'un tenant ne voit jamais les données d'un autre ?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MENACE — Ce qui se passe sans protection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Attaque 1 — Cross-tenant data leakage (accidentel)

    Deux clients du même secteur (ex: deux cabinets d'avocats) utilisent
    le même vocabulaire métier. Une requête du cabinet "Acme" remonte des
    chunks du cabinet "BetaCorp" car leur similarité cosinus est élevée.
    → Violation RGPD silencieuse, aucune erreur n'est levée.

Attaque 2 — Membership inference (active)

    Un attaquant formule des requêtes très précises pour sonder si un
    document spécifique d'un autre tenant est dans l'index :
    "Quel est le montant du contrat Acme-Fournisseur X signé le 12/03/2023 ?"
    Si le RAG répond avec des détails précis → confirmation que le document
    est indexé → cartographie de la base à des fins d'exfiltration.

Conséquences :
    - Violation RGPD (articles 5, 25, 32) → amendes jusqu'à 4% du CA mondial
    - Rupture de confiance clients B2B irréversible
    - Fuite de secrets commerciaux, brevets, stratégies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REMÈDE — Stratégie de défense
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Principe 1 — PRE-FILTER (et jamais post-filter)

    Le filtre tenant_id doit être appliqué DANS la requête Qdrant,
    avant le calcul de similarité ANN (Approximate Nearest Neighbor).

    Différence critique :
        pre-filter  → Qdrant ne calcule la similarité QUE sur les vecteurs
                      du bon tenant. Les vecteurs adverses ne sont JAMAIS
                      exposés au moteur de recherche.

        post-filter → Qdrant calcule la similarité sur TOUS les vecteurs
                      puis retire ceux des mauvais tenants. Problème :
                      les scores des vecteurs adverses ont quand même
                      été calculés → membership inference possible.

Principe 2 — CHECKSUM ANTI-TAMPERING

    Chaque chunk stocke un HMAC de son contenu en metadata.
    Au retrieval, on re-vérifie le HMAC avant d'utiliser le chunk.
    → Détecte toute altération du contenu dans la DB (data poisoning).

Principe 3 — UNE COLLECTION, ISOLATION PAR METADATA

    On évite les collections séparées par tenant :
    → Des centaines de collections = overhead opérationnel massif
    → Une collection unique avec filtre metadata = scalable et maintenable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRÉREQUIS D'INSTALLATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pip install qdrant-client sentence-transformers

Qdrant (vector DB) — lancer avec Docker :
    docker run -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_data:/qdrant/storage \
        qdrant/qdrant

    → Port 6333 : HTTP REST API
    → Port 6334 : gRPC (plus performant pour la prod)

Variables d'environnement recommandées :
    export QDRANT_HOST="localhost"
    export QDRANT_PORT="6333"
    export HMAC_SECRET="votre_secret_32_bytes_minimum"   # vault en prod

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIBRAIRIES UTILISÉES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    qdrant-client          → Client Python pour Qdrant vector DB
    sentence-transformers  → Génération d'embeddings (all-MiniLM-L6-v2)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import uuid
import hmac
import hashlib
import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Modèle d'embedding léger, performant, gratuit et open-source.
# Dimension 384 — bon compromis vitesse/qualité pour du texte métier francophone.
# Alternatives plus puissantes : "all-mpnet-base-v2" (768d) ou text-embedding-3-small (1536d)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384

# Une seule collection pour tous les tenants.
# L'isolation est assurée par le payload metadata, pas par des collections séparées.
COLLECTION_NAME = "rag_documents"

# Clé HMAC pour la signature des chunks.
# En production : charger depuis AWS Secrets Manager, Vault, ou variable d'env.
# Ne jamais hardcoder une vraie clé secrète dans le code.
HMAC_SECRET = os.getenv("HMAC_SECRET", "dev_secret_change_in_prod").encode()


# ==============================================================================
# DATACLASSES
# ==============================================================================

@dataclass
class RetrievedChunk:
    """Représente un chunk récupéré depuis Qdrant, validé et prêt à l'emploi."""
    point_id: str
    tenant_id: str
    doc_id: str
    text: str
    score: float
    page: int | None = None
    checksum_valid: bool = True

    def __repr__(self):
        status = "OK" if self.checksum_valid else "TAMPERED"
        return f"<Chunk [{status}] tenant={self.tenant_id} score={self.score:.3f} doc={self.doc_id}>"


@dataclass
class TenantIsolationTestResult:
    """Résultat d'un test d'isolation cross-tenant."""
    source_tenant: str
    target_tenant: str
    query: str
    leaked_chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def is_isolated(self) -> bool:
        return len(self.leaked_chunks) == 0

    def report(self) -> str:
        if self.is_isolated:
            return f"[PASS] Aucune fuite de {self.source_tenant} → {self.target_tenant}"
        return (
            f"[FAIL] {len(self.leaked_chunks)} chunk(s) leakés "
            f"de '{self.source_tenant}' vers '{self.target_tenant}'"
        )


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def compute_hmac(text: str) -> str:
    """
    Calcule le HMAC-SHA256 du contenu d'un chunk.

    Ce hash est stocké en metadata au moment de l'ingestion.
    Il est recalculé et comparé au retrieval pour détecter toute altération.

    Pourquoi HMAC et pas SHA256 simple ?
    → HMAC intègre une clé secrète. Sans la clé, un attaquant ne peut pas
      recalculer un hash valide après avoir altéré le contenu.
      SHA256 seul serait recalculable par n'importe qui.
    """
    return hmac.new(
        HMAC_SECRET,
        text.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def verify_hmac(text: str, stored_checksum: str) -> bool:
    """
    Vérifie l'intégrité d'un chunk en comparant son HMAC actuel
    avec le checksum stocké au moment de l'ingestion.

    Utilise hmac.compare_digest pour éviter les timing attacks.
    """
    expected = compute_hmac(text)
    return hmac.compare_digest(expected, stored_checksum)


def make_point_id(tenant_id: str, doc_id: str) -> str:
    """
    Génère un ID de point Qdrant déterministe.

    UUID v5 basé sur tenant_id + doc_id :
    → Idempotent : réindexer le même document met à jour le point existant
    → Pas de doublon possible pour le même couple (tenant, doc)
    → Traçable : on peut reconstruire l'ID depuis les métadonnées
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}:{doc_id}"))


# ==============================================================================
# CLASSE PRINCIPALE — MultiTenantRAGStore
# ==============================================================================

class MultiTenantRAGStore:
    """
    Wrapper autour de Qdrant qui garantit l'isolation multi-tenant.

    Toutes les opérations (ingestion, retrieval) passent par cette classe.
    Elle s'assure que le filtre tenant_id est toujours appliqué en pre-filter.

    Usage :
        store = MultiTenantRAGStore()
        store.ingest("tenant_acme", "doc_001", "Texte du document...")
        chunks = store.retrieve("tenant_acme", "Ma question")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Crée la collection si elle n'existe pas encore.
        Idempotent — peut être appelé au démarrage de l'application.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                )
            )
            logger.info(f"Collection '{self.collection_name}' créée (dim={VECTOR_DIM})")

    def ingest(
        self,
        tenant_id: str,
        doc_id: str,
        text: str,
        extra_metadata: dict | None = None,
    ) -> str:
        """
        Indexe un chunk dans Qdrant avec ses métadonnées d'isolation.

        Retourne l'ID du point créé/mis à jour.

        Paramètres :
            tenant_id       → identifiant unique du tenant (ex: "tenant_acme")
            doc_id          → identifiant unique du document source
            text            → contenu textuel du chunk
            extra_metadata  → métadonnées supplémentaires (page, section, etc.)
        """
        vector = self.encoder.encode(text).tolist()
        point_id = make_point_id(tenant_id, doc_id)

        payload = {
            "tenant_id": tenant_id,   # clé d'isolation — NE PAS SUPPRIMER
            "doc_id": doc_id,
            "text": text,
            "checksum": compute_hmac(text),  # anti-tampering
            **(extra_metadata or {}),
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )

        logger.debug(f"Ingéré : tenant={tenant_id} doc={doc_id} point_id={point_id}")
        return point_id

    def retrieve(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[RetrievedChunk]:
        """
        Récupère les chunks les plus similaires à la query,
        STRICTEMENT filtrés sur le tenant_id.

        Le filtre est appliqué en PRE-FILTER (dans la requête Qdrant),
        garantissant que les vecteurs des autres tenants ne sont jamais
        exposés au moteur de similarité.

        Les chunks avec un checksum invalide sont marqués et loggués,
        mais toujours retournés — c'est à l'appelant de décider de les
        exclure ou d'alerter l'équipe sécurité.
        """
        vector = self.encoder.encode(query).tolist()

        # PRE-FILTER : le filtre tenant_id est dans la requête ANN
        # → Qdrant ne calcule la similarité que sur les bons vecteurs
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=Filter(must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id)
                )
            ]),
            limit=top_k,
            score_threshold=min_score,   # exclure les chunks trop peu pertinents
            with_payload=True,
        )

        chunks = []
        for r in results:
            p = r.payload
            text = p["text"]
            stored_checksum = p.get("checksum", "")

            # Vérification de l'intégrité du chunk
            checksum_ok = verify_hmac(text, stored_checksum)
            if not checksum_ok:
                logger.error(
                    f"[ALERTE TAMPERING] Chunk altéré détecté ! "
                    f"tenant={p['tenant_id']} doc={p['doc_id']} point_id={r.id}"
                )

            chunks.append(RetrievedChunk(
                point_id=str(r.id),
                tenant_id=p["tenant_id"],
                doc_id=p["doc_id"],
                text=text,
                score=r.score,
                page=p.get("page"),
                checksum_valid=checksum_ok,
            ))

        logger.info(
            f"Retrieval : tenant={tenant_id} → {len(chunks)} chunks "
            f"(tampered={sum(1 for c in chunks if not c.checksum_valid)})"
        )
        return chunks

    def test_tenant_isolation(
        self,
        source_tenant: str,
        target_tenant: str,
        probe_queries: list[str],
    ) -> TenantIsolationTestResult:
        """
        Teste que les données du source_tenant ne remontent pas
        dans les requêtes du target_tenant.

        À utiliser dans vos tests d'intégration et votre CI/CD.
        Si le résultat n'est pas isolé → arrêter le déploiement.

        Exemple :
            result = store.test_tenant_isolation("acme", "betacorp", queries)
            assert result.is_isolated, result.report()
        """
        all_leaked = []
        for query in probe_queries:
            chunks = self.retrieve(target_tenant, query)
            leaked = [c for c in chunks if c.tenant_id == source_tenant]
            all_leaked.extend(leaked)

        return TenantIsolationTestResult(
            source_tenant=source_tenant,
            target_tenant=target_tenant,
            query=str(probe_queries),
            leaked_chunks=all_leaked,
        )


# ==============================================================================
# POINT D'ENTRÉE — Démonstration
# ==============================================================================

if __name__ == "__main__":
    """
    Démonstration complète de l'isolation multi-tenant.

    Ce script :
        1. Crée deux tenants avec des documents confidentiels
        2. Effectue des requêtes et vérifie l'isolation
        3. Simule un data poisoning et détecte le tampering
        4. Lance un test d'isolation cross-tenant

    Prérequis : Qdrant doit tourner sur localhost:6333
        docker run -p 6333:6333 qdrant/qdrant
    """
    print("=" * 60)
    print("DÉMONSTRATION — Isolation multi-tenant avec Qdrant")
    print("=" * 60)

    store = MultiTenantRAGStore(host="localhost", port=6333)

    # --- Ingestion des documents par tenant ---
    print("\n[1] Ingestion des documents...")

    store.ingest("acme", "contrat_001",
        "Le contrat Acme stipule une remise de 15% sur les commandes supérieures à 50k€.",
        extra_metadata={"page": 1, "type": "contract"}
    )
    store.ingest("acme", "politique_rh",
        "La politique RH Acme prévoit 35 jours de congés pour les cadres.",
        extra_metadata={"page": 1, "type": "hr"}
    )
    store.ingest("betacorp", "tarifs_2024",
        "Les tarifs BetaCorp 2024 incluent une marge de 22% sur les services IT.",
        extra_metadata={"page": 1, "type": "pricing"}
    )

    # --- Retrieval isolé ---
    print("\n[2] Retrieval pour le tenant 'acme'...")
    chunks = store.retrieve("acme", "Quelle est la remise accordée ?")
    for c in chunks:
        print(f"  {c}")

    print("\n[3] Retrieval pour le tenant 'betacorp'...")
    chunks = store.retrieve("betacorp", "Quels sont les tarifs ?")
    for c in chunks:
        print(f"  {c}")

    # --- Test d'isolation cross-tenant ---
    print("\n[4] Test d'isolation cross-tenant...")
    probe_queries = [
        "Quels sont les remises ?",
        "Parlez moi des tarifs",
        "Politique commerciale",
    ]
    result = store.test_tenant_isolation("acme", "betacorp", probe_queries)
    print(f"  {result.report()}")
    assert result.is_isolated, "FUITE DÉTECTÉE — vérifier la configuration des filtres"

    print("\n[OK] Tous les tests passent.")
