"""
Pipeline RAG complet avec LlamaIndex
======================================
Architecture : ingestion → chunking → embedding → index → retrieval → génération

LlamaIndex est pensé nativement pour le RAG : ses abstractions correspondent
exactement aux étapes du pipeline, sans la verbosité de LangChain.

Dépendances :
    pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
    pip install llama-index-vector-stores-chroma chromadb pypdf
"""

import os
from pathlib import Path

# ── LlamaIndex core ──────────────────────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import (
    SentenceSplitter,          # chunking fixe/récursif
    SemanticSplitterNodeParser, # chunking sémantique (avancé)
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# LlamaIndex utilise un objet Settings global — plus besoin de passer
# llm et embeddings à chaque composant.
# ════════════════════════════════════════════════════════════════════════════

def configure(api_key: str):
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=api_key,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    # Taille de chunk et overlap définis globalement
    # (overridable par parseur si besoin)
    Settings.chunk_size = 800
    Settings.chunk_overlap = 100


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — INGESTION
# SimpleDirectoryReader gère PDF, TXT, DOCX, HTML, JSON, CSV…
# ════════════════════════════════════════════════════════════════════════════

def load_documents(source_dir: str = "./docs"):
    """
    Charge tous les fichiers supportés d'un répertoire.
    LlamaIndex les convertit automatiquement en objets Document
    avec métadonnées (nom fichier, page, etc.).
    """
    reader = SimpleDirectoryReader(
        input_dir=source_dir,
        recursive=True,          # sous-dossiers inclus
        required_exts=[".pdf", ".txt", ".md"],
    )
    documents = reader.load_data()
    print(f"[Ingestion] {len(documents)} documents chargés depuis {source_dir}")
    return documents


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHUNKING (parsing en nodes)
# LlamaIndex parle de "nodes" plutôt que "chunks" — même concept.
# ════════════════════════════════════════════════════════════════════════════

def parse_nodes_fixed(documents):
    """
    Chunking fixe/récursif — bon point de départ.
    Équivalent du RecursiveCharacterTextSplitter de LangChain.
    """
    parser = SentenceSplitter(
        chunk_size=800,
        chunk_overlap=100,
        paragraph_separator="\n\n",
    )
    nodes = parser.get_nodes_from_documents(documents)
    print(f"[Chunking fixed] {len(documents)} docs → {len(nodes)} nodes")
    return nodes


def parse_nodes_semantic(documents):
    """
    Chunking sémantique — coupe quand le sens change, pas à taille fixe.
    Plus précis mais plus lent (appel embedding par phrase).
    Requiert que Settings.embed_model soit configuré.
    """
    parser = SemanticSplitterNodeParser(
        buffer_size=1,           # phrases de contexte de chaque côté
        breakpoint_percentile_threshold=95,  # seuil de rupture sémantique
        embed_model=Settings.embed_model,
    )
    nodes = parser.get_nodes_from_documents(documents)
    print(f"[Chunking sémantique] {len(documents)} docs → {len(nodes)} nodes")
    return nodes


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — INDEX + VECTOR STORE
# ════════════════════════════════════════════════════════════════════════════

def build_index(nodes, persist_dir: str = "./chroma_db") -> VectorStoreIndex:
    """
    Encode les nodes en vecteurs et les stocke dans ChromaDB.

    En production, remplacer ChromaDB par :
    - Pinecone  → ChromaVectorStore → PineconeVectorStore
    - Weaviate  → WeaviateVectorStore
    - pgvector  → PGVectorStore
    - Qdrant    → QdrantVectorStore
    (imports depuis llama_index.vector_stores.*)
    """
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("rag_docs")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"[Index] {len(nodes)} nodes indexés dans {persist_dir}")
    return index


def load_index(persist_dir: str = "./chroma_db") -> VectorStoreIndex:
    """Recharge un index ChromaDB existant."""
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("rag_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — RETRIEVAL + RERANKING + GÉNÉRATION
# ════════════════════════════════════════════════════════════════════════════

RAG_PROMPT = PromptTemplate(
    """Tu es un assistant expert. Réponds à la question en te basant UNIQUEMENT
sur le contexte fourni. Si la réponse n'est pas dans le contexte, dis-le
clairement — ne fabrique pas de réponse.

Contexte :
-----------
{context_str}
-----------

Question : {query_str}

Réponse (cite les sources si possible) :
"""
)


def build_query_engine(index: VectorStoreIndex, use_reranker: bool = False):
    """
    Assemble le retriever + postprocessors + synthesizer en un query engine.

    search_type options :
    - "similarity"      : top-k par similarité cosinus (défaut)
    - "mmr"             : max marginal relevance (+ diversité)
    """
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=6,      # récupère 6 candidats
        vector_store_query_mode="default",
    )

    # ── Post-processors : filtrage + reranking ───────────────────────────
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.7),  # filtre les chunks peu pertinents
    ]

    if use_reranker:
        # Reranker LLM-based : re-note les chunks récupérés (+ précis, + lent)
        # Alternative : CohereRerank (plus rapide)
        node_postprocessors.append(
            LLMRerank(
                choice_batch_size=6,
                top_n=3,            # garde seulement les 3 meilleurs après reranking
            )
        )

    # ── Synthesizer : transforme les nodes en réponse ───────────────────
    # "compact"     : concatène les chunks, un seul appel LLM (défaut, économique)
    # "refine"      : itère chunk par chunk, meilleur pour longs documents
    # "tree_summarize" : structure arborescente, idéal pour résumés globaux
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=RAG_PROMPT,
        verbose=True,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
        response_synthesizer=response_synthesizer,
    )
    return query_engine


def query(engine, question: str):
    """Pose une question et affiche la réponse avec les sources."""
    response = engine.query(question)

    print(f"\nQuestion : {question}")
    print(f"\nRéponse :\n{response}")
    print("\nSources utilisées :")
    for i, node in enumerate(response.source_nodes, 1):
        source = node.metadata.get("file_name", "inconnue")
        page   = node.metadata.get("page_label", "?")
        score  = round(node.score or 0, 3)
        print(f"  [{i}] {source} — page {page} — score {score}")
        print(f"      {node.text[:120].strip()}…")

    return response


# ════════════════════════════════════════════════════════════════════════════
# MAIN — PIPELINE COMPLET
# ════════════════════════════════════════════════════════════════════════════

def main():
    SOURCE_DIR  = "./docs"
    PERSIST_DIR = "./chroma_db"
    API_KEY     = os.getenv("OPENAI_API_KEY")

    configure(API_KEY)

    # ── Indexation (une seule fois) ────────────────────────────────────────
    if not Path(PERSIST_DIR).exists():
        print("=== Première indexation ===")
        documents = load_documents(SOURCE_DIR)

        # Choix du chunking : fixed pour démarrer, sémantique pour la prod
        nodes = parse_nodes_fixed(documents)
        # nodes = parse_nodes_semantic(documents)  # activer pour la prod

        index = build_index(nodes, PERSIST_DIR)
    else:
        print("=== Chargement de l'index existant ===")
        index = load_index(PERSIST_DIR)

    # ── Query engine ────────────────────────────────────────────────────────
    engine = build_query_engine(index, use_reranker=False)
    # engine = build_query_engine(index, use_reranker=True)  # + précis, + lent

    # ── Requêtes ────────────────────────────────────────────────────────────
    questions = [
        "Quelle est la politique de remboursement ?",
        "Quels sont les délais de livraison garantis ?",
        "Comment contacter le support client ?",
    ]

    for q in questions:
        query(engine, q)
        print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY manquant.")
    main()


# ════════════════════════════════════════════════════════════════════════════
# VARIANTE : SMALL-TO-BIG (Parent Document Retrieval)
# Indexe des petits chunks pour la précision, renvoie le parent pour le contexte
# ════════════════════════════════════════════════════════════════════════════
#
#   from llama_index.core.node_parser import HierarchicalNodeParser
#   from llama_index.core.retrievers import AutoMergingRetriever
#   from llama_index.core.storage.docstore import SimpleDocumentStore
#
#   # Parser hiérarchique : crée 3 niveaux de granularité
#   parser = HierarchicalNodeParser.from_defaults(
#       chunk_sizes=[2048, 512, 128]  # grands → moyens → petits
#   )
#   nodes = parser.get_nodes_from_documents(documents)
#
#   # Stocke tous les niveaux dans le docstore
#   docstore = SimpleDocumentStore()
#   docstore.add_documents(nodes)
#   storage_ctx = StorageContext.from_defaults(
#       vector_store=vector_store,
#       docstore=docstore,
#   )
#   index = VectorStoreIndex(nodes, storage_context=storage_ctx)
#
#   # Retriever qui remonte automatiquement au parent si majoritaire
#   base_retriever = index.as_retriever(similarity_top_k=6)
#   retriever = AutoMergingRetriever(
#       base_retriever,
#       storage_context=storage_ctx,
#       verbose=True,
#   )
#
# ════════════════════════════════════════════════════════════════════════════
# VARIANTE : HYBRID SEARCH (dense + BM25)
# ════════════════════════════════════════════════════════════════════════════
#
#   pip install llama-index-retrievers-bm25
#
#   from llama_index.retrievers.bm25 import BM25Retriever
#   from llama_index.core.retrievers import QueryFusionRetriever
#
#   dense_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
#   bm25_retriever  = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
#
#   hybrid_retriever = QueryFusionRetriever(
#       retrievers=[dense_retriever, bm25_retriever],
#       similarity_top_k=5,
#       num_queries=1,           # pas de query expansion
#       mode="reciprocal_rerank", # fusion RRF (Reciprocal Rank Fusion)
#   )
#
# ════════════════════════════════════════════════════════════════════════════
# VARIANTE : SUB-QUESTION QUERY ENGINE
# Décompose une question complexe en sous-questions, interroge des index
# spécialisés, puis fusionne les réponses — point fort de LlamaIndex vs LangChain
# ════════════════════════════════════════════════════════════════════════════
#
#   from llama_index.core.query_engine import SubQuestionQueryEngine
#   from llama_index.core.tools import QueryEngineTool
#
#   tools = [
#       QueryEngineTool.from_defaults(
#           query_engine=engine_contrats,
#           name="contrats",
#           description="Informations sur les contrats clients",
#       ),
#       QueryEngineTool.from_defaults(
#           query_engine=engine_produits,
#           name="produits",
#           description="Catalogue et fiches produits",
#       ),
#   ]
#
#   sub_engine = SubQuestionQueryEngine.from_defaults(
#       query_engine_tools=tools,
#       verbose=True,
#   )
#   # Exemple : "Quel est le délai de retour pour le produit X selon le contrat Y ?"
#   # → LlamaIndex décompose automatiquement en 2 sous-questions
