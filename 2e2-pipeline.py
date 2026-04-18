"""
Pipeline RAG complet avec LangChain
====================================
Architecture : ingestion → chunking → embedding → vector store → retrieval → génération

Dépendances :
    pip install langchain langchain-openai langchain-community chromadb tiktoken pypdf
"""

import os
from pathlib import Path

# ── LangChain core ──────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — INGESTION
# ════════════════════════════════════════════════════════════════════════════

def load_documents(source_dir: str = "./docs") -> list[Document]:
    """
    Charge tous les PDFs d'un répertoire.
    Alternatives : WebBaseLoader, CSVLoader, UnstructuredMarkdownLoader…
    """
    loader = DirectoryLoader(
        source_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"[Ingestion] {len(documents)} pages chargées depuis {source_dir}")
    return documents


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHUNKING
# ════════════════════════════════════════════════════════════════════════════

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Découpe les documents en chunks.

    Paramètres clés :
    - chunk_size     : taille en tokens (512–1024 typiquement)
    - chunk_overlap  : chevauchement pour ne pas couper le contexte (10–15%)
    - separators     : ordre de priorité pour la coupure
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Chunking] {len(documents)} pages → {len(chunks)} chunks")
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — EMBEDDING + VECTOR STORE
# ════════════════════════════════════════════════════════════════════════════

def build_vector_store(
    chunks: list[Document],
    persist_dir: str = "./chroma_db",
) -> Chroma:
    """
    Encode les chunks en vecteurs et les stocke dans ChromaDB.

    En production, remplacer Chroma par :
    - Pinecone  → cloud managé, scalable
    - Weaviate  → open-source, hybrid search natif
    - pgvector  → si vous êtes déjà sur PostgreSQL
    - Qdrant    → performant, self-hosted
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # ou "text-embedding-3-large" pour + de précision
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    print(f"[Vector Store] {len(chunks)} chunks indexés dans {persist_dir}")
    return vectorstore


def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """Recharge un index existant (évite de ré-indexer à chaque démarrage)."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — RETRIEVAL + GÉNÉRATION
# ════════════════════════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE = """
Tu es un assistant expert. Réponds à la question en te basant UNIQUEMENT
sur le contexte fourni ci-dessous. Si la réponse n'est pas dans le contexte,
dis-le clairement — ne fabrique pas de réponse.

Contexte :
-----------
{context}
-----------

Question : {question}

Réponse (cite les sources si possible) :
"""


def build_rag_chain(vectorstore: Chroma) -> RetrievalQA:
    """
    Assemble le retriever + LLM en une chaîne RAG.

    Paramètres retriever :
    - search_type   : "similarity" | "mmr" (max marginal relevance, + diversité)
    - k             : nombre de chunks retournés (3–5 en général)
    - score_threshold : filtrer les chunks peu pertinents (0.0–1.0)
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,        # 0 = réponses déterministes, idéal pour RAG factuel
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",      # MMR évite les chunks redondants
        search_kwargs={
            "k": 5,
            "fetch_k": 20,      # pool initial avant sélection MMR
        },
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",     # "stuff" = concatène tous les chunks dans le prompt
                                # Alternatives : "map_reduce" si beaucoup de chunks
                                #               "refine" pour réponses itératives
        retriever=retriever,
        return_source_documents=True,   # renvoie les chunks utilisés (pour citations)
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def query(chain: RetrievalQA, question: str) -> dict:
    """Pose une question et retourne la réponse avec les sources."""
    result = chain.invoke({"query": question})

    print(f"\nQuestion : {question}")
    print(f"\nRéponse :\n{result['result']}")
    print("\nSources utilisées :")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "inconnue")
        page   = doc.metadata.get("page", "?")
        print(f"  [{i}] {source} — page {page}")
        print(f"      {doc.page_content[:120].strip()}…")

    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN — PIPELINE COMPLET
# ════════════════════════════════════════════════════════════════════════════

def main():
    """
    Exemple d'utilisation complet.
    Adaptez SOURCE_DIR et les questions à votre cas d'usage.
    """
    SOURCE_DIR  = "./docs"
    PERSIST_DIR = "./chroma_db"

    # ── Étape 1 : indexation (à faire une seule fois) ──────────────────────
    if not Path(PERSIST_DIR).exists():
        print("=== Première indexation ===")
        documents = load_documents(SOURCE_DIR)
        chunks    = split_documents(documents)
        vectorstore = build_vector_store(chunks, PERSIST_DIR)
    else:
        print("=== Chargement de l'index existant ===")
        vectorstore = load_vector_store(PERSIST_DIR)

    # ── Étape 2 : construction de la chaîne RAG ────────────────────────────
    chain = build_rag_chain(vectorstore)

    # ── Étape 3 : requêtes ─────────────────────────────────────────────────
    questions = [
        "Quelle est la politique de confidentialité des données ?",
        "Quelles sont les étapes d'onboarding pour un nouveau client ?",
        "Quels sont les SLAs garantis dans le contrat ?",
    ]

    for q in questions:
        query(chain, q)
        print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    # Assurez-vous que OPENAI_API_KEY est défini dans votre environnement
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY manquant. Exportez-le avant de lancer.")
    main()


# ════════════════════════════════════════════════════════════════════════════
# VARIANTE : HYBRID SEARCH (dense + sparse BM25)
# ════════════════════════════════════════════════════════════════════════════
#
# Pour de meilleurs résultats en production, combinez recherche vectorielle
# et recherche lexicale (BM25) :
#
#   pip install langchain-community rank_bm25
#
#   from langchain.retrievers import BM25Retriever, EnsembleRetriever
#
#   bm25_retriever = BM25Retriever.from_documents(chunks)
#   bm25_retriever.k = 3
#
#   dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#
#   hybrid_retriever = EnsembleRetriever(
#       retrievers=[bm25_retriever, dense_retriever],
#       weights=[0.4, 0.6],   # ajuster selon vos données
#   )
#
#   # Puis passer hybrid_retriever à RetrievalQA.from_chain_type(...)
#
# ════════════════════════════════════════════════════════════════════════════
# VARIANTE : AJOUT D'UN RERANKER (Cohere)
# ════════════════════════════════════════════════════════════════════════════
#
#   pip install cohere langchain-cohere
#
#   from langchain.retrievers import ContextualCompressionRetriever
#   from langchain_cohere import CohereRerank
#
#   reranker = CohereRerank(
#       model="rerank-multilingual-v3.0",
#       top_n=3,
#       cohere_api_key=os.getenv("COHERE_API_KEY"),
#   )
#
#   compressed_retriever = ContextualCompressionRetriever(
#       base_compressor=reranker,
#       base_retriever=dense_retriever,
#   )
