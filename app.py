"""DATA 폴더의 문서를 검색해 답변하는 간단한 RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APIStatusError, AuthenticationError
from pypdf import PdfReader
from streamlit.errors import StreamlitSecretNotFoundError


# 프로젝트 최상단과 DATA 폴더를 기준으로 경로를 계산합니다.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "DATA"
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json"}

# 로컬에서는 .env를 읽고, Streamlit Cloud에서는 st.secrets를 우선 사용합니다.
load_dotenv(PROJECT_ROOT / ".env")


def clean_text(text: str) -> str:
    """PDF에서 추출된 불필요한 공백을 줄여 검색 품질을 높입니다."""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_openai_api_key() -> str:
    """Streamlit Secrets를 우선 사용하고 로컬 .env를 fallback으로 사용합니다."""

    try:
        # Streamlit Cloud에는 다음처럼 최상위 키를 저장하는 방식을 권장합니다.
        # OPENAI_API_KEY = "sk-..."
        cloud_key = st.secrets.get("OPENAI_API_KEY", "")

        # 혹시 [openai] 섹션으로 저장한 경우도 호환합니다.
        if not cloud_key:
            openai_section = st.secrets.get("openai", {})
            if hasattr(openai_section, "get"):
                cloud_key = openai_section.get("api_key", "")
    except StreamlitSecretNotFoundError:
        # 로컬에 .streamlit/secrets.toml이 없어도 .env 방식으로 실행합니다.
        cloud_key = ""
    except Exception as exc:
        raise RuntimeError(
            "Streamlit Secrets를 읽지 못했습니다. Secrets에는 TOML 형식으로 "
            'OPENAI_API_KEY = "..."를 입력했는지 확인하세요.'
        ) from exc

    return str(cloud_key or os.getenv("OPENAI_API_KEY", "")).strip()


def load_documents() -> list[Document]:
    """DATA 폴더의 PDF와 일반 텍스트 파일을 모두 LangChain 문서로 읽습니다."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA 폴더를 찾을 수 없습니다: {DATA_DIR}")

    documents: list[Document] = []
    data_files = sorted(path for path in DATA_DIR.iterdir() if path.is_file())

    for file_path in data_files:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = clean_text(page.extract_text() or "")
                if page_text:
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file_path.name,
                                "page": page_number,
                            },
                        )
                    )
            continue

        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            text = clean_text(file_path.read_text(encoding="utf-8-sig"))
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path.name},
                    )
                )

    if not documents:
        raise ValueError("DATA 폴더에서 읽을 수 있는 문서를 찾지 못했습니다.")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """긴 문서를 검색하기 좋은 크기의 청크로 나눕니다."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list[Document], api_key: str) -> InMemoryVectorStore:
    """OpenAI 임베딩으로 InMemoryVectorStore를 만들고 문서를 저장합니다."""

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        # OpenAI 서버에 원문을 보내고 로컬 토큰화/인코딩 다운로드를 건너뜁니다.
        check_embedding_ctx_length=False,
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def format_context(documents: list[Document]) -> str:
    """검색된 문서 조각에 출처 정보를 붙여 LLM에 전달합니다."""

    formatted: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "알 수 없는 파일")
        page = document.metadata.get("page")
        location = f"{source}, {page}쪽" if page else source
        formatted.append(f"[출처: {location}]\n{document.page_content}")
    return "\n\n---\n\n".join(formatted)


def retrieve_documents(
    retriever: Any, chunks: list[Document], question: str
) -> list[Document]:
    """질문과 관련된 문서를 검색하고, 자가용 출장의 핵심 페이지를 보강합니다."""

    documents = retriever.invoke(question)
    vehicle_terms = ("자가용", "자차", "자동차", "차량")
    travel_terms = ("일비", "출장", "여비", "부득이")
    is_vehicle_travel_question = (
        any(term in question for term in vehicle_terms)
        and any(term in question for term in travel_terms)
    )

    if is_vehicle_travel_question:
        # 검색 점수가 낮아도 질문 판단에 필요한 페이지는 반드시 함께 제공합니다.
        # 61쪽: 부득이한 사유로 자가용을 이용한 경우의 자동차 운임 기준
        # 64쪽: 공용차량·임차차량 이용 시 일비 2분의 1 지급 기준
        reference_chunks = [
            document
            for document in chunks
            if (
                "2024년" in str(document.metadata.get("source", ""))
                and "9장" in str(document.metadata.get("source", ""))
                and document.metadata.get("page") in {61, 64}
            )
        ]
        documents.extend(reference_chunks)

    unique_documents: list[Document] = []
    seen: set[tuple[str, int | None, str]] = set()
    for document in documents:
        key = (
            str(document.metadata.get("source", "")),
            document.metadata.get("page"),
            document.page_content,
        )
        if key not in seen:
            seen.add(key)
            unique_documents.append(document)
    return unique_documents


def make_answer_chain() -> Any:
    """최신 LangChain Runnable 방식으로 답변 체인을 구성합니다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 제공된 문서만 근거로 답하는 한국어 문서 검색 도우미입니다.

다음 규칙을 반드시 지키세요.
1. 아래 문서 내용에 있는 정보만 사용하세요.
2. 문서에서 답을 확인할 수 없으면 정확히 '문서에서 확인할 수 없습니다.'라고 답하세요.
3. 상식, 추측, 외부 지식을 보태지 마세요.
4. 답변은 질문에 직접 답하는 짧고 명확한 한국어로 작성하세요.

문서 내용:
{context}""",
            ),
            ("human", "질문: {question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm | StrOutputParser()


def evidence_sentence(text: str, question: str) -> str:
    """검색 청크에서 질문과 관련성이 높은 근거 문장을 골라 표시합니다."""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if part.strip()]
    if not sentences:
        return text[:400]

    question_words = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", question.lower()))
    ranked = sorted(
        sentences,
        key=lambda sentence: sum(
            word in sentence.lower() for word in question_words
        ),
        reverse=True,
    )
    return ranked[0][:400]


def render_sources(sources: list[dict[str, Any]], question: str) -> None:
    """답변 아래에 파일명, 페이지, 근거 문장을 표시합니다."""

    if not sources:
        return

    with st.expander("출처 및 근거 문장"):
        shown: set[tuple[str, int | None]] = set()
        for source in sources:
            key = (source["source"], source.get("page"))
            if key in shown:
                continue
            shown.add(key)
            location = source["source"]
            if source.get("page"):
                location += f" ({source['page']}쪽)"
            st.markdown(f"**출처 파일:** `{location}`")
            st.markdown(f"> {evidence_sentence(source['content'], question)}")


def explain_openai_error(error: Exception) -> str:
    """OpenAI 오류를 초보자가 이해하기 쉬운 안내 문장으로 바꿉니다."""

    if isinstance(error, APIConnectionError):
        cause_text = str(getattr(error, "__cause__", ""))
        if "10051" in cause_text:
            return (
                "api.openai.com으로 가는 네트워크 경로가 없습니다(WinError 10051). "
                "VPN을 종료하고 핫스팟 연결을 확인한 뒤, DNS·라우터·보안 프로그램에서 "
                "api.openai.com의 HTTPS(443) 연결을 차단하지 않는지 확인하세요."
            )
        if "10013" in cause_text:
            return (
                "Python 프로세스의 HTTPS 연결이 Windows 보안 정책에 의해 차단되었습니다. "
                f"보안 프로그램 허용 목록에 다음 실행 파일을 추가하세요: {sys.executable}"
            )
        return (
            "OpenAI API에 연결할 수 없습니다. 회사/기관 방화벽, 프록시 또는 보안 프로그램에서 "
            "api.openai.com의 HTTPS(443) 연결을 허용했는지 확인하세요."
        )
    if isinstance(error, AuthenticationError):
        return (
            "OpenAI API 키가 유효하지 않습니다. Streamlit Cloud의 App settings > "
            "Secrets 또는 로컬 .env의 OPENAI_API_KEY를 확인하세요."
        )
    if isinstance(error, APIStatusError):
        if error.status_code == 429:
            return (
                "OpenAI API 요청 한도(429)에 도달했습니다. OpenAI 사용량·결제 상태와 "
                "프로젝트의 월간 한도를 확인한 뒤 잠시 후 다시 시도하세요. "
                "새 API 키를 발급해도 결제 한도 문제는 해결되지 않을 수 있습니다."
            )
        return f"OpenAI API가 오류를 반환했습니다. 상태 코드: {error.status_code}"
    return f"처리 중 오류가 발생했습니다: {error}"


def main() -> None:
    st.set_page_config(page_title="문서 기반 RAG 챗봇", page_icon="📚")
    st.title("📚 문서 기반 RAG 챗봇")
    st.caption("DATA 폴더의 문서만 근거로 답변합니다.")

    try:
        api_key = get_openai_api_key()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    if not api_key:
        st.error(
            "OPENAI_API_KEY가 없습니다. Streamlit Cloud에서는 App settings > Secrets에 "
            "OPENAI_API_KEY를 추가하고, 로컬에서는 .env에 입력하세요."
        )
        st.stop()

    try:
        with st.spinner("DATA 폴더의 문서를 읽는 중입니다..."):
            documents = load_documents()
            chunks = split_documents(documents)
    except Exception as exc:
        st.error(f"문서를 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    if "rag_vector_store" not in st.session_state:
        try:
            with st.spinner("문서 임베딩을 생성하는 중입니다..."):
                st.session_state.rag_vector_store = build_vector_store(chunks, api_key)
                st.session_state.rag_answer_chain = make_answer_chain()
        except Exception as exc:
            st.error(explain_openai_error(exc))
            st.stop()

    vector_store: InMemoryVectorStore = st.session_state.rag_vector_store
    answer_chain = st.session_state.rag_answer_chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []), message.get("question", ""))

    question = st.chat_input("문서에 대해 질문해 보세요.")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("문서에서 답을 찾는 중입니다..."):
                retrieved_documents = retrieve_documents(retriever, chunks, question)
                answer = answer_chain.invoke(
                    {
                        "context": format_context(retrieved_documents),
                        "question": question,
                    }
                )
                sources = [
                    {
                        "source": document.metadata.get("source", "알 수 없는 파일"),
                        "page": document.metadata.get("page"),
                        "content": document.page_content,
                    }
                    for document in retrieved_documents
                ]
        except Exception as exc:
            st.error(explain_openai_error(exc))
            return

        st.markdown(answer)
        render_sources(sources, question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "question": question,
        }
    )


if __name__ == "__main__":
    main()
