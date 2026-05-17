from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

GROUNDED_PROMPT = """
You are a helpful AI document assistant.

Answer ONLY from the provided context.

If the answer is not found in the context,
say:
"I cannot find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""


def build_rag_chain(vectorstore):

    prompt = PromptTemplate(
        template=GROUNDED_PROMPT,
        input_variables=[
            "context",
            "question"
        ],
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    class RAGChain:

        def __init__(
            self,
            retriever,
            prompt,
            llm
        ):

            self.retriever = retriever
            self.prompt = prompt
            self.llm = llm

        def invoke(self, input_dict):

            question = input_dict.get(
                "input",
                input_dict.get("question")
            )

            docs = self.retriever.invoke(question)

            if not isinstance(docs, list):
                docs = [docs]

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            final_prompt = self.prompt.format(
                context=context,
                question=question
            )

            response = self.llm.invoke(final_prompt)

            return {
                "answer": response.content,
                "context": docs,
            }

    return RAGChain(
        retriever,
        prompt,
        llm
    )


def ask_question(qa_chain, question):

    result = qa_chain.invoke({
        "input": question
    })

    docs = result["context"]

    return {
        "answer": result["answer"],
        "sources": [
            {
                "page": doc.metadata.get(
                    "page",
                    "N/A"
                ),
                "type": doc.metadata.get(
                    "type",
                    "text"
                ),
                "snippet": (
                    doc.page_content[:200] + "..."
                ),
            }
            for doc in docs
        ],
    }