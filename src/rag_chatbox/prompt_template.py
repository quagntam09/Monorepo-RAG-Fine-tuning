from __future__ import annotations

# Shared synthesis prompt to keep CLI and service behavior consistent.
PROMPT_TEMPLATE = """You are an intelligent assistant. Answer the question based on the following context and extracted answer candidate.
{context}

Reader Candidate Answer:
{reader_answer}
Reader Candidate Score:
{reader_span_score}

Previous Conversation History:
{chat_history}

Requirements:
- Only answer based on the provided context. Ignore personal/world knowledge not present in the context.
- If the answer is not explicitly supported by the context, say you don't know.
- Prefer the Reader Candidate Answer when it is supported by context.
- Always respond in the same language as the user's question. If the user's question is in Vietnamese, respond in Vietnamese.
- At the end of your response, add a section named `Nguồn:` with bullet points that only reference file/page pairs present in the context.
- Do not cite a source that is not present in the context block above.

Question: {question}
Answer:"""
