
def contextualize_question(llm, question, chat_history=None):
    """
    Convert a follow-up question into a standalone question
    using the previous conversation.
    """

    if not chat_history:
        return question

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-10:]
    )

    prompt = f"""
You are a question contextualization assistant.

Rewrite the latest student question as a standalone question
using the conversation history.

Rules:
1. Preserve the original meaning.
2. Resolve references such as it, this, that, and they.
3. Do not answer the question.
4. Do not add unrelated information.
5. Return only the rewritten question.

Conversation history:
{history_text}

Latest question:
{question}

Standalone question:
"""

    try:
        response = llm.invoke(prompt)
        rewritten_question = response.content.strip()

        return rewritten_question or question

    except Exception as error:
        print("Question contextualization error:", error)
        return question
