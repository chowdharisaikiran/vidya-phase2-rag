
from rag import ask_vidya


def main():

    print("\n========== VIDYA AI CHATBOT ==========")
    print("Ask any educational question.")
    print("Type 'exit' to stop.")
    print("======================================")

    chat_history = []

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nThank you for using Vidya!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            answer = ask_vidya(
                question,
                chat_history=chat_history
            )

            print("\nVidya:", answer)

            chat_history.append({
                "role": "user",
                "content": question
            })

            chat_history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:
            print("\nError:", error)


if __name__ == "__main__":
    main()
