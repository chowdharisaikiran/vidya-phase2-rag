
from rag import ask_vidya

print("\n========== VIDYA AI CHATBOT ==========")
print("Ask any educational question.")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Vidya: Thank you! Happy learning.")
        break

    if not question:
        print("Vidya: Please enter a question.")
        continue

    try:
        print("\nVidya:", ask_vidya(question), "\n")
    except Exception as error:
        print("Error:", error)
