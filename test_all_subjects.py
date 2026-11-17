
from rag import ask_vidya

tests = [
    ("Mathematics", "Explain the Pythagorean theorem", "Mathematics_Foundations.pdf"),
    ("Physics", "Explain Newton laws of motion", "Physics_Fundamentals.pdf"),
    ("Chemistry", "Explain the periodic table", "Chemistry_Fundamentals.pdf"),
    ("Biology", "Explain photosynthesis", "Biology_Fundamentals.pdf"),
    ("Computer Science", "Explain database management systems", "Computer_Science_Fundamentals.pdf"),
    ("Engineering", "Explain basic engineering principles", "Engineering_Principles.pdf"),
    ("Commerce & Management", "Explain demand and supply", "Commerce_and_Management_Fundamentals.pdf"),
    ("Languages", "Explain English communication skills", "English_Communication_Fundamentals.pdf"),
    ("AI & Data Science", "Explain machine learning", "AI_and_Data_Science_Fundamentals.pdf"),
    ("General Knowledge", "Explain effective study strategies", "Study_Skills_and_Learning_Strategies.pdf"),
]

print("\n========== VIDYA RAG SUBJECT TESTING ==========\n")

for subject, question, expected_source in tests:
    print(f"\nTesting: {subject}")
    print("-" * 55)

    try:
        answer = ask_vidya(question)

        expected_found = expected_source.lower() in answer.lower()
        answer_generated = len(answer.strip()) > 50

        if answer_generated and expected_found:
            status = "PASS"
        elif answer_generated:
            status = "CHECK SOURCE"
        else:
            status = "FAIL"

        print("Status:", status)
        print("Expected Source:", expected_source)

        if "**Knowledge Base Sources:**" in answer:
            sources = answer.split("**Knowledge Base Sources:**")[-1].strip()
            print("Retrieved Sources:", sources)
        else:
            print("Retrieved Sources: Not displayed")

    except Exception as error:
        print("Status: FAIL")
        print("Error:", error)

print("\n========== TESTING COMPLETED ==========")
