from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined FAQs
faq_data = {
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Machine Learning?": "Machine Learning is a subset of AI that enables systems to learn from data.",
    "What is deep learning?": "Deep learning is a part of machine learning that uses neural networks.",
    "What is Python?": "Python is a high-level programming language used in AI, web development, and more.",
    "How does AI work?": "AI works by using data and algorithms to make decisions or predictions."
}

# Separate questions and answers
questions = list(faq_data.keys())
answers = list(faq_data.values())

# Chat function
def get_response(user_input):
    questions.append(user_input)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    idx = similarity.argmax()
    questions.pop()  # Remove user input from list
    return answers[idx]

# CLI Chat Loop
print("🤖 Chatbot is ready! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = get_response(user_input)
    print("Bot:", response)