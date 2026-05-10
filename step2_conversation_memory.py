from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are MedRAG, a helpful medical information assistant.
Your job is to explain medical conditions, symptoms, and treatments
in simple, clear language that anyone can understand.

IMPORTANT RULES:
- Never provide a personal diagnosis
- Always recommend consulting a real doctor for personal health issues
- If asked about emergencies, tell the user to call emergency services immediately
- Base your answers on established medical knowledge only
- Keep answers clear and structured
"""

def chat(user_message, conversation_history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content

    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, conversation_history


history = []

print("--- Question 1 ---")
reply, history = chat("What causes high blood pressure?", history)
print(reply)

print("\n--- Question 2 (follow-up, it should remember the context) ---")
reply, history = chat("How is it treated?", history)
print(reply)