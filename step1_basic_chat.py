from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()  

def chat(user_message):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content


answer = chat("What is diabetes?")
print(answer)