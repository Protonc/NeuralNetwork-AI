import cohere

# Replace this with your actual API key from cohere.com
co = cohere.Client("Oii2a0ySDIvi7fAaces1A5Pj4Yju42SYupNWcrxd")

print("🤖 Cohere Chatbot (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = co.chat(
        model="command-r-plus",
        message=user_input
    )
    
    print("Bot:", response.text)
