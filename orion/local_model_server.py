from flask import Flask, request, jsonify 
from download_and_load.downloadandload import ask# import the ask function from the downloadandload module
app=Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data=request.json
    messages=data['messages']
# Convert messages to prompt
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    # Get response from your local model
    response = ask(prompt)
    
    return jsonify({
        "choices": [{
            "message": {
                "role": "assistant", 
                "content": response
            }
        }],
        "usage": {"total_tokens": len(response.split())}
    })

if __name__ == '__main__':
    app.run(port=8000)