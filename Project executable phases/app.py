from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained model for question answering
qa_pipeline = pipeline("question-answering")

@app.route('/query', methods=['POST'])
def query_document():
    data = request.json
    document = data.get('document')
    question = data.get('question')
    
    if not document or not question:
        return jsonify({"error": "Document and question are required"}), 400
    
    # Use the model to answer the question based on the document
    result = qa_pipeline(question=question, context=document)
    
    return jsonify({"answer": result['answer']})

if __name__ == '__main__':
    app.run(debug=True)