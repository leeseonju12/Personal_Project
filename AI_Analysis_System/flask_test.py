from flask import Flask, jsonify, request

app = Flask(__name__)

# Spring이 GET 요청으로 데이터를 가져갈 수 있는 엔드포인트
@app.route('/api/data', methods=['GET'])
def send_data():
    data = {
        "message": "Hello from Flask!",
        "value": 42,
        "items": ["apple", "banana", "cherry"]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)