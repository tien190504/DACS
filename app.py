from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import ta

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/price')
def get_price():
    symbol = request.args.get('symbol', 'XAUUSD=X')
    interval = request.args.get('interval', '15m')
    data = yf.download(symbol, period='5d', interval=interval)
    data.reset_index(inplace=True)
    data['time'] = data['Datetime'].astype(str)
    return data[['time', 'Open', 'High', 'Low', 'Close', 'Volume']].to_json(orient='records')

@app.route('/api/indicator')
def get_indicator():
    symbol = request.args.get('symbol', 'XAUUSD=X')
    indicator = request.args.get('indicator', 'rsi')
    interval = request.args.get('interval', '15m')
    data = yf.download(symbol, period='5d', interval=interval)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data.reset_index(inplace=True)
    return jsonify({
        'rsi': data['RSI'].fillna('').tolist(),
        'close': data['Close'].tolist(),
        'time': data['Datetime'].astype(str).tolist()
    })

@app.route('/api/marketdata')
def market_data():
    symbol = request.args.get('symbol', 'XAUUSD=X')
    info = yf.Ticker(symbol).info
    last_price = info.get('regularMarketPrice', 0)
    return jsonify({
        'symbol': symbol,
        'price': last_price,
        'currency': info.get('currency', 'USD'),
        'performance': {
            '1w': info.get('52WeekChange', 0),
            '1y': info.get('52WeekChange', 0),
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    # Nhận prompt từ user (ví dụ dùng mô hình GPT hoặc model custom)
    data = request.json
    prompt = data.get('prompt', '')
    # Tích hợp model dự báo của bạn ở đây
    result = f"Đây là kết quả mô phỏng cho prompt: '{prompt}'."
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
