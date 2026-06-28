from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Загрузка обученной ML-модели
try:
    with open("models/gb_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Внимание: Модель не найдена. Сначала запустите model_training.py")

# Порог срабатывания, обоснованный в статье
TAU_THRESHOLD = 0.70

@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint для интеграции со средой AnyLogic.
    Принимает вектор телеметрии X_t, возвращает Risk_Score и флаг тревоги.
    """
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    
    # Извлечение вектора телеметрии (X_t)
    features = np.array([[
        data.get('spindle_vibration_hz', 50.0),
        data.get('bearing_temperature_c', 70.0),
        data.get('current_deviation_a', 0.0)
    ]])
    
    # Вычисление Risk Score = Model.predict_proba(X_t)
    risk_score = model.predict_proba(features)[0][1]
    
    # Логика принятия решения: R(t) >= tau
    trigger_maintenance = bool(risk_score >= TAU_THRESHOLD)
    
    response = {
        "risk_score": float(risk_score),
        "threshold": TAU_THRESHOLD,
        "trigger_maintenance": trigger_maintenance,
        "status": "Профилактика" if trigger_maintenance else "Штатная работа"
    }
    
    return jsonify(response)
if __name__ == "__main__":
    # Локальный сервер для двунаправленной интеграции ML <-> AnyLogic
    app.run(host='0.0.0.0', port=5000, debug=False)
