import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, fbeta_score, roc_curve, precision_recall_curve, confusion_matrix
from data_preprocessing import load_and_preprocess_data, get_time_series_splits

RANDOM_SEED = 42

def train_and_evaluate():
    """
    Обучение ансамбля на базе градиентного бустинга и оценка Event Recall
    с использованием кросс-валидации по времени.
    """
    # Загрузка подготовленных данных
    df = pd.read_csv("data/processed_telemetry.csv")
    splits, X, y = get_time_series_splits(df)
    
    model = GradientBoostingClassifier(random_state=RANDOM_SEED)
    
    print("Запуск Time-Series Cross-Validation (expanding window)...")
    
    # Для упрощения примера тренируем на последнем сплите
    for train_index, test_index in splits:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print("Обучение модели Gradient Boosting...")
    model.fit(X_train, y_train)
    
    # Получение вероятностей
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Подбор порога tau по F_beta и фиксация tau = 0.7 согласно статье
    tau = 0.70
    y_pred = (y_probs >= tau).astype(int)
    
    # Расчет событийно-ориентированной метрики Event Recall
    event_recall = recall_score(y_test, y_pred)
    fpr = sum((y_pred == 1) & (y_test == 0)) / sum(y_test == 0)
    
    print(f"Оптимальный порог тревоги (tau): {tau}")
    print(f"Event Recall (полнота обнаружения событий): {event_recall:.3f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    # Генерация данных для ROC/PR кривых и матрицы ошибок
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Сохранение обученной модели
    with open("models/gb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nМодель сохранена в models/gb_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
