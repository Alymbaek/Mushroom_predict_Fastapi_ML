🍄 Mushroom Classification & FastAPI Service  
Кратко: обучаем модели (Logistic Regression 📈 и Decision Tree 🌳) на датасете «Mushrooms», сравниваем метрики, оборачиваем лучшую модель в FastAPI, деплоим на AWS EC2.
📑 СодержаниеЦели проекта
Данные
Пайплайн работы
Результаты моделей
Структура репозитория
Быстрый старт
Использование API
Деплой на AWS
Лицензия
Цели проектаОбработать категориальные данные (one‑hot / label encoding).
Построить и сравнить две модели — Logistic Regression и Decision Tree.
Прокачать EDA, визуализацию, предобработку и ML‑навыки.
Реализовать REST‑сервис на FastAPI и задеплоить его на сервер.
ДанныеИсточникMushroom Dataset (Kaggle)Записей8 124Признаков22 (все категориальные)Целеваяclass → e (съедобный) / p (ядовитый)Полное описание признаков см. в конце файла.
Пайплайн работыflowchart TD
    A[EDA] --> B[Предобработка]
    B --> C1[Logistic Regression]
    B --> C2[Decision Tree]
    C1 & C2 --> D[Сравнение метрик]
    D --> E[Сохранение лучшей модели]
    E --> F[FastAPI сервис]
    F --> G[Деплой AWS EC2]Этап 1 — EDAОбзор df.head() / info()
Проверка пропусков, баланса классов
Частотные графики (sns.countplot, pd.crosstab)
Корреляции после временного one‑hot
Этап 2 — ПредобработкаЗамена ? в stalk-root на NaN, заполнение модой.
Кодирование:
One‑Hot (LogReg)
LabelEncoder (LogReg)
без кодирования (Decision Tree)
Train / Test = 80 / 20.
Этап 3 — МоделиПараметры по умолчаниюОсобенностиLogisticRegressionsolver='lbfgs', C подбираетсяТребует масштабирования + one‑hotDecisionTreeClassifiermax_depth — подборУмеет с категориями напрямуюМетрики: accuracy, precision, recall, F1, ROC‑AUC + confusion matrix.
Этап 4 — Сериализацияjoblib.dump(best_model, 'mushroom_model.pkl')
joblib.dump(scaler,       'mushroom_scaler.pkl')   # если использовалсяЭтап 5 — FastAPIPOST /predict/
При старте приложения подгружаются .pkl‑файлы.
Ответ:
{
  "poisonous": true,
  "probability": 0.96
}Этап 6 — ДеплойEC2 Ubuntu 22.04 (t2.micro).
Открыть порт 8000 в Security Group.
uvicorn main:mushroom_app --host 0.0.0.0 --port 8000.
Результаты моделейЗаполните после обучения.
МодельAccuracyPrecisionRecallF1ROC AUCLogistic Regression — — — — —Decision Tree — — — — —Структура репозитория├── data/                 # raw & processed csv
├── notebooks/            # Jupyter EDA & обучение
├── mushroom_predict/     # FastAPI приложение
│   ├── main.py           # эндпоинт /predict
│   └── ...
├── models/
│   ├── mushroom_model.pkl
│   └── mushroom_scaler.pkl
├── requirements.txt
└── README.md             # вы здесь!Быстрый старт# 1. Клонируем
$ git clone https://github.com/username/mushroom-predict.git && cd mushroom-predict

# 2. Создаём окружение
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 3. Запускаем API (модель лежит в models/)
$ uvicorn mushroom_predict.main:mushroom_app --reload
# → http://127.0.0.1:8001/docsИспользование APIПример запросаcurl -X POST http://127.0.0.1:8001/predict/ \
     -H "Content-Type: application/json" \
     -d '{
           "cap_shape": "x",
           "cap_surface": "s",
           "cap_color": "n",
           "bruises": "t",
           "odor": "p",
           ...
         }'Пример ответа{
  "poisonous": true,
  "probability": 0.96
}Деплой на AWS EC2Создать инстанс (t2.micro, Ubuntu 22.04).
sudo apt update && sudo apt install python3-pip -y.
Склонировать репозиторий или scp -r.
pip install -r requirements.txt.
nohup uvicorn mushroom_predict.main:mushroom_app --host 0.0.0.0 --port 8000 &.
Проверить curl http://<EC2-IP>:8000/docs.
ЛицензияПроект распространяется под лицензией MIT — см. файл LICENSE.
Автор: Алымбек Ибрагимов, 2025 г.
Описание признаков#ПризнакЗначенияРасшифровка1classe, pedible / poisonous2cap_shapeb, c, x, f, k, sbell, conical, convex, flat, knobbed, sunken3cap_surfacef, g, y, sfibrous, grooves, scaly, smooth4cap_colorn, b, c, g, r, p, u, e, w, ybrown, buff, …5bruisest, fbruises / no bruises6odora, l, c, y, f, m, n, p, salmond, anise, ……………(Полный список см. в ТЗ выше)
