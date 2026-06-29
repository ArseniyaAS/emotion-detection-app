# Emotion Detection App 🎭

Веб-приложение для распознавания эмоций в англоязычном тексте на базе DistilBERT. Проект включает обучение и сравнение моделей, оценку качества и интерфейс для локального запуска через Streamlit.

Пользователь вводит текст, а модель на основе DistilBERT определяет одну из 6 эмоций: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.

![Status](https://img.shields.io/badge/status-prototype--ready-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![NLP](https://img.shields.io/badge/NLP-DistilBERT-orange)

## Содержание

- [О проекте](#о-проекте)
- [Стек](#стек)
- [Структура проекта](#структура-проекта)
- [Подход](#подход)
- [Модель](#модель)
- [Тестирование](#тестирование)
- [Текущий статус](#текущий-статус)
- [Начало работы](#начало-работы)
- [Что можно улучшить](#что-можно-улучшить)
- [Автор](#автор)
- [Источники](#источники)

## О проекте

Цель проекта — построить модель классификации эмоций в тексте, сравнить baseline-подход и transformer-модель, а затем упаковать решение в прототип веб-приложения.

## Стек

- Python
- Streamlit
- Hugging Face Transformers
- DistilBERT
- Scikit-learn
- Pandas
- Matplotlib / Seaborn
- PyTorch

## Структура проекта

```text
emotion-detection-app/
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── emotion_classification_experiments.ipynb
├── screenshots/
│   ├── app-main-screen.png
│   ├── prediction-example.png
│   └── model-metrics.png
```

## Подход

Были протестированы два подхода:

1. Baseline-модель: TF-IDF + Logistic Regression.
2. Основная модель: fine-tuned DistilBERT для многоклассовой классификации эмоций.

После оценки качества модель была подключена к Streamlit-приложению для локального использования.

## Модель

Приложение ожидает наличие локально сохранённой модели в папке `emotion_model_final`.

Если папка отсутствует, приложение не запустится до добавления модели в проект или изменения пути загрузки.

## Тестирование

| Модель | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 86.45% | 0.80 |
| DistilBERT (fine-tuned) | 93–94% | ~0.93 |

## Текущий статус

Проект оформлен как локально запускаемый прототип. Публикация демо-версии и контейнеризация могут быть добавлены на следующем этапе.

## Начало работы

1. Клонировать репозиторий

```bash
git clone <repo-link>
cd emotion-detection-app
```

2. Установить зависимости

```bash
pip install -r requirements.txt
```

3. Запустить приложение

```bash
streamlit run app.py
```

После запуска приложение будет доступно по адресу:
`http://localhost:8501`

## Что можно улучшить

- Поддержка русского языка
- Вывод top-3 вероятностей по эмоциям
- Улучшение интерфейса Streamlit
- Добавление сохранения истории запросов
- Публикация демо-версии

## Автор

Синявская А.А.

## Источники

- Датасет: `dair-ai/emotion`
- Базовая модель: `distilbert-base-uncased`

 