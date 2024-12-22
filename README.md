# github q&a rag

## Описание проекта

Это приложение, созданное на основе **Streamlit**, позволяет работать с GitHub-репозиториями. Основные возможности:
- Клонирование репозиториев с GitHub и выбор отдельных файлов для анализа.
- Генерация описания проекта и ответы на вопросы о репозитории с использованием Retrieval-Augmented Generation (RAG).

---

## Установка и запуск

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/PeMikj/streamlit-rag-app.git
cd streamlit-rag-app
```
### 2. Создайте и активируйте виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate
```
### 3. Установите зависимости
```bash
pip install -r requirements.txt
```
### 4. Запустите приложение
```bash
streamlit run app.py
```
