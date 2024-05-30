import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from answer_processing import get_answer
# Подключение к базе данных
conn = psycopg2.connect(
    dbname='postgres', 
    user='postgres.zzyahwklsrihlglqbsfd', 
    password='xy9$G/Cy~b~)&+_', 
    host='aws-0-eu-central-1.pooler.supabase.com',  
    port='5432'
)

# Функция для получения всех вопросов и их реальных ответов из базы данных
def get_questions_and_answers():
    cur = conn.cursor()
    query = """
    SELECT q.question, a.answer_processed
    FROM questions q
    JOIN answers a ON q.id_answer = a.id_answer;
    """
    cur.execute(query)
    rows = cur.fetchall()
    questions = [row[0] for row in rows]
    real_answers = [row[1] for row in rows]
    cur.close()
    return questions, real_answers

# Функция для оценки и визуализации
def evaluate_and_visualize():
    questions, real_answers = get_questions_and_answers()
    cosine_similarities = []
    correct_count = 0

    for i, question in enumerate(questions):
        generated_answer, similarity = get_answer(question)
        cosine_similarities.append(similarity)
        if generated_answer == real_answers[i]:  # Сравнение с реальным ответом
            correct_count += 1
    
    # Построение графика
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if generated_answer == real_answers[i] else 'red' for i, generated_answer in enumerate(questions)]
    ax.bar(range(len(cosine_similarities)), cosine_similarities, color=colors)
    ax.set_xlabel('Question Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity of Generated Answers to Real Answers')
    plt.show()

    print(f'Correct Answers: {correct_count}/{len(questions)}')

evaluate_and_visualize()
