# Пользовательский ввод
user_input = input("Задайте свой вопрос: ")

# Предварительная обработка ввода
preprocessed_input = preprocess_text(user_input)

# Преобразование текста в числовую последовательность
input_sequence = text_to_sequence(preprocessed_input, word_index)

# Преобразование в тензор и добавление размерности батча
input_tensor = torch.tensor(input_sequence).unsqueeze(0)

# Генерация ответа с помощью модели
generated_response = generate_response(model, input_tensor, max_length, word_index, index_word)

# Вывод ответа
print("Ответ:", generated_response)