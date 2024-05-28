import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import nltk
from nltk.tokenize import word_tokenize
import psycopg2
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
# Загрузка RuBERT
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")

nltk.download('punkt')

# Функция для построения словаря
def build_vocab(texts):
    word_index = {'<PAD>': 0, '<UNK>': 1}
    current_index = len(word_index)
    
    for text in texts:
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if token not in word_index:
                word_index[token] = current_index
                current_index += 1
                
    return word_index

# Функция предварительной обработки текста
def preprocess_text(text):
    return text.lower()

# Подключение к базе данных и извлечение данных
conn = psycopg2.connect(
    dbname='postgres',  # имя базы данных
    user='postgres.zzyahwklsrihlglqbsfd',  # имя пользователя
    password='xy9$G/Cy~b~)&+_',  # пароль
    host='aws-0-eu-central-1.pooler.supabase.com',  # хост
    port='5432'  # порт
)

# Извлечение данных из базы
cur = conn.cursor()
cur.execute("SELECT name_of_block, question, answer FROM processed_data")
rows = cur.fetchall()
cur.close()

# Разделение данных на обучающую и тестовую выборки (70/30)
train_size = int(0.7 * len(rows))
train_data = rows[:train_size]
test_data = rows[train_size:]

# Функция для создания батчей данных
def create_batch(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Извлечение вопросов и ответов из обучающей и тестовой выборок
train_questions = [row[1] for row in train_data]
train_answers = [row[2] for row in train_data]
test_questions = [row[1] for row in test_data]
test_answers = [row[2] for row in test_data]


# Предварительная обработка текста вопросов и ответов
preprocessed_train_questions = [preprocess_text(question) for question in train_questions]
preprocessed_train_answers = [preprocess_text(answer) for answer in train_answers]
preprocessed_test_questions = [preprocess_text(question) for question in test_questions]
preprocessed_test_answers = [preprocess_text(answer) for answer in test_answers]

# Построение словаря на основе всех текстов (вопросы + ответы)
all_texts = preprocessed_train_questions + preprocessed_train_answers
word_index = build_vocab(all_texts)

# Функция для преобразования текста в числовые последовательности
def text_to_sequence(text, word_index):
    tokens = word_tokenize(text.lower())
    sequence = [word_index.get(token, word_index['<UNK>']) for token in tokens]
    return sequence
# Преобразование текстов в числовые последовательности
train_questions_seq = [text_to_sequence(text, word_index) for text in preprocessed_train_questions]
train_answers_seq = [text_to_sequence(text, word_index) for text in preprocessed_train_answers]
test_questions_seq = [text_to_sequence(text, word_index) for text in preprocessed_test_questions]
test_answers_seq = [text_to_sequence(text, word_index) for text in preprocessed_test_answers]

# Преобразование данных в тензоры
train_questions_tensor = [torch.tensor(seq) for seq in train_questions_seq]
train_answers_tensor = [torch.tensor(seq) for seq in train_answers_seq]
test_questions_tensor = [torch.tensor(seq) for seq in test_questions_seq]
test_answers_tensor = [torch.tensor(seq) for seq in test_answers_seq]


# Создание маски внимания для тензоров с паддингом
def create_attention_mask(sequences, padding_token):
    attention_masks = []
    for seq in sequences:
        seq_length = len(seq)
        mask = [1] * seq_length + [0] * (max_seq_length - seq_length)
        attention_masks.append(mask)
    return torch.tensor(attention_masks)

# Определение максимальной длины последовательности
max_seq_length = max(len(seq) for seq in all_texts)

# Создание маски внимания для обучающего и тестового наборов
padding_token = word_index['<PAD>']
train_attention_mask = create_attention_mask(train_questions_seq, padding_token)
test_attention_mask = create_attention_mask(test_questions_seq, padding_token)


# Построение словаря на основе всех текстов (вопросы + ответы)
all_texts = preprocessed_train_questions + preprocessed_train_answers
word_index = build_vocab(all_texts)

# Определение модели Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, bert_model, hidden_size, output_size, num_layers=2):
        super(Seq2Seq, self).__init__()
        self.bert = bert_model
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, src, trg, attention_mask = None):
        trg_len = trg.shape[1]
        batch_size = trg.shape[0]
        trg_vocab_size = self.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        src_embedded = self.bert(src)[0][:, 0, :]  # Получаем эмбеддинги только для первого токена [CLS]
        
        hidden = src_embedded.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        
        trg_embedded = self.embedding(trg)
        
        for t in range(trg_len):
            output, (hidden, cell) = self.decoder(trg_embedded[:, t, :].unsqueeze(1), (hidden, cell))
            output = self.fc_out(output.squeeze(1))  # Применяем линейный слой
            outputs[:, t, :] = output
        
        return outputs

# Создание модели Seq2Seq с RuBERT
model = Seq2Seq(bert, hidden_size=bert.config.hidden_size, output_size=len(word_index))

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл дообучения модели
num_epochs = 1
batch_size = 128

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for q_batch, a_batch, mask_batch in zip(create_batch(train_questions_tensor, batch_size),
                                            create_batch(train_answers_tensor, batch_size),
                                            create_batch(train_attention_mask, batch_size)):
        q_batch = rnn_utils.pad_sequence(q_batch, batch_first=True, padding_value=word_index['<PAD>'])
        a_batch = rnn_utils.pad_sequence(a_batch, batch_first=True, padding_value=word_index['<PAD>'])
        
        optimizer.zero_grad()
        output = model(q_batch, a_batch[:, :-1], attention_mask=torch.tensor(mask_batch).to(q_batch.device))
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        a_batch = a_batch[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, a_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Эпоха {epoch+1}/{num_epochs}, Потери: {epoch_loss/len(train_questions_tensor)}")

# Функция для оценки модели на тестовых данных
def evaluate(model, test_questions, test_answers, test_attention_mask):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for q_batch, a_batch, mask_batch in zip(create_batch(test_questions, batch_size),
                                                create_batch(test_answers, batch_size),
                                                create_batch(test_attention_mask, batch_size)):
            q_batch = rnn_utils.pad_sequence(q_batch, batch_first=True, padding_value=word_index['<PAD>'])
            a_batch = rnn_utils.pad_sequence(a_batch, batch_first=True, padding_value=word_index['<PAD>'])
            mask_batch = rnn_utils.pad_sequence(mask_batch, batch_first=True, padding_value=0)

            output = model(q_batch, a_batch[:, :-1], attention_mask=mask_batch[:, :-1].to(q_batch.device))
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            a_batch = a_batch[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, a_batch)
            epoch_loss += loss.item()

# Оценка модели на тестовой выборке
test_loss = evaluate(model, test_questions_tensor, test_answers_tensor, test_attention_mask)
print(f"Test Loss : {test_loss}")

# Функция для генерации ответа модели на вопрос
def generate_answer(question, model, max_length=50):
    model.eval()
    question_seq = text_to_sequence(preprocess_text(question), word_index)
    question_tensor = torch.tensor(question_seq).unsqueeze(0)
    attention_mask = torch.tensor([1 if token != word_index['<PAD>'] else 0 for token in question_seq]).unsqueeze(0)
    
    trg = torch.zeros(1, max_length, dtype=torch.long)  # Пустой тензор trg с размерностью [1, max_length]
    
    with torch.no_grad():
        output = model(question_tensor, trg, attention_mask=attention_mask)
    
    generated_answer = []
    for idx in output[0]:
        token_index = torch.argmax(idx).item()  # Находим индекс максимального значения
        token = list(word_index.keys())[list(word_index.values()).index(token_index)]  # Получаем токен из словаря по индексу
        if token == '<END>' or len(generated_answer) >= max_length:
            break
        generated_answer.append(token)
    
    generated_answer = ' '.join(generated_answer)
    return generated_answer





# Пример использования функции для генерации ответа
example_question = "какой метод оценка знание мочь наиболее подходящий"
generated_answer = generate_answer(example_question, model)
print(f"Сгенерированный ответ: {generated_answer}")

