// Get chatbot elements
const chatbot = document.getElementById('chatbot');
const conversation = document.getElementById('conversation');
const inputForm = document.getElementById('input-form');
const inputField = document.getElementById('input-field');

// Add event listener to input form




inputForm.addEventListener('submit', function(event) {
  // Prevent form submission
  event.preventDefault();

  // Get user input
  const input = inputField.value;

  // Clear input field
  inputField.value = '';
  const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: "2-digit" });

  // Add user input to conversation
  let message = document.createElement('div');
  message.classList.add('chatbot-message', 'user-message');
  message.innerHTML = `<p class="chatbot-text" sentTime="${currentTime}">${input}</p>`;
  conversation.appendChild(message);





  // Generate chatbot response
  const response = generateResponse(input);

  // Add chatbot response to conversation
  message = document.createElement('div');
  message.classList.add('chatbot-message','chatbot');
  message.innerHTML = `<p class="chatbot-text" sentTime="${currentTime}">${response}</p>`;
  conversation.appendChild(message);
  message.scrollIntoView({behavior: "smooth"});
});



// Generate chatbot response function
function generateResponse(input) {
    // Add chatbot logic here
    const responses = [
      "Случайный ответ 1",
      "Второй случайный ответ",
      "Если сложить 1 и два то получится номер этого случайного ответа",
      "Супер секретный случайный ответ",
      
    ];

    // отправка на сервер
    fetch('http://localhost:5000/bot', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data: input }), 
    })
    .then(response => response.json())
    .then(data => {
      // Получение текста из переменной processed_data
      let answer = data.answer
      console.log("Answer:", answer); // Проверка данных в консоли
  })
  .catch(error => {
        console.error('Ошибка:', error);
      });
    
    // Return a random response
    return responses[answer]
    return responses[Math.floor(Math.random() * responses.length)];
  }
  