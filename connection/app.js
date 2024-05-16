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
  const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Add user input to conversation
  let message = document.createElement('div');
  message.classList.add('chatbot-message', 'user-message');
  message.innerHTML = `<p class="chatbot-text" sentTime="${currentTime}">${input}</p>`;
  conversation.appendChild(message);

  // Generate chatbot response
  generateResponse(input).then(response => {
    // Add chatbot response to conversation
    const botMessage = document.createElement('div');
    botMessage.classList.add('chatbot-message', 'chatbot');
    botMessage.innerHTML = `<p class="chatbot-text" sentTime="${currentTime}">${response}</p>`;
    conversation.appendChild(botMessage);
    botMessage.scrollIntoView({ behavior: 'smooth' });
  });
});

// Generate chatbot response function
function generateResponse(input) {
  return fetch('http://localhost:5000/bot', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ data: input }),
  })
    .then(response => response.json())
    .then(data => {
      // Получение текста из переменной processed_data
      let answer = data.answer;
      console.log("Answer:", answer); // Проверка данных в консоли
      return answer;
    })
    .catch(error => {
      console.error('Ошибка:', error);
      return 'Произошла ошибка. Попробуйте еще раз.';
    });
}