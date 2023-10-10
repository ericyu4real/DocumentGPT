// script.js

// JavaScript code to handle form submission and display chatbot response
const initializeForm = document.getElementById('initialize-form');
const questionForm = document.getElementById('question-form');
const chatHistory = document.getElementById('chat-history');

initializeForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(initializeForm);
    try {
        const response = await fetch('http://127.0.0.1:5002/api/bot_initialize', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        alert(data.data); // Display a success message (you can customize this)
    } catch (error) {
        console.error('Initialization error:', error);
        alert('Initialization error. Please try again.');
    }
});

questionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(questionForm);
    try {
        const response = await fetch('http://127.0.0.1:5002/api/query_bot', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        const chatItem = document.createElement('li');
        chatItem.textContent = `You: ${formData.get('question')}`;
        chatHistory.appendChild(chatItem);

        const chatbotItem = document.createElement('li');
        chatbotItem.textContent = `Chatbot: ${data.response}`;
        chatHistory.appendChild(chatbotItem);

        // Clear the input field
        questionForm.reset();
    } catch (error) {
        console.error('Question submission error:', error);
        alert('Question submission error. Please try again.');
    }
});
