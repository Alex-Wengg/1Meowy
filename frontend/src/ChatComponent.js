import React, { useState } from "react";
import axios from "axios";
import "./ChatComponent.css"; // Assuming you have separate CSS for the chat component

const ChatComponent = ({ aiEndpoint }) => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [isLoading, setIsLoading] = useState(false); // New state for loading

  const sendMessage = (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;

    setIsLoading(true); // Set loading to true
    setMessages([...messages, { sender: "user", text: userInput }]);

    axios.post(aiEndpoint, { context: userInput })
      .then(response => {
        const aiResponse = response.data.generated_text;
        setMessages(prevMessages => [...prevMessages, { sender: "ai", text: aiResponse }]);
        setUserInput(""); // Clear user input after receiving the response
      })
      .catch(error => {
        console.error('Error: ', error);
        setMessages(prevMessages => [...prevMessages, { sender: "ai", text: "Error in response." }]);
      })
      .finally(() => {
        setIsLoading(false); // Reset loading state
      });
  };

  return (
    <div className="chat-container">
      {messages.map((msg, index) => (
        <div key={index} className={`message ${msg.sender}`}>
          {msg.text}
        </div>
      ))}
      <form onSubmit={sendMessage}>
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Type a message..."
          disabled={isLoading} // Disable input during loading
        />
        <button type="submit" disabled={isLoading}>Send</button> {/* Disable button during loading */}
      </form>
    </div>
  );
}

export default ChatComponent;
