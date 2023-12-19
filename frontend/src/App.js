import React from "react";
import ReactDOM from "react-dom/client";
import ChatComponent from "./ChatComponent";
import "./App.css";

export default function App() {
  return (
    <div className="App">
      <ChatComponent aiEndpoint="http://127.0.0.1:5000/generate" />
    </div>
  );
}
