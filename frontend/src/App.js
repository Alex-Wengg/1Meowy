import React from "react";

import ChatComponent from "./ChatComponent";

import "./App.css";

export default function App() {
  return (
    <div className="App">
      <h1>  Meowy the Cowboy Cat  </h1>
      <p>
      A Generatively Pretrained Transformer trained by <a href="https://github.com/Alex-Wengg" target="_blank" rel="noopener noreferrer">Alex Weng</a>
      </p>
      <p>Refresh Background Image For Another Pregenerated Image</p>
      <ChatComponent aiEndpoint="https://meowybackend2-ohjzsdclpa-uc.a.run.app/generate" />
    </div>
  );
}
