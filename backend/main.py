from flask import Flask, request, jsonify
import torch
from flask_cors import CORS, cross_origin

from gptBuilder.TransformerSetups import GPTLanguageModel

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

with open('./gptBuilder/CleanData/refined/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, '') for c in s
                    ]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos.get(i, '') for i in l])

vocab_size = 96
n_embd = 384
block_size = 256
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4
dropout = 0.2

def load_checkpoint(filename):
  checkpoint = torch.load(filename, map_location=device)
  model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)
  model.to(device)
  model.load_state_dict(checkpoint, strict=False)

  model.eval()

  context = torch.zeros((1, 1), dtype=torch.long, device=device)

  generated = model.generate(context, max_new_tokens=500) 
  generated_text = decode(generated[0].tolist())
  print("Generated Sample:", generated_text)
  checkpoint = torch.load(checkpoint_path, map_location='cpu')

  return model

checkpoint_path = './gptBuilder/finished.pth'
model = load_checkpoint(checkpoint_path)

@app.route('/generate', methods=['POST'])
@cross_origin()
def generate_text():
  try:
    data = request.json
    context = data.get('context')

    # Convert context to tensor and preprocess if necessary
    input_data = encode(context)
    input_data = torch.tensor([input_data], dtype=torch.long).to(device)

    # Generate text
    with torch.no_grad():
      generated = model.generate(input_data, max_new_tokens=500)
      generated_text = decode(generated[0].tolist())

    return jsonify({"generated_text": generated_text})
  except Exception as e:
    return jsonify({"error": str(e)}), 200


if __name__ == '__main__':
    app.run(debug=False, port=5000)
