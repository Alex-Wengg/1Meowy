import torch
import time

from TransformerSetups import GPTLanguageModel
from TransformerTrainers import get_batch, estimate_loss, save_checkpoint, load_checkpoint

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 4000
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
checkpoint_path = 'model_checkpoint.pth'  # Path for saving the checkpoint

with open("./CleanData/refined/input.txt", 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = 96
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s
                    ]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join(
    [itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_iter = load_checkpoint(model, optimizer)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

for iter in range(start_iter, max_iters):

  # every once in a while evaluate the loss on train and val sets
  if (iter % eval_interval == 0 or iter == max_iters - 1)  and iter != start_iter:
    losses = estimate_loss(model, eval_iters, train_data, block_size, val_data, batch_size)
    save_checkpoint(model, optimizer, iter)
    print(
        f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )
  if (iter % 500 == 0):

    print(decode(model.to(device).generate(context, max_new_tokens=100)[0].tolist()))


  # sample a batch of data
  xb, yb = get_batch('train', train_data, block_size, val_data, batch_size)

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  time.sleep(0.5)

# generate from the model
torch.save(model.state_dict(), '../finished.pth')
open('output.txt',  'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# def load_checkpoint(filename):
#   checkpoint = torch.load(filename)
#   model = GPTLanguageModel()  # Ensure this is the same model architecture as before
#   model.to(device)
#   optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Replace with your actual learning rate and optimizer
  
#   model.load_state_dict(checkpoint['model_state_dict'])
#   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#   iteration = checkpoint['iteration']
  
#   return model, optimizer, iteration

# checkpoint_path = 'model_checkpoint.pth'  # Adjust the path if necessary
# model, optimizer, start_iteration = load_checkpoint(checkpoint_path)

# # Decode the generated ids to text
# model.eval()  # Set the model to evaluation mode

# # Generate a sequence of token IDs with the model
# generated_ids = model.generate(context, max_new_tokens=500)

# # Decode the generated token IDs to a string
# generated_text = decode(generated_ids[0].tolist())

# # Print the generated text
# print(generated_text)

# # Optionally, write the generated text to a file
# with open('generated_text.txt', 'w', encoding='utf-8') as f:
#     f.write(generated_text)

# # If you want to generate more text (e.g., 10000 tokens), just change the max_new_tokens parameter
# # and write it to another file, if needed.
# generated_ids_large = model.generate(context, max_new_tokens=10000)
# generated_text_large = decode(generated_ids_large[0].tolist())
# with open('more.txt', 'w', encoding='utf-8') as f:
#     f.write(generated_text_large)