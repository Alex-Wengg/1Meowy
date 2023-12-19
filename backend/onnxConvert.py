# import torch
# import torch.onnx
# from gptBuilder.TransformerSetups import GPTLanguageModel

# vocab_size = 96
# n_embd = 384
# block_size = 256
# n_head = 6
# n_layer = 6
# dropout = 0.2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# filename = './gptBuilder/finished.pth'
# checkpoint = torch.load(filename, map_location=device)

# model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)
# model.to(device)
# model.load_state_dict(checkpoint, strict=False)

# model.eval()

# context = torch.zeros((1, 1), dtype=torch.long, device=device)

# torch.onnx.export(model, context, 'model.onnx', export_params=True, opset_version=11,
#                   do_constant_folding=True, input_names=['input'], output_names=['output'],
#                   dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},)
from onnx_tf.backend import prepare

import onnx

# Load the ONNX file
model_onnx = onnx.load('model.onnx')

# Prepare the TensorFlow representation
tf_rep = prepare(model_onnx)

# Export the model to a TensorFlow-friendly format (e.g., SavedModel or HDF5)
tf_rep.export_graph('tf_form')
