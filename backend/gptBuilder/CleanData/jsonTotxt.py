import json
from profanity_filter import ProfanityFilter

# Assuming 'data' is your JSON data
with open('input1.json', 'r') as file:
  data = json.load(file)
pf = ProfanityFilter()

with open('input.txt', 'a') as file:
  for entry in data:
    inner = entry['data']['datasetEntries']
    for item in inner:
      input_text = item['input']
      output_text = item['output']
      if pf.is_profane(input_text) or pf.is_profane(output_text):
        continue
      file.write(f"{input_text}\n {output_text}\n\n")
