### MIMICAUSE DATASET

huggingface_dataset_name = "pensieves/mimicause"

raw_dataset = load_dataset(huggingface_dataset_name, data_dir="/content/mimicause.zip")
#raw_dataset = load_dataset(huggingface_dataset_name)

#dataset = load_dataset(huggingface_dataset_name)
raw_dataset

def linearize_triplet(example):
  if str(example["Label"])=='0':
    example["linearized_triple"] = '(' + str(example['E1']) + ' # ' + 'Cause' + ' # ' + str(example['E2']) + ')'
  elif str(example["Label"])=='1':
    example["linearized_triple"] = '(' + str(example['E2']) + ' # ' + 'Cause' + ' # ' + str(example['E1']) + ')'
  elif str(example["Label"])=='2':
    example["linearized_triple"] = '(' + str(example['E1']) + ' # ' + 'Enable' + ' # ' + str(example['E2']) + ')'
  elif str(example["Label"])=='3':
    example["linearized_triple"] = '(' + str(example['E2']) + ' # ' + 'Enable' + ' # ' + str(example['E1']) + ')'
  elif str(example["Label"])=='4':
    example["linearized_triple"] = '(' + str(example['E1']) + ' # ' + 'Prevent' + ' # ' + str(example['E2']) + ')'
  elif str(example["Label"])=='5':
    example["linearized_triple"] = '(' + str(example['E2']) + ' # ' + 'Prevent' + ' # ' + str(example['E1']) + ')'
  elif str(example["Label"])=='6':
    example["linearized_triple"] = '(' + str(example['E1']) + ' # ' + 'Hinder' + ' # ' + str(example['E2']) + ')'
  elif str(example["Label"])=='7':
    example["linearized_triple"] = '(' + str(example['E2']) + ' # ' + 'Hinder' + ' # ' + str(example['E1']) + ')'
  elif str(example["Label"])=='8':
    example["linearized_triple"] = '(' + str(example['E1']) + ' # ' + 'Other' + ' # ' + str(example['E2']) + ')'
  return example

new_column =raw_dataset['train']["Label"]
new_column_string = map(str, new_column)
raw_dataset['train'] = raw_dataset['train'].add_column("linearized_triple", new_column_string)

new_column =raw_dataset['validation']["Label"]
new_column_string = map(str, new_column)
raw_dataset['validation'] = raw_dataset['validation'].add_column("linearized_triple", new_column_string)


new_column =raw_dataset['test']["Label"]
new_column_string = map(str, new_column)
raw_dataset['test'] = raw_dataset['test'].add_column("linearized_triple", new_column_string)

raw_dataset['train'] = raw_dataset['train'].map(linearize_triplet, batched=False)
raw_dataset['validation'] = raw_dataset['validation'].map(linearize_triplet, batched=False)
raw_dataset['test'] = raw_dataset['test'].map(linearize_triplet, batched=False)

exclude_idx = [1, 2, 256]

# create new dataset exluding those idx
raw_dataset['test'] = raw_dataset['test'].select(
    (
        i for i in range(len(raw_dataset['test']))
        if i not in set(exclude_idx)
    )
)

train_dataset = raw_dataset.get("train")
eval_dataset = raw_dataset.get("validation")
test_dataset = raw_dataset.get("test")

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
tokenized_test_dataset = test_dataset.map(generate_and_tokenize_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt_for_Llama3)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt_for_Llama3)
tokenized_test_dataset = test_dataset.map(generate_and_tokenize_prompt_for_Llama3)

print(tokenized_train_dataset[4]['input_ids'])
print(len(tokenized_train_dataset[4]['input_ids']))

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_train_dataset.shape}")
print(f"Validation: {tokenized_val_dataset.shape}")
print(f"Test: {tokenized_test_dataset.shape}")
