


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained(r'E:\USTC\Code\Github\LGR\logic_process\pretrain')
model = BertModel.from_pretrained(r"E:\USTC\Code\Github\LGR\logic_process\pretrain")
Logic_predicate = "Replace me by Logic Predicates."
encoded_input = tokenizer(Logic_predicate, return_tensors='pt')
# print('encoded_input[0]=',encoded_input['input_ids'])
output = model(**encoded_input)

print('output[0].shape=',output[0].shape)
print('output[1].shape=',output[1].shape)


