# Video detection ------------------------------------------------------------------|
import torch

# Run below in terminal to train model on specific dataset 
# python3 train.py --data ../data.yaml --weights yolov5s.pt --epochs 5   

vmodel = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', source='local')
"""img = "test2.png" 
results = vmodel(img)


#returns -> number xmin ymin xmax ymax confidence class name
print(results.pandas().xyxy[0].name)

results.show()"""



# Text inference ------------------------------------------------------------------|

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

dataset = load_dataset("Deeraj1234/Classify-Questions-Deeraj")
split_dataset = dataset["train"].train_test_split(test_size=.2)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


id2label = {0:"count",1:"locate_area",2:"segment",3:"detect",4:"describe",5:"classify"}
label2id = {"count":0,"locate_area":1,"segment":2,"detect":3,"describe":4,"classify":5}
def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example
split_dataset = split_dataset.map(encode_labels)


model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", 
                                                            num_labels=6,
                                                            id2label=id2label,
                                                            label2id=label2id)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_data = split_dataset.map(tokenize, batched=True)


args = TrainingArguments(output_dir="./results", eval_strategy="epoch", per_device_train_batch_size=8)

trainer = Trainer(model=model, args=args,
                  train_dataset=tokenized_data["train"],
                  eval_dataset=tokenized_data["test"])

trainer.train()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)






# Put together ------------------------------------------------------------------|


text = "how many cups are there?"
img = "test.jpg" 





inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=1)

#the type of question
question = id2label[pred.item()]

#whats in the image
results = vmodel(img)

import spacy

#to get the subject of the sentence
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
subjects = [token.lemma_.lower() for token in doc if token.dep_ in ("nsubj", "nsubjpass", "ROOT", "attr") and token.pos_ == "NOUN"]

image_labels = [label.lower() for label in results.pandas().xyxy[0]['name'].tolist()]

if(question == 'count'):
    final = 0
    for i in range(len(subjects)):
        for l in range(len(results.pandas().xyxy[0].name)):
            if(results.pandas().xyxy[0].name[l].lower() == subjects[i]):
                final = final + 1
    print(final)


elif question == 'classify':
    if len(results.pandas().xyxy[0]) > 0:
        label = results.pandas().xyxy[0].loc[0, 'name']
        print(f"The object is: {label}")
    else:
        print("Nothing here")


elif question == 'detect':
    found = set()
    for label in image_labels:
        if label in subjects:
            found.add(label)
    print(f"Detected: {', '.join(found) if found else 'None'}")
    


elif question == 'segment':
    results.show()

