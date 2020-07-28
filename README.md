# Targeted-Aspect-Sentiment

run main.py

data is present in data/
create an experiments folder, which saves the weights based on --save-freq in main.py
saved weights in experiments/on_dev/weights


logs losses in log.txt 
After 1 epoch on dev dataset:
train: loss: 0.210370, aspect bce: 0.218309, sent bce: 0.202431

Tested on test dataset:
val: loss: 0.192302, aspect bce: 0.202852, sent bce: 0.181752