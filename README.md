# Targeted-Aspect-Sentiment

run main.py

data is present in data/<br/>
create an experiments folder, which saves the weights based on --save-freq in main.py<br/>
saved weights in experiments/on_dev/weights<br/>


logs losses in log.txt <br/>
After 1 epoch on dev dataset:<br/>
train: loss: 0.210370, aspect bce: 0.218309, sent bce: 0.202431<br/>

Tested on test dataset:<br/>
val: loss: 0.192302, aspect bce: 0.202852, sent bce: 0.181752