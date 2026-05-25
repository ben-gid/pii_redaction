# Pii Redaction
**incomplete; project in progress**
## description
A production-ready API that detects and redacts personally identifiable information from text before it enters an LLM pipeline.
**Stack:** DeBERTa-v3 · HuggingFace Trainer · FastAPI · Docker · AWS ECS Fargate · HuggingFace Spaces

## uncleaned project process and stumbles
I thought i would start with tokenizing and training my dataset for distilbert to make sure i had fulll pipeline that worked. i thought distilbert would be the optimal model as its light, and classic for nlp classification. after tokenizing the dataset with the bert tokenizer i trained distilbert on it to make sure it worked well. it did. then i moved on to using the same functions i used to tokenize and label my dataset for distilbert for deberta-v3. it didn't work. why? because it turns out the dataset was masked with indices starting and ending with the first and last characters of it's entity respectively. so in `get_ner_tags` i would use:
```python
if offset[0] >= privacy_mask["start"] and offset[1] <= privacy_mask["end"]:
    label = privacy_mask["label"]
    if offset[0] == privacy_mask["start"]:
        label = "B-" + label
    else:
        label = "I-" + label
```
this worked well for the distilbert tokenizer since the offsets returned from the tokenizer matched the way the authors masked the dataset. However, deberta uses a different type of tokenizer [TODO: add type tokenizer], which tokenizes words by their starting white space. so `get_ner_tags` from above wouldn't register the first token of the entity if it started with a whitespace. for example, the time entity from the first training example was tagged like this:
| Row            | Token 1      | Token 2      | Token 3      | Token 4      |
|----------------|--------------|--------------|--------------|--------------|
| **tokens**     | `▁10`        | `:`          | `20`         | `am`         |
| **labels**     | `O`          | `I-TIME`     | `I-TIME`     | `I-TIME`     |
| **word_ids**   | `49`         | `49`         | `49`         | `49`         |
| **token_offsets** | `(310, 313)` | `(313, 314)` | `(314, 316)` | `(316, 318)` |
the "_10" token wasn't labeled with "B_TIME" since its offset started at "310" and the masks offset the authors of the dataset set starts at 311. 
```json
{'value': '10:20am', 'start': 311, 'end': 318, 'label': 'TIME'}
```
so i had to change my `get_ner_labels` to instead tag differently:
```python
# if statement is switched to check if any character of the token falls within the mask
if offset[1] > privacy_mask["start"] and offset[0] < privacy_mask["end"]:
    label = privacy_mask["label"]
    # switch if statement to also include less than
    if offset[0] <= privacy_mask["start"]:
        label = "B-" + label
    else:
        label = "I-" + label
```