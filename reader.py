import json
import os 
import tqdm
import pandas as pd

filepath = os.path.join('data', "train-v2.0.json")
with open(filepath) as f:
    data = json.load(f)

def clean_text(text):
    text = text.replace("]", " ] ")
    text = text.replace("[", " [ ")
    text = text.replace("\n", " ")
    text = text.replace("''", '" ').replace("``", '" ')

    return text


def make_dataset(data, mode = 'train'):
    contexts = []
    questions = []
    answers = []
    for article_id in tqdm.tqdm(range(len(data['data']))):
        list_paragraphs = data['data'][article_id]['paragraphs']
        for context_id, paragraph in enumerate(list_paragraphs):
            context =  paragraph['context']
            context = clean_text(context)
            contexts.append([article_id,context_id, paragraph['context']])
            qas = paragraph['qas']
            for qid, qa in enumerate(qas):
                question = qa['question']
                question = clean_text(question)
                questions.append([article_id, context_id, qid, question])

                for answer in qa['answers']:
                    answers.append([article_id, context_id, qid, clean_text(answer['text']), answer['answer_start']])
    
    context_df = pd.DataFrame(contexts,  columns = ['article_id','context_id', 'context'])
    question_df = pd.DataFrame(questions,  columns = ['article_id', 'context_id', 'question_id', 'question'])
    answer_df = pd.DataFrame(answers,  columns = ['article_id','context_id', 'question_id', 'answer', 'answer_start'])
    context_df.to_csv('data/train_contexts.csv', index = False)
    question_df.to_csv('data/train_questions.csv', index = False)
    answer_df.to_csv('data/train_answers.csv', index = False)
    print('Context:', len(context_df))
    print('Questions:', len(question_df))
    print('Answer:', len(answer_df))
    
make_dataset(data)