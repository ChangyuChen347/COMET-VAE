import numpy as np

train_context = np.load('data/empathetic-dialogue/sys_dialog_texts.train.npy', allow_pickle=True)
train_target = np.load('data/empathetic-dialogue/sys_target_texts.train.npy', allow_pickle=True)


dev_context = np.load('data/empathetic-dialogue/sys_dialog_texts.dev.npy', allow_pickle=True)
dev_target = np.load('data/empathetic-dialogue/sys_target_texts.dev.npy', allow_pickle=True)


test_context = np.load('data/empathetic-dialogue/sys_dialog_texts.test.npy', allow_pickle=True)
test_target = np.load('data/empathetic-dialogue/sys_target_texts.test.npy', allow_pickle=True)
#tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
all_c = []


def clean(sentence):
    word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}
    # sentence = sentence.lower()
    # sentence = sentence.split()
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    return sentence


for i, c in enumerate(train_context):
    #res = 'CLS '
    res = ''
    for j, s in enumerate(c):
        res += s
        res += ' <|endoftext|> '
    res += train_target[i] + ' <|endoftext|>'
    all_c.append(clean(res))

with open('train.em.txt', 'w') as f:
    f.write('\n'.join(all_c))
all_c = []
for i, c in enumerate(dev_context):
    res = ''
    for j, s in enumerate(c):
        res += s
        res += ' <|endoftext|> '
    res += dev_target[i] + ' <|endoftext|>'
    all_c.append(clean(res))
with open('dev.em.txt', 'w') as f:
    f.write('\n'.join(all_c))


all_c = []
for i, c in enumerate(test_context):
    res = ''
    for j, s in enumerate(c):
        res += s
        res += ' <|endoftext|> '
    #c += train_target[i] + ' <|endoftext|>'
    all_c.append(clean(res))
with open('test.em.txt', 'w') as f:
    f.write('\n'.join(all_c))

all_c = []


for i, c in enumerate(train_context):
    res = '_context_ '
    for j, s in enumerate(c[-1:]):
        res += s
        res += ' '
    #res = res[-128:]
    res += ' _response_ ' + train_target[i]
    all_c.append(clean(res))
with open('train.ctrl.txt', 'w') as f:
    f.write('\n'.join(all_c))
all_c = []
for i, c in enumerate(dev_context):
    res = '_context_ '
    for j, s in enumerate(c[-1:]):
        res += s
        res += ' '
    #res = res[-128:]

    res += ' _response_ ' + train_target[i]
    all_c.append(clean(res))
with open('dev.ctrl.txt', 'w') as f:
    f.write('\n'.join(all_c))