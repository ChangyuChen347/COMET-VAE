import pickle as pkl
import numpy as np
import torch
from src.empathy_classifier import EmpathyClassifier
import nltk
from nltk.translate.meteor_score import meteor_score
from utils.metric import moses_multi_bleu
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
'''
Example:
'''
ER_model_path = 'output/emo_sample.pth'
IP_model_path = 'output/int_sample.pth'
EX_model_path = 'output/exp_sample.pth'
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")

from src.metric import bleu_corpus

def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len


def get_dist_(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for r in res:
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len


empathy_classifier = EmpathyClassifier(device,
                                           ER_model_path=ER_model_path,
                                           IP_model_path=IP_model_path,
                                           EX_model_path=EX_model_path, )
def count_new_metrics(res_pkl, dataset='daily'):
    if dataset != 'daily':
        test_context = np.load('data/empathetic-dialogue/sys_dialog_texts.test.npy', allow_pickle=True)
    else:
        test_context = pkl.load(open('data/daily/test_context.pkl', 'rb'))
        test_idx = pkl.load(open('data/daily/test_idx.pkl', 'rb'))
        test_context = [test_context[i] for i in test_idx]
    file = pkl.load(open(res_pkl, 'rb'))
    rf = file['ref']
    all_c = []
    for i, c in enumerate(test_context):
        res = ''
        for i, s in enumerate(c[-3:]):
            res += s
            res += ' '
        all_c.append(clean(res))
    seeker_posts = all_c
    response_posts = file['hyp_g']
    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist_([nltk.word_tokenize(clean(r)) for r in response_posts])
    print("mi_dist1", mi_dist1)
    print("mi_dist2", mi_dist2)
    print("avg_len", avg_len)
    print("& %.4f & %.4f & %.2f" \
          % (mi_dist1 * 100, mi_dist2 * 100, avg_len))

    from src.metric import rouge
    bleus = []
    rouges = []
    bleus2 = []
    bleus3 = []
    bleus4 = []
    meteors = []
    for i in range(len(seeker_posts)):
        b1,b2,b3,b4 = bleu_corpus([response_posts[i]], [rf[i]])
        bleus.append(b1)
        bleus2.append(b2)
        bleus3.append(b3)
        bleus4.append(b4)
        rouges.append(rouge(nltk.word_tokenize(response_posts[i]), nltk.word_tokenize(rf[i])))
        #meteors.append(meteor_score([rf[i]], response_posts[i], 4))
    mose_bleu = moses_multi_bleu(np.array(response_posts), np.array(rf), lowercase=True)
    #print('meteor', np.mean(meteors))
    print('mose_bleu', mose_bleu)
    print('bleu1', np.mean(bleus))
    print('bleu2', np.mean(bleus2))
    print('bleu3', np.mean(bleus3))
    print('bleu4', np.mean(bleus4))
    print(np.mean(bleus + bleus2 + bleus3 + bleus4) / 4)
    print('rouge', np.mean(rouges))
    er = .0
    ip = .0
    ex = .0
    count = 0

    # hyp = ['i am sure he will be fine', 'oh , that is so exciting ! i hope you you can find a new job.',
    #        'that is good news . i hope he can find a job.', 'that is a great job . i hope he can find a job.',
    #        'maybe he can get another job. do you know what you will do.', 'oh that is a bummer . i hope he gets a job .']
    # seeker_posts = ['my husband lost a job but i am hoping he can find a job soon.' for _ in range(len(hyp))] + seeker_posts
    # response_posts = hyp + response_posts
    for i in range(len(seeker_posts)):
        try:
            (logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX,
             predictions_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])
        except Exception:
            continue
        count += 1
        er += predictions_ER[0]
        ip += predictions_IP[0]
        ex += predictions_EX[0]
        #print(predictions_ER[0], predictions_IP[0], predictions_EX[0])
        #pbar.set_description("er:{:.3f} ip:{:.3f} ex:{:.3f}".format(er / count, ip / count, ex / count))
    print(count)
    print('er', er / (count))
    print('ip', ip / (count))
    print('ex', ex / (count))
    print('avg', (er+ip+ex)/(count))
    print('avg', (er + ip + ex) / (count) / 3)
    return er / count, ip / count, ex / count, mose_bleu, np.mean(rouges), mi_dist1, mi_dist2, avg_len
#output_file.close()

if __name__ == "__main__":
    count_new_metrics('ex5129999.res.pkl', 'daily')

# no = 0

# for response_posts in x[:]:
#     print(no)
#     count = 0
#     er = .0
#     ip = .0
#     ex = .0
#     no += 1
#     for i in range(len(seeker_posts)):
#         try:
#             (logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX,
#              predictions_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])
#         except Exception:
#             continue
#         count += 1
#         er += predictions_ER[0]
#         ip += predictions_IP[0]
#         ex += predictions_EX[0]
#         # print(predictions_ER[0], predictions_IP[0], predictions_EX[0])
#         # pbar.set_description("er:{:.3f} ip:{:.3f} ex:{:.3f}".format(er / count, ip / count, ex / count))
#     print(count)
#     print('er', er / (count))
#     print('ip', ip / (count))
#     print('ex', ex / (count))
#     print('avg', (er + ip + ex) / (count))
#     print('avg', (er + ip + ex) / (count) / 3)