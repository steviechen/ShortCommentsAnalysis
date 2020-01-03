# rt_index = curr_index
# 
# SentimentDocument = namedtuple('SentimentDocument', 'words tags')
# 
# 
# class Doc_list(object):
#     def __init__(self, f):
#         f = f
# 
#     def __iter__(self):
#         for i, line in enumerate(codecs.open(f, encoding='utf8')):
#             words = line.strip().split(' ')
#             tags = [int(words[0][2:])]
#             words = words[1:]
#             yield SentimentDocument(words, tags)


# temp_doc = doc_cause[doc_cause["cause"] == cause]["doc"]

splited_test = pd.Series(documents.apply(conc_list)[rt_index])

doc_f = codecs.open('/Users/steviechen1982/Documents/bems/BEMS/data/corpus/doc_for_d2v_12w_test.txt', 'w',
                    encoding='utf8')
for i, contents in enumerate(splited_test):
    words = []
    for word in contents.split(' '):
        words.append(word)
    tags = [i]
    if i % 10000 == 0:
        log('iter = %d' % i)
    doc_f.write(u'_*{} {}\n'.format(i, ' '.join(words)))
doc_f.close()
log('Job Done.')

############################ 用新的tracking_id的数据来更新 d2v模型 ############################
d2v_test = Doc2Vec.load('/Users/steviechen1982/Documents/bems/BEMS/data/models/dbow_d2v_12w.model')
doc_list_test = Doc_list('/Users/steviechen1982/Documents/bems/BEMS/data/corpus/doc_for_d2v_12w_test.txt')
d2v_test.build_vocab(doc_list_test, update=True)
# d2v.train(texts, total_examples=model.corpus_count, epochs=model.iter)
for i in range(10):
    log('pass: ' + str(i))
    doc_list_test = Doc_list(
        '/Users/steviechen1982/Documents/bems/BEMS/data/corpus/doc_for_d2v_12w_test.txt')
    d2v_test.train(doc_list_test, total_examples=d2v_test.corpus_count, epochs=d2v_test.iter)
#     X_d2v = np.array([d2v.docvecs[i] for i in range(splited_test.size)])

x_sp_test = np.array([d2v_test.docvecs[i] for i in range(splited_test.size)])

############################ 加载数据和模型更新 w2v ############################
documents = splited_test
texts = [[word for word in document.split(' ')] for document in documents]
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
# texts = [[token for token in text if frequency[token] >= 2] for text in texts]

model_test = Word2Vec.load('/Users/steviechen1982/Documents/bems/BEMS/data/models/w2v_12w.model')

model_test.build_vocab(texts, update=True)
model_test.train(texts, total_examples=model_test.corpus_count, epochs=model_test.iter)
############################ 由w2v模型来计算向量 ############################
log('Start get w2v feat..')
w2v_feat = np.zeros((len(texts), 300))
w2v_feat_avg = np.zeros((len(texts), 300))
i = 0
for line in texts:
    num = 0
    for word in line:
        num += 1
        vec = model_test[word]
        w2v_feat[i, :] += vec
    w2v_feat_avg[i, :] = w2v_feat[i, :] / num
    i += 1
    if i % 1000 == 0:
        log(i)

df_w2v_test = pd.DataFrame(w2v_feat)
df_w2v_test.columns = ['w2v_' + str(i) for i in df_w2v_test.columns]
df_w2v_test.to_csv('/Users/steviechen1982/Documents/bems/BEMS/data/feature/w2v/w2v_12w_test.csv',
                   encoding='utf8', index=None)
df_w2v_avg_test = pd.DataFrame(w2v_feat_avg)
df_w2v_avg_test.columns = ['w2v_avg_' + str(i) for i in df_w2v_avg_test.columns]
df_w2v_avg_test.to_csv('/Users/steviechen1982/Documents/bems/BEMS/data/feature/w2v/w2v_avg_12w_test.csv',
                       encoding='utf8', index=None)

log('Save w2v and w2v_avg feat done!')

EMBEDDING_FILE = '/Users/steviechen1982/Documents/bems/BEMS/data/models/crawl-300d-2M.vec'


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
embed_size = 300

w2v_feat_global = np.zeros((len(texts), 300))
w2v_feat_avg_global = np.zeros((len(texts), 300))
i = 0
for line in texts:
    num = 0
    for word in line:
        num += 1
        vec = embeddings_index.get(word)
        if str(vec) == "None":
            continue
        w2v_feat_global[i, :] += vec
    w2v_feat_avg_global[i, :] = w2v_feat_global[i, :] / num
    i += 1

df_w2v_test_global = pd.DataFrame(w2v_feat_global)
df_w2v_test_global.columns = ['w2v_global_' + str(i) for i in df_w2v_test_global.columns]
df_w2v_test_global.to_csv('/Users/steviechen1982/Documents/bems/BEMS/data/feature/w2v/w2v_12w_test_global.csv',
                          encoding='utf8', index=None)
df_w2v_avg_test_global = pd.DataFrame(w2v_feat_avg_global)
df_w2v_avg_test_global.columns = ['w2v_avg_global_' + str(i) for i in df_w2v_avg_test_global.columns]
df_w2v_avg_test_global.to_csv('/Users/steviechen1982/Documents/bems/BEMS/data/feature/w2v/w2v_avg_12w_test_global.csv',
                              encoding='utf8', index=None)

log('Save w2v_global and w2v_avg_global feat done!')

############################ 合并所有向量用于计算距离 ############################
df_test = pd.concat([pd.DataFrame(x_sp_test), df_w2v_test, df_w2v_avg_test, df_w2v_avg_test_global], axis=1)
df_test.set_index(splited_test.index)

dist = []
for i in range(0, df_result.shape[0]):
    dist.append([i, cosine_distances(np.array([df_test.iloc[0], df_result.iloc[i]]))[0][1]])
result_temp = pd.DataFrame(dist, columns=["index", "distance"])

origin = allDFComponents.iloc[rt_index: rt_index + 1]
test = allDFComponents.iloc[
    result_temp.sort_index(by=['distance'], ascending=True).head(6).reset_index().iloc[1:6]["index"].values]
test["distance"] = pd.DataFrame(
    result_temp.sort_index(by=['distance'], ascending=True).head(6).reset_index().iloc[1:6]["distance"]).set_index(
    test.index)