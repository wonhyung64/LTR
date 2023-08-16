#%%
import json


def get_corpus(root, task, corpus_id):
    (string1, string2, bundlenum, position) = corpus_id.split('_')
    assert string1 == 'msmarco' and string2 == task

    with open(f'{root}/{task}/msmarco_v2_{task}/msmarco_{task}_{bundlenum}', 'rt', encoding='utf8') as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        corpus = json.loads(json_string)
        # assert corpus['docid'] == corpus_id

        return corpus


#%%
task = "doc"
root = f"/Volumes/LaCie/data/trec"
corpus_id = f'msmarco_doc_01_1987545310'


document = get_corpus(root, task, corpus_id)
print(document.keys())
document["url"]
document["title"]
document["headings"]
document["body"]
document["docid"]


# %%
task = "passage"
root = f"/Volumes/LaCie/data/trec"
corpus_id = f'msmarco_passage_02_0'


corpus = get_corpus(root, task, corpus_id)
print(corpus.keys())
corpus["pid"]
corpus["passage"]
corpus["spans"]
corpus["docid"]

{
    "pid": 'msmarco_passage_02_0',
    "passage": "Utility - A champion's ability to grant beneficial effects to their allies or to provide vision. Note that the client rates champions on a scale of 1-3, with champions that feature both None and Low in a particular strength being marked equally.",
    "spans": '(1712,1808),(1809,1957)',
    "docid": ,
}