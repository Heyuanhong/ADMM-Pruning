import os

def Read_scores(file):
    score = {}
    with open(file,"r",encoding="utf-8")as r:
        for l in r:
            k,v = l.strip().split("=")
            score[k] = v
    return score

path = "./milnews_output"
file_dirs = os.listdir(path)
results = "./results.txt"

with open(results,"w",encoding="utf-8")as w:
    w.write("acc\tacc_and_f1\tf1\n")
    for d in file_dirs:
        f = os.path.join(path,d,"checkpoint_eval_results.txt")
        score = Read_scores(f)
        # print(score)
        for k in score:
            print(score[k])
            w.write(score[k].strip()+'\t')
        w.write('\n')