# advices = [l.strip() for l in open("./test.advices.select-one").readlines()]
# hyps = [l.strip() for l in open("./test.de.out.model-81.pt").readlines()]

import re
import sys
ad_path = sys.argv[1]
hyp_path = sys.argv[2]
advices = [l.strip() for l in open(ad_path, 'r').readlines()]
hyps = [l.strip() for l in open(hyp_path, 'r').readlines()]
lyt_al = []
lyt_tf = []
lyt_wr = []
bad_cases = []
for ad, hyp in zip(advices, hyps):
    als,tfs,_ = ad.split("</SEP>")
    als = als.split("\t")
    tfs = tfs.split("\t")
    if len(als)>0 and als[0]!='':
        for al in als:
            al = al.split("</AM>")[1].strip()
            if al in hyp:
                lyt_al.append(1)
            else:
                lyt_al.append(0)
                bad_cases.append((hyp,ad))

    
    if len(tfs)>0 and tfs[0]!='':
        for tf in tfs:
            ori_tf = tf
            tf = " ".join(tf.split()[1:])

            tf_flag = False
            
            if "</TI>" in ori_tf:
                if tf in hyp:
                    tf_flag = True
            elif "</TB>" in ori_tf:
                if hyp.startswith(tf):
                    tf_flag = True      
            elif "</TE>" in ori_tf:
                if hyp.endswith(tf):
                    tf_flag = True
            else:
                print(ori_tf)
                raise ValueError

            if tf_flag:
                lyt_tf.append(1)
            else:
                bad_cases.append((hyp,ad))
                lyt_tf.append(0)


if len(lyt_al) > 0:
    print(sum(lyt_al)/len(lyt_al))
if len(lyt_tf) > 0:
    print(sum(lyt_tf)/len(lyt_tf))
print(sum(lyt_al+lyt_tf)/len(lyt_al+lyt_tf))
