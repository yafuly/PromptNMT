# advices = [l.strip() for l in open("./test.advices.select-one").readlines()]
# hyps = [l.strip() for l in open("./test.de.out.model-81.pt").readlines()]


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
    al,tf,wr = ad.split("</SEP>")
#     print(al)
#     print(tf)
    if not al or not tf:
        continue
    al = al.split("</AM>")[1].strip()
    tf = " ".join(tf.split()[1:])
    if al in hyp:
        lyt_al.append(1)
    else:
        lyt_al.append(0)
        bad_cases.append((hyp,ad))
    if tf in hyp:
        lyt_tf.append(1)
    else:
        lyt_tf.append(0)
        bad_cases.append((hyp,ad))

print(sum(lyt_al)/len(lyt_al))
print(sum(lyt_tf)/len(lyt_tf))