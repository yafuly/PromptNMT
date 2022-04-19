import sys
inp = sys.argv[1]
out = sys.argv[2]

MARKER = [
'</AB>',
'</TI>',
'</TB>',
'</TE>',
'</RB>',
'</END>'
]

lines = [l.strip() for l in open(inp, 'r').readlines()]

advices = []
for line in lines[:]:
    tokens = line.split() + ['</END>']
    cache = []
    A = []
    T = []
    M = []
    for i,t in enumerate(tokens):
        if t in MARKER and len(cache) > 0:
            ad = " ".join(cache)
            if '</AB>' in cache:
                A.append(ad)
            elif '</TI>' in cache or '</TB>' in cache or '</TE>' in cache:
                T.append(ad)
            elif '</RB>' in cache:
                M.append(ad)
            else:
                raise ValueError
            cache = []
        cache.append(t)
    # print("\t".join(A))
    
    advices.append("\t".join(A)+"</SEP>"+"\t".join(T)+"</SEP>"+"\t".join(M))

with open(out, 'w') as f:
    f.write("\n".join(advices))