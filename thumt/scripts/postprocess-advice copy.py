import sys
inp = sys.argv[1]
out = sys.argv[2]

MARKER = [
'</AB>',
'</TI>',
'</TB>',
'</TE>',
'</RB>',
]

lines = [l.strip() for l in open(inp, 'r').readlines()]
A = []
T = []
M = []
advices = []
for line in lines:
    tokens = line.split()
    cache = []
    for t in tokens:
        if t in MARKER:
            if cache is not None:
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
    advices.append("\t".join(A)+"</SEP>"+"\t".join(T)+"</SEP>"+"\t".join(A))
        # acache = []
        # tcache = []
        # mcache = []
        # if t == '</AB>':
        #     if acache is not None:
        #         A.append(" ".join(acache))
        #         acache = []
        #     acache.append(t)
        # elif t in ['</TI>', '</TB>', '</TE>']:
        #     if tcache is not None:
        #         A.append(" ".join(tcache))
        #         tcache = []
        #     tcache.append(t)
        # elif t == '</RB>':
        #     if mcache is not None:
        #         A.append(" ".join(mcache))
        #         mcache = []
        #     mcache.append(t)
    with open(out, 'w') as f:
        f.write("\n".join(advices))