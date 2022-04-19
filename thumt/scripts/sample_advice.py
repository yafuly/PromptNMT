import sys
sys.path.append('/home/amax/Codes/nmt-inter-state/THUMT/thumt')
from utils.random import RandomGenerator
ad_path = sys.argv[1]
sample_path = sys.argv[2]
def sample_advice_fixn(
    advice, 
    random_generator=None, 
    epoch=-1,
    no_sample=False,
    ):
    if no_sample or random_generator is None: # sample only in train mode
        advice = [" ".join(advice[0]).split()]
        return advice   

    A, T, M = advice
    A = A.split(b'\t')
    T = T.split(b'\t')
    M = M.split(b'\t')

    ####################
    # 3 types of advices
    # flip a coin a decide whether take it
    # for A advice, sample a ratio of them
    # for T advice, sample only on of them
    # for M advice, keep it
    def _sample(l, ratio, max_len):
        # shuf
        l = random_generator.shuffle(l)
        l = l[:max_len] if len(l)>max_len else l

        return l

    ratio = 1
    max_len = [5,2,3]
    A = _sample(A, ratio=ratio, max_len=max_len[0])
    T = _sample(T, ratio=ratio, max_len=max_len[1])
    M = _sample(M, ratio=ratio, max_len=max_len[2])
    advice = b" ".join(A + T + M).split()

    return advice


rng = RandomGenerator(1)
ad_lines = [l.strip() for l in open(ad_path, 'rb').readlines()]
sample_advice = []
SEP=b'</SEP>'
for line in ad_lines:
    sample = sample_advice_fixn(line.split(SEP), rng)
    sample = b' '.join(sample)
    sample_advice.append(sample)

with open(sample_path, 'wb') as f:
    f.write(b'\n'.join(sample_advice))
