import abc
from sys import flags
from thumt.utils.random import RandomGenerator

from typing import List, NoReturn


class Tokenizer(object):

    def __init__(self, name: str):
        self._name = name

    @abc.abstractmethod
    def __repr__(self) -> NoReturn:
        raise NotImplementedError("Tokenizer.__repr__ not implemented.")

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def encode(self, inp: bytes) -> NoReturn:
        raise NotImplementedError("Tokenizer.encode not implemented.")

    @abc.abstractmethod
    def decode(self, inp: List[bytes]) -> NoReturn:
        raise NotImplementedError("Tokenizer.decode not implemented.")


class WhiteSpaceTokenizer(Tokenizer):

    def __init__(self):
        super(WhiteSpaceTokenizer, self).__init__("WhiteSpaceTokenizer")

    def __repr__(self) -> str:
        return "WhiteSpaceTokenizer()"

    def encode(self, inp: bytes) -> List[bytes]:
        return inp.strip().split()

    def decode(self, inp: List[bytes]) -> bytes:
        return b" ".join(inp)

class AdviceTokenizer(Tokenizer):
    def __init__(self, ratio=0.35, flip_ratio=0.5, crl_factor=-1, sample_mode=0, formalize=False):
        self.sep = b'</SEP>'
        self.ratio = ratio
        self.flip_ratio = flip_ratio
        self.crl_factor = crl_factor # curriculum leanring ratio
        self.random_with_epoch = False # set true to enable randomness w.r.t. epoch number
        self.formalize = formalize # set true to formalize advice, expecially for null advice case
        super(AdviceTokenizer, self).__init__("AdviceTokenizer")
        if sample_mode ==  0: # vannilla mode: with shuffling, coin flipping and sampling
            self.SampleAdvice = self.sample_advice 
        elif sample_mode == 1: # curriculum learning mode
             self.SampleAdvice = self.sample_advice_crl
        elif sample_mode == 2: # fix mode: with shuffling, no coin flipping and no sampling
            self.SampleAdvice = self.sample_advice_fixn
        elif sample_mode == 3: # fix mode: with shuffling, no coin flipping and with sampling
            self.SampleAdvice = self.sample_advice_no_flip
        elif sample_mode == 4: # vanilla mode without deterministic random generator
            self.SampleAdvice = self.sample_advice 
            self.random_with_epoch = True
        elif sample_mode == 5: # raw mode: without shuffling, no coin flipping and no sampling
            self.SampleAdvice = self.raw_advice
        elif sample_mode == 6: # mask mode: randomly mask out some of the adivces
            self.SampleAdvice = self.sample_advice_mask
        elif sample_mode == 7: # vannilla mode: keep all reorder information
            self.SampleAdvice = self.sample_advice_keep_reorder
        elif sample_mode == -1: # non mode: return non advice
            self.SampleAdvice = self.return_non_adivce
        else:
            raise ValueError('Invalid sample mode: %d' % sample_mode)

    def __repr__(self) -> str:
        return "AdviceTokenizer()"

    def encode(self, inp: bytes, random_generator: RandomGenerator, epoch: int = -1) -> List[bytes]:
        inp = inp.strip().split(self.sep)

        if self.random_with_epoch:
            random_generator = RandomGenerator(epoch) # re-inint random generator with seed as current epoch number
        advice = self.SampleAdvice(inp, random_generator=random_generator, epoch=epoch)
        # print(epoch,advice)
        # print(b' '.join(advice).decode())
        # xxxx
        return advice
        # return inp.strip().split()

    def decode(self, inp: List[bytes]) -> bytes:
        return b" ".join(inp)

    def formalize_advice(self, l, mark):
        if len(l) > 0:
            return [mark]+l
        else:
            return [mark]+[b'</NULL>']

    def return_non_adivce(
        self,
        advice, 
        random_generator=None, 
        epoch=-1,
        no_sample=False,
        ):
        return []

    def sample_advice(
        self,
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
            # ratio
            _f = random_generator.uniform_ratio(th=1.0)
            _r = random_generator.uniform_ratio(th=ratio)

            l = l[:round(_r*len(l))] if _f > (1-self.flip_ratio) else []
            l = l[:max_len] if len(l)>max_len else l

            return l

        # if self.crl_factor < 0:
        #     ratio = self.ratio
        # else:
        #     ratio = min(self.crl_factor/epoch, 1.0) * self.ratio
        #     ratio = 0 if ratio < 0.05 else ratio # force turn into baseline traning if ratio is too low

        ratio = self.ratio
        A = _sample(A, ratio=ratio, max_len=5)
        T = _sample(T, ratio=ratio, max_len=2)
        M = _sample(M, ratio=ratio, max_len=3)
        if self.formalize:
            
            A = self.formalize_advice(A, mark=b"</A>")
            T = self.formalize_advice(T, mark=b"</T>")
            M = self.formalize_advice(M, mark=b"</R>")
        advice = b" ".join(A + T + M).split()

        return advice

    def sample_advice_crl(
        self,
        advice, 
        random_generator=None, 
        epoch=-1,
        no_sample=False,
        ):
        """
        Sample advice in a curriculum learning way:
            linearly decrease the # advices from max to 0; without coin flipping
            -self.crl_factor: refers to the # epoch of the end of crl training
        """ 

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
            # ratio
            _r = random_generator.uniform_ratio(th=ratio)

            l = l[:round(_r*len(l))]
            l = l[:max_len] if len(l)>max_len else l

            return l
        if self.crl_factor < 0:
            ratio = self.ratio
        else:
            ratio = min(self.crl_factor/epoch, 1.0) * self.ratio
            ratio = 0 if ratio < 0.05 else ratio # force turn into baseline traning if ratio is too low
        # print("*****")    
        # print(ratio)
        assert self.crl_factor > 0
        ratio = ratio*(1-epoch/self.crl_factor)
        max_len = [5,2,3]
        for i,m in enumerate(max_len):
            m = int(round(m*(1-epoch/self.crl_factor)))
            max_len[i] = m
        
        A = _sample(A, ratio=ratio, max_len=max_len[0])
        T = _sample(T, ratio=ratio, max_len=max_len[1])
        M = _sample(M, ratio=ratio, max_len=max_len[2])
        advice = b" ".join(A + T + M).split()

        return advice

    def sample_advice_fixn(
        self,
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

        ratio = self.ratio
        max_len = [5,2,3]
        A = _sample(A, ratio=ratio, max_len=max_len[0])
        T = _sample(T, ratio=ratio, max_len=max_len[1])
        M = _sample(M, ratio=ratio, max_len=max_len[2])
        advice = b" ".join(A + T + M).split()

        return advice

    def sample_advice_no_flip(
        self,
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
            # ratio
            _r = random_generator.uniform_ratio(th=ratio)

            l = l[:round(_r*len(l))]
            l = l[:max_len] if len(l)>max_len else l

            return l

        # if self.crl_factor < 0:
        #     ratio = self.ratio
        # else:
        #     ratio = min(self.crl_factor/epoch, 1.0) * self.ratio
        #     ratio = 0 if ratio < 0.05 else ratio # force turn into baseline traning if ratio is too low

        ratio = self.ratio
        max_len = [5,2,3]
        A = _sample(A, ratio=ratio, max_len=max_len[0])
        T = _sample(T, ratio=ratio, max_len=max_len[1])
        M = _sample(M, ratio=ratio, max_len=max_len[2])
        advice = b" ".join(A + T + M).split()

        return advice

    def raw_advice(
        self,
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
        if self.formalize:
            A = self.formalize_advice(A,mark=b"</A>")
            T = self.formalize_advice(T,mark=b"</T>")
            M = self.formalize_advice(M,mark=b"</R>")

        advice = b" ".join(A + T + M).split()
        # print(advice)
        return advice

    def sample_advice_mask(
        self,
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


        def _mask(l, ratio):
            # shuf
            l = random_generator.shuffle(l)
            # ratio
            _f = random_generator.uniform_ratio(th=1.0)
            _r = random_generator.uniform_ratio(th=ratio)
            if _f > 0.2:
                masked_l = l
            
            else: # for the rest 20%, conduct masking
                index = [i for i in range(len(l))]
                index = random_generator.shuffle(index)
                index = index[:round(_r*len(index))]
                masked_l = [e for i,e in enumerate(l) if i in index]
 
            return masked_l
            
        ratio = self.ratio
        A = _mask(A, ratio=ratio)
        T = _mask(T, ratio=ratio)
        M = _mask(M, ratio=ratio)
        advice = b" ".join(A + T + M).split()
        
        return advice


    def sample_advice_keep_reorder(
        self,
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
        def _sample(l, ratio, max_len, keep=False):
            # shuf
            l = random_generator.shuffle(l)
            if keep and len(l)>0 and l[0]!=b'': # keep at least one advice
                at_least = [l[0]]
            # ratio
            _f = random_generator.uniform_ratio(th=1.0)
            _r = random_generator.uniform_ratio(th=ratio)

            l = l[:round(_r*len(l))] if _f > (1-self.flip_ratio) else []
            l = l[:max_len] if len(l)>max_len else l
            if keep and len(l)>0 and l[0]!=b'':
                l = at_least + l

            return l

        ratio = self.ratio

        A = _sample(A, ratio=ratio, max_len=5)
        T = _sample(T, ratio=ratio, max_len=2)
        M = _sample(M, ratio=ratio, max_len=3, keep=True) # only shuffle, keep all reorder info
        # if len(M)>0 and M[0] == b'':
        #     print(M)
        #     xxx
        if self.formalize:
            A = self.formalize_advice(A, mark=b"</A>")
            T = self.formalize_advice(T, mark=b"</T>")
            M = self.formalize_advice(M, mark=b"</R>")
        advice = b" ".join(A + T + M).split()

        return advice