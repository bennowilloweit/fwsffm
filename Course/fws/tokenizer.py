class FWSTokenizer():
    fmap = {
        '(' : '',
        ')' : '',
        '[' : '',
        ']' : ''
    }
    scrs_map = {
        '\n' : '<NL>',
        '?' : '<QM>',
        '!' : '<EM>',
        ':' : '<CO>',
        ';' : '<SC>',
        ',' : '<CM>',
        '.' : '<PD>',
        '"' : '<QT>',
        '---' : '<D3>',
        '--' : '<D2>',
        '-' : '<D1>'
        
    }
    inv_scrs_map = {v: k for k, v in scrs_map.items()}
    vsize = 0
    stoi = {}
    itos = {}
    
    
    #sample tokens from text
    def __init__(self, text):
        super().__init__()
        tokens = self.tokenize(text)
        vocab = sorted(list(set(tokens)))
        self.vsize = len(vocab)
        self.stoi = { t:i for i,t in enumerate(vocab) }
        self.itos = { i:t for i,t in enumerate(vocab) }
        
    def vocab_int(self):
        return list(self.itos.keys())
        
    def vocab_str(self):
        return [self.itos[key] for key in self.itos.keys()]  
        
    def tokenize(self, text):
        ptext = text
        for k,v in self.fmap.items():
            ptext = ptext.replace(k, v)
        
        for k,v in self.scrs_map.items():
            ptext = ptext.replace(k, ' ' + v + ' ')
        
        tokens = list(filter(None, ptext.split(' ')))
        return tokens
    
    def tok(self, idx):
        return self.itos.get(idx, '')
    
    def idx(self, tok):
        return self.stoi.get(tok, -1)

    def encode(self, tok_seq):
        return [self.idx(tok) for tok in tok_seq]

    def decode_list(self, idx_seq):
        toks = [self.tok(idx) for idx in idx_seq]
        return [self.inv_scrs_map.get(tok, tok) for tok in toks]
    
    def decode(self, idx_seq):
        ptext = ' '.join([self.tok(idx) for idx in idx_seq])
        for k,v in self.inv_scrs_map.items():
            ptext = ptext.replace(k, v)
        ptext = ptext.replace('\n ', '\n')
        for k,v in self.scrs_map.items():
            ptext = ptext.replace(' ' + k, k)
        return ptext
        