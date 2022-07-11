
class TopicClassifier(Classifier): 
    """Uses Latent Dirichlet Allocation to classify cells into regions types. 

    Decision on number of topics comes from maximizing overall entropy. 
    """
    def __init__(self,TG, ordr=4):
        """
        Parameters
        ----------
        TG : TissueGraph
            Required parameter - the TissueGraph object we're going to use for unsupervised clustering. 
        """
        if not isinstance(TG, TissueGraph.TissueGraph): 
            raise ValueError("TopicClassifier requires a (cell level) TissueGraph object as input")

        super().__init__(tax = None)
        self.tax.name = 'topics'
        self._TG = TG
        Env = self._TG.extract_environments(ordr=ordr)
        row_sums = Env.sum(axis=1)
        row_sums = row_sums[:,None]
        Env = Env/row_sums
        self.Env = Env

    def train(self, use_parallel=False, max_num_of_topics=100):
        """fit multiple LDA models and chose the one with highest type entropy
        """
        
        Ntopics = np.arange(2,max_num_of_topics)  
        
        if use_parallel: 
            logging.info("Running LDA in parallel")
            # define function handle
            def lda_fit(n_topics,Env = None):
                lda = LatentDirichletAllocation(n_components=n_topics)
                B = lda.fit(Env)
                T = lda.transform(Env)
                return (B,T)

            # start parallel engine
            rc = ipp.Client()
            dview = rc[:]
            dview.push({'Env': self.Env})
            with dview.sync_imports():
                # import sklearn.decomposition
                pass
            # add env to lda_fit to create new function handle we can use in parallel
            g = functools.partial(lda_fit, Env=self.Env)
            result = dview.map_sync(g,Ntopics)
        else:
            logging.info("Running LDA n serial") 
            result = list()
            for i in range(len(Ntopics)):
                lda = LatentDirichletAllocation(n_components=Ntopics[i])
                B = lda.fit(self.Env)
                T = lda.transform(self.Env)
                result.append((B,T))
                
        IDs = np.zeros((self._TG.N,len(result)))
        for i in range(len(result)):
            IDs[:,i] = np.argmax(result[i][1],axis=1)
        Type_entropy = np.zeros(IDs.shape[1])
        for i in range(IDs.shape[1]):
            _,cnt = np.unique(IDs[:,i],return_counts=True)
            cnt=cnt/cnt.sum()
            Type_entropy[i] = entropy(cnt,base=2) 

        self._lda = result[np.argmax(Type_entropy)][0]

        return self
