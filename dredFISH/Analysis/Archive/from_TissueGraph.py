
    # def multi_optim_Leiden_from_existing_types(self,base_types,types_to_expand, 
    #                                            FeatureMat, max_subtypes = 10000, 
    #                                            opt_params = {'iters' : 10, 'n_consensus' : 50}):
        
    #     def ObjFunLeidenRes_FG(res,FGtosplit,ix,TypeVec,return_types = False):
    #         """
    #         Basic optimization routine for Leiden resolution parameter. 
    #         Implemented using igraph leiden community detection
    #         """
            
    #         mask = np.zeros(TypeVec.shape,dtype='bool')
    #         mask[ix] = True
    #         mx_id = TypeVec[~mask].max()+1
            
    #         # if asked to return types, run this only once, no need for averaging. 
    #         iter_to_avg = opt_params['iters']
    #         if return_types:
    #             iter_to_avg=1
            
    #         EntropyVec = np.zeros(opt_params['iters'])
    #         for i in range(iter_to_avg):
    #             # split cells in the provided graph
    #             SplitTypes = FGtosplit.community_leiden(resolution_parameter=res,objective_function='modularity').membership
    #             # adjust ids - make into numpy array and shift to account for existing types
    #             SplitTypes = np.array(SplitTypes).astype(np.int64) + mx_id
                
    #             # recreate type vector for the whole tissue
    #             TypeVec2 = TypeVec.copy()
    #             TypeVec2[ix] = TypeVec2[ix] + SplitTypes
    #             EntropyVec[i] = self.contract_graph(TypeVec2).cond_entropy()
                
    #         Entropy = EntropyVec.mean()
    #         if return_types: 
    #             return -Entropy,TypeVec2
    #         else:
    #             return(-Entropy)
            
    #     # to ease bookkeeping, multiply cell type integers with a large const so we could add subtypes later
    #     # ix_expand = np.isin(base_types,types_to_expand)
    #     # base_types[ix_expand] = base_types[ix_expand] * max_subtypes
    #     # types_to_expand = [x * max_subtypes for x in types_to_expand]
        
    #     # Build subgraphs    
    #     start = time.time()

    #     print(f"Optimize each type to see if it can be split further")
    #     for i in range(len(types_to_expand)):
            
    #         # get indexes of cells with this type
    #         ix = np.flatnonzero(base_types == types_to_expand[i])
            
    #         # create a subgraph for these cells 
    #         sub_FG = self.build_feature_graph(FeatureMat[ix,:],metric = 'cosine',accuracy=3,return_graph = True)
            
    #         # Cond entropy optimization only for these cells
    #         type_copy = base_types.copy()
    #         n_before = len(np.unique(base_types))
    #         sol = minimize_scalar(ObjFunLeidenRes_FG, bounds = (0.1,30), 
    #                                                   method='bounded',
    #                                                   args = (sub_FG,ix,type_copy),
    #                                                   options={'xatol': 1e-1, 'disp': 3})
    #         # get types
    #         opt_res = sol['x']
    #         ent,base_types = ObjFunLeidenRes_FG(opt_res,sub_FG,ix,base_types,return_types=True)
    #         n_after = len(np.unique(base_types))
    #         print(f'i: {i} time: {time.time()-start:.2f} type before: {n_before} added: {n_after-n_before}')
            
    #     return base_types
                
    
    # def multilayer_Leiden_with_cond_entropy(self,base_types = None, 
    #                                         FeatureMat = None, 
    #                                         return_res = False,
    #                                         opt_params = {'iters' : 10, 'n_consensus' : 50}): 
    #     """
    #         Find optimial clusters by peforming clustering on two-layer graph. 
            
    #         Input
    #         -----
    #         TG : A TissueGraph that has matching SpatialGraph (SG) and FeatureGraph (FG)
    #         optimization is done on resolution parameter  
            
    #     """
    #     start = time.time()
    #     if base_types is not None: 
    #         unq_types = np.unique(base_types)
    #         if FeatureMat is None: 
    #             raise ValueError('if types are supplied then a features matrix must be included')
    #         sub_FGs = list()
    #         all_ix = list()
    #         print(f"Building feature subgraphs for each type")
    #         for i in range(len(unq_types)):
    #             ix = np.flatnonzero(base_types == unq_types[i])
    #             all_ix.append(ix)
    #             # create a subgraph 
    #             sub_FGs.append(self.build_feature_graph(FeatureMat[ix,:],metric = 'cosine',accuracy=1,return_graph=True))
    #         print(f'done, time: {time.time()-start:.2f}')
            

    #     def ObjFunLeidenRes(res,return_types = False):
    #         """
    #         Basic optimization routine for Leiden resolution parameter. 
    #         Implemented using igraph leiden community detection
    #         """
    #         EntropyVec = np.zeros(opt_params['iters'])
    #         if base_types is not None:
    #             for i in range(opt_params['iters']):
    #                 all_sub_Types = list()
    #                 for j in range(len(unq_types)):
    #                     sub_TypeVec = sub_FGs[j].community_leiden(resolution_parameter=res,
    #                                                               objective_function='modularity').membership
    #                     sub_TypeVec = np.array(sub_TypeVec).astype(np.int64)
    #                     all_sub_Types.append(sub_TypeVec)
                        
    #                 # bookeeping: rename all clusters so that numbers are allways unique. 
    #                 # find the largest number of subtypes we need to add
    #                 # take log10 and ceil so that we find a place value that is larget that that
    #                 # multiply base_types with that value. 
    #                 mxdec = np.max(10**np.ceil(np.log10([len(np.unique(x)) for x in all_sub_Types])))
    #                 TypeVec = base_types * mxdec
    #                 for j in range(len(all_ix)): 
    #                     TypeVec[all_ix[j]] = TypeVec[all_ix[j]] + all_sub_Types[j]
    #                 EntropyVec[i] = self.contract_graph(TypeVec).cond_entropy()
    #             Entropy = EntropyVec.mean()
                    
    #         else:
    #             for i in range(opt_params['iters']):
    #                 TypeVec = self.FG.community_leiden(resolution_parameter=res,
    #                                                    objective_function='modularity').membership
    #                 TypeVec = np.array(TypeVec).astype(np.int64)
    #                 EntropyVec[i] = self.contract_graph(TypeVec).cond_entropy()
    #             Entropy = EntropyVec.mean()
    #         if return_types: 
    #             return -Entropy,TypeVec
    #         else:
    #             return(-Entropy)

    #     print(f"Calling initial optimization")
    #     sol = minimize_scalar(ObjFunLeidenRes, bounds = (0.1,30), 
    #                                            method='bounded',
    #                                            options={'xatol': 1e-2, 'disp': 3})
    #     opt_res = sol['x']
        
    #     # consensus clustering
    #     TypeVec = np.zeros((self.N,opt_params['n_consensus']))
    #     for i in range(opt_params['n_consensus']):
    #         ent,TypeVec[:,i] = ObjFunLeidenRes(opt_res,return_types = True)
            
    #     if opt_params['n_consensus']>1:
    #         cmb = np.array(list(itertools.combinations(np.arange(opt_params['n_consensus']), r=2)))
    #         rand_scr = np.zeros(cmb.shape[0])
    #         for i in range(cmb.shape[0]):
    #             rand_scr[i] = adjusted_rand_score(TypeVec[:,cmb[i,0]],TypeVec[:,cmb[i,1]])
    #         rand_scr = squareform(rand_scr)
    #         total_rand_scr = rand_scr.sum(axis=0)
    #         TypeVec = TypeVec[:,np.argmax(total_rand_scr)]
                                                  

    #     print(f"Number of types: {len(np.unique(TypeVec))} initial entropy: {-sol['fun']} number of evals: {sol['nfev']}")
    #     if return_res: 
    #         return TypeVec,opt_res
    #     else: 
    #         return TypeVec


        # Add taxonomy (type) information. 
        # There are three possiblities, unsupervized, supervized, and hybrid
        
#         # completely unsupervised:
#         if celltypes_org is None: 
#             # cluster cell types optimally - all cells from scratch
#             celltypes,optres = TG.multilayer_Leiden_with_cond_entropy(return_res = True)
            
#             cell_taxonomy = Taxonomy()
#             cell_taxonomy.add_labels(feature_mat = FISHbasis,labels = celltypes)
        
#         # hybrid: 
#         elif expand_types is not None : 
#             celltypes = celltypes_org.copy()
#             mx_subtypes = 10000
#             celltypes = TG.multi_optim_Leiden_from_existing_types(base_types = celltypes,
#                                                                         types_to_expand = expand_types,
#                                                                         FeatureMat = FeatureMat,
#                                                                         max_subtypes = mx_subtypes)
#             ix_exanded_types = np.isin(celltypes_org,expand_types)
#             cell_taxonomy.add_types(feature_mat = FISHbasis[ix_exanded_types,:],labels = celltypes[ix_exanded_types])
#         # completely supervised
#         else: 
#             celltypes = celltypes_org.copy()
               
        
#         # add types and key data