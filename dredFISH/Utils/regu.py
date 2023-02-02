"""Utilities for spatial registration of dredFISH/MERFISH data
"""

import numpy as np
import pandas as pd
import nrrd
import ants
import nibabel as nib
import os
import h5py
import glob
import logging


from .__init__plots import *
sns.set_style('white')
from . import imageu
from . import basicu
from . import powerplots

class RegData():
    """
        self.points_raw -- starting point
        self.points_rot -- after rotation
        
        self.img_rot -- image after 
        self.img_rotpad -- image aligned
        self.img_affine = ..
        self.img_reg = ..
        
        self.trans_rot
        self.trans_pad

        self.img_rot_coords

        
        self.ccfidx -- matched CCF plate index
        self.img_ccf_template -- matched CCF template
        self.img_ccf_annot -- matched CCF annotation (sid)
    """
    def __init__(self, name, points_raw):
        self.name = name 
        self.points_raw = points_raw # 2d ndarray
        # self.pca_rotate()
        return
    
    def add_matched_ccf(self, idx_ccf, img_ccf_template, img_ccf_annot):
        self.idx_ccf = idx_ccf 
        self.img_ccf_template = img_ccf_template
        self.img_ccf_annot = img_ccf_annot
        assert self.img_ccf_annot.dtype == 'uint32' # has to be this to avoid error
        assert self.img_ccf_template.shape == self.img_ccf_annot.shape
        return 
    
    def pca_rotate(self, allow_reflection=False):
        prot, Vt = imageu.pca_rotate(self.points_raw, allow_reflection=allow_reflection)
        self.points_rot = prot
        self.trans_rot = Vt.T # V -- right dot
        self.img_rot, self.img_rot_coords = imageu.pointset_to_image(prot, resolution=10, return_coords=True)
        return 
    
    def flip_rotated(self):
        prot = imageu.flip_points(self.points_rot)
        self.points_rot = prot 
        self.trans_rot = self.trans_rot.dot(np.array([[-1,0],[0,-1]]))
        self.img_rot, self.img_rot_coords = imageu.pointset_to_image(prot, resolution=10, return_coords=True) # update
        return 
    
    def pad(self, fixed_shape=(0,0)):
        if fixed_shape == (0,0):
            fixed_shape = self.img_ccf_template.shape
        self.img_rotpad, self.trans_pad = imageu.broad_padding(self.img_rot, fixed_shape)
        return 
    
    def add_affine(self, reg):
        """Record affine results
        """
        # assert len(reg['fwdtransforms']) == 1
        # assert len(reg['invtransforms']) == 1

        self.img_affine = reg['warpedmovout'].numpy()
        self.img_affine_inv = reg['warpedfixout'].numpy()

        self.trans_affine = reg['fwdtransforms'][0]
        self.trans_affine_inv = reg['invtransforms'][0]
        return
        
    def add_syn(self, reg):
        """Record syn results
        """
        self.img_syn = reg['warpedmovout'].numpy()
        self.img_syn_inv = reg['warpedfixout'].numpy()

        self.trans_syn = reg['fwdtransforms']
        self.trans_syn_inv = reg['invtransforms']
        return
    
    def run_affine(self, 
                  outprefix,
                  downsample_factor=1, 
                  type_of_transform='Affine',
                  aff_metric='GC', # 'meansquares', 'MI'
                  verbose=False,
                  grad_step=0.2, # step of gradient
                  flow_sigma=3, # gaussian regularize the flow but not everything
                  total_sigma=0, # ?
                  aff_iterations=(1000,1000,1000), # max iterations
                  aff_shrink_factors=(10, 4, 2), # shrink -- no effect 1
                  aff_smoothing_sigmas=(2, 2, 1), # smooth -- no effect 0
                  aff_sampling=0, # nbins or radius; should not affect results for GC and meaasures
                  **kwargs,
                  ):
        # data 
        fixed = self.img_ccf_template
        moving = self.img_rotpad

        # downsample
        if downsample_factor > 1:
            factor = downsample_factor
            fixed = imageu.block_mean(fixed, factor) 
            moving = imageu.block_mean(moving, factor)

        # normalization (norm intensities to [0,1] max 99%)
        fp = imageu.max_norm(fixed).astype(np.float32)
        mp = imageu.max_norm(moving).astype(np.float32)

        # format
        fant = ants.from_numpy(fp)
        mant = ants.from_numpy(mp)

        reg = ants.registration(fant, mant, 
                          type_of_transform=type_of_transform, 
                          initial_transform=None, 
                          outprefix=outprefix, 
                          mask=None, 
                          verbose=verbose, 
                          random_seed=0, 
                          aff_metric=aff_metric, #
                          aff_random_sampling_rate=1, # 1 is all 
                          grad_step=grad_step, # step of gradient
                          flow_sigma=flow_sigma, # gaussian regularize the flow but not everything
                          total_sigma=total_sigma, # ?
                          aff_iterations=aff_iterations, # max iterations
                          aff_shrink_factors=aff_shrink_factors, # shrink -- no effect 1
                          aff_smoothing_sigmas=aff_smoothing_sigmas, # smooth -- no effect 0
                          aff_sampling=aff_sampling, # nbins or radius; should not affect results for GC and meaasures
                          **kwargs,
                          )
        self.add_affine(reg)
        return reg
        
    def run_syn(self, 
                outprefix,
                downsample_factor=1, 
                type_of_transform='SyNOnly', # SyNCC (CC) or SyNOnly (MI)
                syn_metric='MI', # 'CC' 5
                syn_sampling=20, # sync_sampling determines nbins or region radius meansquares, CC, MI/mattes, demons
                verbose=False,
                grad_step=0.2, # step of gradient
                flow_sigma=3, # gaussian regularize the flow but not everything
                total_sigma=0, # ?
                reg_iterations=(100, 100, 50, 30, 0), # what are the resolutions?
                **kwargs,
                ):
        # data 
        fixed = self.img_ccf_template
        moving = self.img_affine

        # downsample
        if downsample_factor > 1:
            factor = downsample_factor
            fixed = imageu.block_mean(fixed, factor) 
            moving = imageu.block_mean(moving, factor)

        # normalization (norm intensities to [0,1] max 99%)
        fp = imageu.max_norm(fixed).astype(np.float32)
        mp = imageu.max_norm(moving).astype(np.float32)

        # format
        fant = ants.from_numpy(fp)
        mant = ants.from_numpy(mp)

        reg = ants.registration(fant, mant, 
                          outprefix=outprefix,
                          type_of_transform=type_of_transform, 
                          syn_metric=syn_metric,
                          syn_sampling=syn_sampling,
                          grad_step=grad_step, # step of gradient
                          flow_sigma=flow_sigma, # gaussian regularize the flow but not everything
                          total_sigma=total_sigma, # ?
                          reg_iterations=reg_iterations, 
                          verbose=verbose, 
                          random_seed=0, 
                          **kwargs,
                          )

        self.add_syn(reg)
        return reg
        
    def render_img(self, mode, plot=True):
        if mode == 'raw':
            img_plot = imageu.pointset_to_image(self.points_raw, resolution=10)
        elif mode == 'rot':
            img_plot = self.img_rot
        elif mode == 'ccf':
            img_plot = self.img_ccf_template
        elif mode == 'rotpad':
            img_plot = self.img_rotpad
        elif mode == 'affine':
            img_plot = self.img_affine
        elif mode == 'syn':
            img_plot = self.img_syn
        else:
            raise ValueError('No such image')
            
        mat = imageu.max_norm(img_plot)
        
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(mat)
            ax.axis('off')
            plt.show()
            return 
        else:
            return mat
        
    def render_affine_comparison(self, plot=True):
        """
        """
        keys = [self.name, f'{self.name} moved', 'CCF', 'CCF moved', ]
        plotmats = [
            self.render_img('rotpad', plot=False), 
            self.render_img('affine', plot=False), 
            self.render_img('ccf', plot=False),
            self.img_affine_inv, 
        ]
        if plot:
            fig, axs = plt.subplots(1,4,figsize=(4*4,3*1))
            for ax, key, mat in zip(axs.flat, keys, plotmats):
                ax.imshow(mat)
                ax.set_title(key)
                ax.axis('off')
            fig.subplots_adjust(wspace=0)
            plt.show()
        else:
            return keys, plotmats
         
    def render_flow(self, dfactor=30, plot=True):
        """
        """
        fwdflow = nib.load([file for file in self.trans_syn if file.endswith('nii.gz')][0])
        fwdflowmat = fwdflow.get_fdata()[:,:,0,0,:]

        u = fwdflowmat[:,:,0]
        v = fwdflowmat[:,:,1]
        # dimx, dimy = u.shape
        # x, y = np.meshgrid(np.arange(dimx), np.arange(dimy))
        ud = imageu.block_mean(u, dfactor)
        vd = imageu.block_mean(v, dfactor)

        if plot:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.quiver(ud, vd)
            ax.set_aspect('equal')
            ax.set_title('Forward flow')
            plt.show()
        else:
            return ud, vd
    
    def render_syn_comparison(self, flow=True, plot=True):
        """
        """
        keys = [self.name, f'{self.name} moved', 'CCF', 'CCF moved', 'forward flow']
        plotmats = [
            self.render_img('affine', plot=False), 
            self.render_img('syn', plot=False), 
            self.render_img('ccf', plot=False),
            self.img_syn_inv,
            self.render_flow(plot=False), 
        ]
        if not flow:
            keys = keys[:4]
            plotmats = plotmats[:4]
        n = len(keys)
        
        if plot:
            fig, axs = plt.subplots(1,n,figsize=(4*n,3*1))
            fig.subplots_adjust(wspace=0)
            for i, (ax, key, mat) in enumerate(zip(axs.flat, keys, plotmats)):
                ax.axis('off')
                ax.set_title(key)
                if i < 4:
                    ax.imshow(mat)
                elif i == 4:
                    ud, vd = mat
                    ax.quiver(ud, vd)
                    ax.set_aspect('equal')
        else:
            return keys, plotmats
    
    def render_all_comparison(self, plot=True):
        """
        """
        keys = ['Raw', 'Rotated', 'Affined', 'Registered', 'CCF']
        plotmats = [
            self.render_img('raw', plot=False), 
            self.render_img('rotpad', plot=False), 
            self.render_img('affine', plot=False), 
            self.render_img('syn', plot=False), 
            self.render_img('ccf', plot=False),
        ]
        n = len(keys)
        
        if plot:
            fig, axs = plt.subplots(1,n,figsize=(4*n,3*1))
            fig.subplots_adjust(wspace=0)
            for i, (ax, key, mat) in enumerate(zip(axs.flat, keys, plotmats)):
                ax.axis('off')
                ax.set_title(key)
                ax.imshow(mat)
            plt.show()
        else:
            return keys, plotmats

    def render_sanity_check(self, plot=True):
        """
        """
        keys = [f'Raw ({self.name})', 'Rotated', 'CCF']
        plotmats = [
            self.render_img('raw', plot=False), 
            self.render_img('rotpad', plot=False), 
            self.render_img('ccf', plot=False),
        ]
        n = len(keys)
        if plot:
            fig, axs = plt.subplots(1,n,figsize=(4*n,3*1))
            fig.subplots_adjust(wspace=0)
            for i, (ax, key, mat) in enumerate(zip(axs.flat, keys, plotmats)):
                ax.axis('off')
                ax.set_title(key)
                ax.imshow(mat)
            plt.show()
        else:
            return keys, plotmats
        
    def save(self, file, force=False):
        """
        """
        keys = [
         'name',

         'idx_ccf',
         'img_ccf_template',
         'img_ccf_annot',

         'points_raw',

         # 'rot'
         'points_rot',
         'img_rot',
         'trans_rot',
         'img_rot_coords',

         # 'rotpad'
         'img_rotpad',
         'trans_pad',

         # 'affine',
         'img_affine',
         'img_affine_inv',
         'trans_affine',
         'trans_affine_inv',

         # 'syn',
         'img_syn',
         'img_syn_inv',
         'trans_syn',
         'trans_syn_inv',

         # results
         'region_id',
         'region_acronym',
         'region_color',
        ]
        assert file.endswith('.hdf5')
        if not force:
            assert not os.path.isfile(file)

        with h5py.File(file, 'w') as fh:
            for key in keys:
                fh.create_dataset(key, data=getattr(self, key))
        print(f"saved to {file}")
        return
    
    def apply_transformation(self, mode='register', interpolator='linear'):
        """
        Mode: choose from
        - preproc
        - affine
        - register
        """
        assert mode in ['preproc', 'affine', 'register']

        ## Preproc
        # apply rotation, start from scratch (self.points_raw)
        ps_rot = (self.points_raw).dot(self.trans_rot)
        # turn to image
        img = imageu.pointset_to_image(ps_rot, resolution=10)
        # pad zeros 
        img = np.pad(img, self.trans_pad)
        
        if mode == 'preproc':
            return img
        else:
            ## prepare 
            fixed = ants.from_numpy(self.img_ccf_template.astype(np.float32))
            moving = ants.from_numpy(img.astype(np.float32))
            
            # apply affine
            tx = ants.read_transform(self.trans_affine)
            # affined = tx.apply_to_image(moving) # equivolent, without interpolator
            affined = ants.apply_transforms(fixed, moving, self.trans_affine, 
                interpolator=interpolator,
                ) 
                
            if mode == 'affine':
                return affined.numpy()
            
            elif mode == 'register': 
                # apply syn
                registered = ants.apply_transforms(fixed, affined, self.trans_syn, interpolator=interpolator)
                return registered.numpy() 

    def apply_inv_transformation(self, img_mask, mode='moved', interpolator='nearestNeighbor', has_components=False):
        """
        """
        assert mode in ['moved', 'pre-affine', 'pre-register']

        fixed = ants.from_numpy(self.img_affine)
        moving = ants.from_numpy(img_mask, has_components=has_components)
        moved_presyn = ants.apply_transforms(fixed, moving, self.trans_syn_inv, interpolator=interpolator)
        if mode == 'pre-register':
            return moved_presyn.numpy()
        elif mode == 'pre-affine' or mode == 'moved':
            fixed = ants.from_numpy(self.img_rotpad)
            moved_preaffine = ants.apply_transforms(fixed, moved_presyn, 
                self.trans_affine, 
                whichtoinvert=[True],
                interpolator=interpolator,
                )
            return moved_preaffine.numpy()
    
    def assign_region_ids(self, mask, mode='moved'):
        """For each point (cell), assign region id from the mask
        the mask is presumably from the Allen Brain Atlas (regions are coded)
        """
        assert mode in ['fixed', 'pre-affine', 'pre-register', 'moved']
        # point coordinates to row and col index ids
        rowidx, colidx = imageu.coords_to_imgidx(self.points_rot[:,0], 
                                         self.points_rot[:,1],
                                         self.img_rot_coords)
        rowidx += self.trans_pad[0][0]
        colidx += self.trans_pad[1][0] # zero padding

        # mask
        mask = mask.astype('uint32')
        if mode == 'fixed':
            mask_moved = mask
        else:
            # move mask
            mask_moved = self.apply_inv_transformation(
                mask, 
                mode=mode, 
                interpolator='nearestNeighbor')

        # assign region ids
        region_ids = mask_moved[rowidx, colidx] 
        self.region_ids = region_ids
        return region_ids 

def read_registered_data(file):
    """
    """
    obj = RegData('', np.zeros((2,2))) # create an empty one
    with h5py.File(file, 'r') as fh:
        for key in fh.keys():
            # print(fh[key])
            mat = np.array(fh.get(key))
            if mat.shape == ():
                mat = mat[()]
               
            setattr(obj, key, mat) 
        
        # special treatment
        for key in ['trans_affine', 'trans_affine_inv',]:
            setattr(obj, key, getattr(obj, key).decode())
        for key in ['trans_syn', 'trans_syn_inv', 'region_acronym', 'region_color',]:
            setattr(obj, key, 
                np.array([item.decode() for item in getattr(obj, key)])
                )

    return obj

def check_run(XY, 
    ccf_template, 
    ccf_annot, 
    ccf_maps,
    idx_ccf, 
    name="",
    flip=False, 
    plot=True,
    ):
    """A n by 2 matrix of cell locations
    ccf_template and ccf_annot are 3-dimensional matrices of the same shape
    """
    # initiation
    data = RegData(name, XY)
    # rotate
    data.pca_rotate()
    if flip:
        data.flip_rotated()
    # add CCF
    ccf_template_2d = ccf_template[idx_ccf,:,:]
    ccf_annot_2d = basicu.encode_mat(ccf_annot[idx_ccf,:,:], ccf_maps['encode']).astype('uint32')
    data.add_matched_ccf(
        idx_ccf, 
        ccf_template_2d, 
        ccf_annot_2d, 
        )
    # pad to ccf
    data.pad()

    if plot:
        data.render_sanity_check()
    
    return data

def real_run(XY, 
    ccf_template, 
    ccf_annot, 
    ccf_maps,
    idx_ccf, 
    flip=False, # depending on the check run -- do you need to flip it?
    name='', # a name
    outprefix='', # output
    force=False, # force overwrite
    type_of_transform='SyNOnly', # SyNCC (CC) or SyNOnly (MI)
    syn_metric='MI', # 'CC' 5
    syn_sampling=20, # sync_sampling determines nbins or region radius meansquares, CC, MI/mattes, demons
    verbose=False,
    grad_step=0.2, # step of gradient
    flow_sigma=3, # gaussian regularize the flow but not everything
    total_sigma=0, # ?
    reg_iterations=(100, 100, 50, 30, 0), 
    ):
    """A n by 2 matrix of cell locations
    """
    # check output files
    if outprefix:
        files = glob.glob(outprefix+'*')
        if len(files) > 0:
            if force: 
                for file in files:
                    os.remove(file)
            else:
                raise ValueError(f'Files with {outprefix} already exist, choose a different one to avoid potential error in ANTs')
    
    # outputs
    outprefix_affine = outprefix + 'affine_'
    outprefix_syn    = outprefix + 'syn_'
    output           = outprefix + 'registered.hdf5'
    logging.info(f"To generate: {outprefix_affine}\n {outprefix_syn}\n {output}")

    # preproc
    data = check_run(XY, 
        ccf_template, 
        ccf_annot, 
        ccf_maps,
        idx_ccf, 
        name=name,
        flip=flip, 
        plot=False,
    )

    # affine
    data.run_affine(outprefix_affine)
    # syn
    data.run_syn(
        outprefix_syn,
        type_of_transform=type_of_transform, # SyNCC (CC) or SyNOnly (MI)
        syn_metric=syn_metric, # 'CC' 5
        syn_sampling=syn_sampling, # sync_sampling determines nbins or region radius meansquares, CC, MI/mattes, demons
        verbose=verbose,
        grad_step=grad_step, # step of gradient
        flow_sigma=flow_sigma, # gaussian regularize the flow but not everything
        total_sigma=total_sigma, # ?
        reg_iterations=reg_iterations, 
        # **kwargs,
        )
    # check results
    data.render_all_comparison()

    # add region ids to cells
    mask = data.img_ccf_annot
    region_ids = data.assign_region_ids(mask, mode='moved') # inv transform mask back to image (rotated and padded)
    region_colors   = basicu.encode_mat(region_ids, ccf_maps['colorhex'])
    region_acronyms = basicu.encode_mat(region_ids, ccf_maps['acronym'])

    # change from "U"type strings to object so it is supported by h5py
    data.region_id = region_ids
    data.region_color = region_colors.astype(object)
    data.region_acronym = region_acronyms.astype(object)

    # # record and save
    data.save(output, force=force)
    return data


def load_allen_template(
    allen_template_path
    ):
    allen_template = np.load(allen_template_path)
    return allen_template
    

def load_allen_tree(
    allen_tree_path
    ):
    """
    Need to encode Allen regions to smaller integers so it does not get lost in float32 transformation and back during ANTs registration.
    - `uint32` is the native dtype of Allen, and is supported as an input to ANTs
    """
    # tree table
    allen_tree = pd.read_json(allen_tree_path)
    allen_tree['sid'] = allen_tree.index+1

    # maps
    encode_map = allen_tree.set_index('id')['sid'].to_dict()
    encode_map[0] = 0
    
    decode_map = allen_tree.set_index('sid')['id'].to_dict()
    decode_map[0] = 0
    
    color_map = allen_tree.set_index('sid')['rgb_triplet'].to_dict() # integers 0~255
    color_map[0] = [0,0,0]

    colorhex_map = {sid: powerplots.rgb_to_hex(*color) for sid, color in color_map.items()}

    acronym_map = allen_tree.set_index('sid')['acronym'].to_dict()
    acronym_map[0] = '' 
    
    allen_maps = {
        'encode': encode_map, # id to sid
        'decode': decode_map, # id to sid
        'color': color_map, # sid to color (integers [0~255, 0~255, 0~255])
        'colorhex': colorhex_map, # sid to color (#xxxxxx)
        'acronym': acronym_map, # sid to acronym
    }

    return allen_tree, allen_maps

def load_allen_annot(
    allen_annot_path, 
    # encode=False,
    # allen_tree_path='',
    ):
    """
    Need to encode Allen regions to smaller integers so it does not get lost in float32 transformation and back during ANTs registration.
    - `uint32` is the native dtype of Allen, and is supported as an input to ANTs
    """
    # takes ~10 seconds
    allen_annot, meta = nrrd.read(allen_annot_path)
    allen_annot = allen_annot.astype('uint32')
    
    # encode the 3D matrix takes too long (depreciated)
    # if encode:
    #     allen_tree, allen_maps = load_allen_tree(allen_tree_path=allen_tree_path)
    #     encode_map = allen_maps['encode']
    #     allen_annot = basicu.encode_mat(allen_annot, encode_map).astype('uint32')
        
    return allen_annot

def expand_regions(allen_tree, regions, by='acronym'):
    """get all regions (their sid) belong to the specified regions.
    """
    # assert by == 'acronym' # others are not implemented
    stree = allen_tree[allen_tree[by].isin(regions)].copy()
    stree_ids = stree['id'].values
    
    # check which regions has their ids in their structure_id_path
    conds = [0]*len(allen_tree)
    for _id in stree_ids:
        cond = allen_tree['structure_id_path'].apply(lambda x: _id in x)
        conds += cond # or
    conds = conds > 0
                       
    return allen_tree[conds]['sid'].values