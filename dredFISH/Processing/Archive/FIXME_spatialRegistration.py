# Spatial registration: 

    def spatial_registration_preview(self, 
        allen_template, allen_annot, allen_maps,
        idx_ccf, 
        flip=False,
        ):
        """
        Check if the selected allen section make sense at all, and if we need to flip the orientation.
        Nothing is saved and this runs fast.
        """
        from dredFISH.Analysis import regu # the package ANTs is often incompatible with a few others
        spatial_data = regu.check_run(self.XY, 
                                allen_template, 
                                allen_annot, 
                                allen_maps,
                                idx_ccf, 
                                flip=flip)

        return spatial_data 

    def spatial_registration(self, 
        allen_template, allen_annot, allen_maps,
        idx_ccf, 
        flip=False,
        outprefix='',
        force=False,
        ):
        """
        """
        from dredFISH.Analysis import regu # the package ANTs is often incompatible with a few others
        spatial_data = regu.real_run(self.XY, 
                        allen_template,
                        allen_annot,
                        allen_maps,
                        idx_ccf, 
                        flip=flip, 
                        dataset=self.name, # a name
                        outprefix=outprefix, 
                        force=force,
                        )

        # update results to anndata (cell level atrributes)
        self.data.obs['coord_x'] = spatial_data.points_rot[:,0]
        self.data.obs['coord_y'] = spatial_data.points_rot[:,1]
        self.data.obs['region_id'] = spatial_data.region_id
        self.data.obs['region_color'] = spatial_data.region_color 
        self.data.obs['region_acronym'] = spatial_data.region_acronym 
        return spatial_data