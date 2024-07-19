#!/usr/bin/env python
import argparse
from dredFISH.Utils.analysisu import *
import socket
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("animal", type=str, help="Name of the animal to analyze")
    parser.add_argument("-p", "--project_path", type=str, dest="project_path", default='/scratchdata1/Images2024/Zach/MouseBrainAtlas', action='store', help="Path to the project directory (default: '/scratchdata1/Images2024/Zach/MouseBrainAtlas')")
    parser.add_argument("-a", "--analysis_path", type=str, dest="analysis_path", default='/scratchdata1/MouseBrainAtlases_V2', action='store', help="Path to the analysis directory (default: '/scratchdata1/MouseBrainAtlases_V0')")

    args = parser.parse_args()
    if not os.path.exists(args.analysis_path):
        os.mkdir(args.analysis_path,mode=0o777)
    if args.animal == 'all':
        pfunc = partial(analyze_mouse_brain_data,
                        project_path=args.project_path,
                        analysis_path=args.analysis_path,
                        verbose=False)
        host = socket.gethostname()
        if host == 'purple':
            animals = ['Tax','WTF01','WTM01','MMSM01']
        elif host == 'blue':
            animals = ['Tax','MMSF01','WTM04','WTM07']
        elif host == 'orange':
            animals = ['Tax','WTF04','WTF06','ASDM02']
        else:
            raise ValueError(f"{host} Host not recognized")
        # animals = ['Tax','WTF01','WTM01','MMSM01','MMSF01','WTM04','WTM07','WTF04','WTF06']
        with multiprocessing.Pool(1) as p:
            p.map(pfunc, animals)
        # for animal in ['Tax','WTF01','WTM01','MMSM01','MMSF01','WTM04','WTM07','WTF04','WTF06']:
        #     pfunc = partial(analyze_mouse_brain_data,
        #                     project_path=args.project_path,
        #                     analysis_path=args.analysis_path,
        #                     verbose=False)
        #     with multiprocessing.Pool(4) as p:
        #         p.map(pfunc, [animal])
        #     try:
        #         analyze_mouse_brain_data(animal,
        #                                 project_path=args.project_path,
        #                                 analysis_path=args.analysis_path,
        #                                 verbose=False)
        #     except Exception as e:
        #         print(animal)
        #         print('Failed')
        #         print(e)
        #         continue
    else:
        analyze_mouse_brain_data(args.animal,
                            project_path=args.project_path,
                            analysis_path=args.analysis_path,
                            verbose=False)