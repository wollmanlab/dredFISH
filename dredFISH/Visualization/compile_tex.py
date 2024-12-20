import os
import glob
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                datefmt='%m-%d %H:%M:%S', 
                level=logging.INFO,
                )
import subprocess as sp

def generate_figure_text(respth, cmplpth, 
    version='figure',
    nfigs_per_page=None):
    """get all figures from respth
    respth: figure inputs
    cmplpth: compiled results output
    
    version: choose from `graph` or `figure` or `page`
    """
    assert version in ['graph', 'figure', 'page']
    
    # input
    figures = glob.glob("fig*.pdf", dir_fd=respth) 
        
    # rename
    for i, fig in enumerate(figures):
        # remove extra .
        fignew = fig
        while fignew.count('.') > 1:
            fignew = fignew.replace('.', 'p', 1)

        # update name (using symlink)
        src = os.path.join(respth, fig)
        dst = os.path.join(cmplpth, fignew)
        if not os.path.isfile(dst):
            os.symlink(src, dst) 
        figures[i] = fignew

    # prep figure text
    figuretext = ""
    for i, fig in enumerate(figures):
        # clear page?
        if isinstance(nfigs_per_page, int) and (i%nfigs_per_page)==0:
                figuretext += "\n \\clearpage \n"

        if version == 'graph':
            figuretext += f"\\centerline{{\\includegraphics[width=\\textwidth]{{{fig}}}}}\n"
        elif version == 'page':
            figuretext += f"\\includepdf[pages=-]{{{fig}}}\n"
        if version == 'figure':
            figcap = os.path.basename(fig).replace('.pdf', '').replace('_', '-')
            figt = f"""
            \\begin{{figure}}[ht]
                \\centerline{{\\includegraphics[width=0.7\\textwidth]{{{fig}}}}}
                \\caption{{{figcap}}}
            \\end{{figure}}
            """
            figuretext += figt

            
    return figuretext

def generate_tex_insertpage(title, author, figuretext):
    """Insert PDF pages
    """
    latex_letter = f"""\\documentclass[11pt]{{article}}
    \\usepackage{{pdfpages}}
    \\usepackage[utf8]{{inputenc}}
    \\usepackage[legalpaper, margin=0.5in]{{geometry}}
    \\begin{{document}}
    \\title{{{title}}}
    \\author{{{author}}}
    \\maketitle
    {figuretext} % \\includepdf[pages=-]{{fig1-2_basis_space_righthalf_2022-07-20.pdf}}
    \\end{{document}}
    """ 
    return latex_letter

def generate_tex(title, author, figuretext):
    """Insert as graphics
    """
    latex_letter = f"""\\documentclass[11pt]{{article}}
    \\renewcommand{{\\familydefault}}{{\\sfdefault}}
    \\usepackage{{graphicx}}
    \\usepackage[utf8]{{inputenc}}
    \\usepackage[legalpaper, margin=0.5in]{{geometry}}
    \\begin{{document}}
    \\title{{{title}}}
    \\author{{{author}}}
    \\maketitle
    \\listoffigures
    {figuretext}
    % \\centerline{{\\includegraphics[width=\\textwidth]{{fig1-2_basis_space_righthalf_2022-07-20.pdf}}}}
    
    \\end{{document}}
    """ 
    return latex_letter

def main(basepth, 
    subpth_res='figures',
    subpth_cmpl='compiled',
    nfigs_per_page=None,
    title='dredFISH default analysis', 
    author='Fangming'):
    """
    Compile .tex file and generate pdf
    """
    respth = os.path.join(basepth, subpth_res)
    # output
    cmplpth = os.path.join(respth, subpth_cmpl)
    if not os.path.isdir(cmplpth):
        os.mkdir(cmplpth)
    for f in glob.glob(os.path.join(cmplpth, '*')):
        os.remove(f) # remove any existing files there - clean up

    figuretext = generate_figure_text(respth, cmplpth, nfigs_per_page=nfigs_per_page)
    latex_letter = generate_tex(title, author, figuretext)
    # print(figuretext)
    # print(latex_letter)

    texpth = os.path.join(cmplpth, 'compiled.tex')
    with open(texpth, 'w') as fh:
        fh.write(latex_letter)
    logging.info(f"Saved to {texpth}")

    # PDF generation
    try:
        # compiling the doc twice is needed for \listoffigures
        sp.run(['pdflatex', 'compiled.tex'], cwd=cmplpth)
        sp.run(['pdflatex', 'compiled.tex'], cwd=cmplpth)
        logging.info(f"PDF generated")
    except:
        logging.info(f"PDF generation failed")
        
    return

if __name__ == '__main__':
    basepth = '/bigstore/GeneralStorage/Data/dredFISH/Dataset1-t5'
    main(basepth)