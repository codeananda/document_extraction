import asyncio
import os
from pathlib import Path

from main import extract_plan_and_content_latex

# Directory containing the LaTeX and BibTeX files
data_dir = './data/latex'

# Process LaTeX files
latex_papers = list(Path(data_dir).glob("*.tex"))
for latex_paper in latex_papers:
    asyncio.run(extract_plan_and_content_latex(latex_paper, without_embeddings=True))