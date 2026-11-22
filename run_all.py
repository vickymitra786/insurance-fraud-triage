"""Rebuild every number, figure, website file and the paper: python run_all.py. Author: Vivek Mitra"""
import shutil
import subprocess
import sys
from pathlib import Path

import check_paper
import export_site
import fetch_data
import figures
import model
import test_core

HERE = Path(__file__).resolve().parent

print("1/6 accuracy checks"); test_core.main()
print("2/6 data");            fetch_data.main()
print("3/6 model");           model.main()
print("4/6 figures");         figures.main()
print("5/6 website");         export_site.main()
print("6/6 paper")
paper = HERE / "paper"
if not (paper / "node_modules").exists():
    subprocess.run("npm install --no-audit --no-fund", cwd=paper, shell=True, check=True)
subprocess.run([shutil.which("node"), "build_paper.js"], cwd=paper, check=True)
check_paper.main()
if sys.platform == "win32":  # PDF needs Microsoft Word
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(paper / "to_pdf.ps1")], check=True)
print("done")
