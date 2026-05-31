from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.latex import paper_latex


def test_paper_latex_generates_arxiv_style_tex(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=8, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    output = tmp_path / "paper.tex"

    generated = paper_latex(result.results_path, output, artifacts_dir=bundle.output_dir)
    tex = output.read_text(encoding="utf-8")

    assert generated == str(output)
    assert "\\documentclass[11pt]{article}" in tex
    assert "\\section{Data Layer}" in tex
    assert "ASR corpus manifest recipe summary" in tex
    assert "\\section{Turn Latency and Deployment}" in tex
    assert "\\begin{tabular}" in tex
    assert "Turn-taking failure taxonomy by baseline" in tex
    assert "Streaming ASR failure taxonomy" in tex
    assert "prediction\\_manifest" in tex
    assert "paper-latex --results" in tex
    assert "\\end{document}" in tex
