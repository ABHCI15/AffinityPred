"""Streamlit front-end for single-pair affinity inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from traitlets import default

from inference_library import InferenceConfig, LibraryInferencer


def _sanitize_path(path_text: str) -> Optional[Path]:
    """Convert user input into a ``Path`` if provided, otherwise return ``None``."""

    text = path_text.strip()
    if not text:
        return None
    return Path(text)


@st.cache_resource(show_spinner=False)
def load_inferencer(weights: Optional[Path], device: Optional[str]) -> LibraryInferencer:
    """Instantiate and cache the ``LibraryInferencer`` for reuse across runs."""

    config = InferenceConfig()
    if weights is not None:
        config.weights_path = weights
    if device:
        config.device = device
    return LibraryInferencer(config)


def main() -> None:
    st.set_page_config(page_title="AffinityPred Inference", layout="centered")
    st.title("AffinityPred Single-Pair Inference")
    st.write(
        "Provide a SMILES string and protein sequence to score their predicted "
        "binding affinity (pChEMBL)."
    )

    with st.expander("Model configuration", expanded=False):
        weights_input = st.text_input(
            "Weights file path",
            placeholder="Leave blank to use the most recent best_model_*.pth",
            value="/home/nroethler/Code/abhiram/chemmap/AffinityPred/weights/best_model_20250909_113311.pth"
            
        )
        device_choice = st.selectbox(
            "Inference device",
            options=("auto", "cuda", "cpu"),
            help="Choose the device for inference. 'auto' selects CUDA when available.",
        )

    st.subheader("Input features")
    smiles = st.text_input("SMILES", placeholder="e.g. C1=CC=CC=C1")
    protein_sequence = st.text_area(
        "Protein sequence",
        placeholder="Enter an amino acid sequence",
        height=160,
    )

    run_inference = st.button("Run inference")

    if not run_inference:
        return

    weights_path = _sanitize_path(weights_input)
    device_arg = None if device_choice == "auto" else device_choice

    if not smiles.strip():
        st.warning("Please provide a valid SMILES string before running inference.")
        return
    if not protein_sequence.strip():
        st.warning("Please provide a protein sequence before running inference.")
        return
    if weights_path is not None and not weights_path.is_file():
        st.error(f"Weights file not found: {weights_path}")
        return

    with st.spinner("Loading model and running inference..."):
        try:
            inferencer = load_inferencer(weights_path, device_arg)
            prediction = inferencer.predict(protein_sequence, smiles)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")
            return

    st.success("Inference completed successfully.")
    st.metric(label="Predicted pChEMBL", value=f"{prediction:.3f}")

    st.caption(
        "Predictions use the current AffinityPred model configuration. "
        "Ensure inputs are valid and sequences are in FASTA-style plain text."
    )


if __name__ == "__main__":
    main()
