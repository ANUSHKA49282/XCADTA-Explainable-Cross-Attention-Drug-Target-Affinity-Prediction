
import { useEffect, useRef, useState } from "react";
import initRDKit from "@rdkit/rdkit";
import wasmUrl from "@rdkit/rdkit/RDKit_minimal.wasm?url";

let rdkitPromise;

function loadRDKit() {
  if (!rdkitPromise) {
    rdkitPromise = initRDKit({
      locateFile: () => wasmUrl,
    });
  }

  return rdkitPromise;
}

export default function MoleculeViewer({ smiles }) {
  const svgRef = useRef(null);
  const [rdkit, setRdkit] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    loadRDKit()
      .then(setRdkit)
      .catch((err) => {
        console.error("RDKit loading error:", err);
        setError("Could not load RDKit.");
      });
  }, []);

  useEffect(() => {
    if (!rdkit || !svgRef.current) return;

    if (!smiles.trim()) {
      svgRef.current.innerHTML = "";
      setError("");
      return;
    }

    let molecule;

    try {
      molecule = rdkit.get_mol(smiles);

      if (!molecule) {
        svgRef.current.innerHTML = "";
        setError("Invalid SMILES. Please check the molecule.");
        return;
      }

      const svg = molecule.get_svg();
      svgRef.current.innerHTML = svg;
      setError("");
    } catch (err) {
      console.error("Molecule drawing error:", err);
      svgRef.current.innerHTML = "";
      setError("Could not draw this molecule.");
    } finally {
      if (molecule) molecule.delete();
    }
  }, [rdkit, smiles]);

  return (
    <div className="molecule-viewer">
      <div className="field-label">2D Molecular Structure</div>

      {!smiles.trim() ? (
        <p className="molecule-message">
          Select a drug to view its structure.
        </p>
      ) : error ? (
        <p className="molecule-error">{error}</p>
      ) : !rdkit ? (
        <p className="molecule-message">Loading RDKit...</p>
      ) : (
        <div ref={svgRef} className="molecule-canvas" />
      )}
    </div>
  );
}
