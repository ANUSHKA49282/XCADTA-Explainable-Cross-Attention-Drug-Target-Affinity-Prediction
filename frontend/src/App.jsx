
import { useEffect, useState } from "react";
import {
  Activity, Atom, BrainCircuit, ChartNoAxesCombined,
  ChevronRight, CircleHelp, Clock3, Database,
  FlaskConical, History, LayoutDashboard, Menu,
  Microscope, Network, ShieldCheck,
  Sparkles, Dna, ArrowUpRight, ArrowRight, Zap,
  X, CheckCircle2, Info, Download
} from "lucide-react";

import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar
} from "recharts";

import "./App.css";
import MoleculeViewer from "./MoleculeViewer";

const API = "http://127.0.0.1:8000";

const metrics = [
  { label: "Test MSE", value: "0.2478", sub: "Lower is better", icon: Activity, color: "violet" },
  { label: "Concordance Index", value: "0.8146", sub: "Higher is better", icon: ChartNoAxesCombined, color: "cyan" },
  { label: "R²m Score", value: "0.6086", sub: "Higher is better", icon: BrainCircuit, color: "blue" },
];

const historyData = [
  { name: "Previous", mse: 0.316 },
  { name: "Latest", mse: 0.248 },
];

const navItems = [
  { id: "dashboard", label: "Overview", icon: LayoutDashboard, group: "WORKSPACE" },
  { id: "predict", label: "Affinity Prediction", icon: FlaskConical },
  { id: "explain", label: "Explainability", icon: Microscope },
  { id: "history", label: "Prediction History", icon: History },
  { id: "model", label: "Model Architecture", icon: Network, group: "PLATFORM" },
];

function Sidebar({ page, setPage, mobileOpen, setMobileOpen }) {
  return (
    <>
      {mobileOpen && <div className="mobile-overlay" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Atom size={25} /></div>
          <div>
            <div className="brand-name">XCADTA</div>
            <div className="brand-caption">AI-POWERED AFFINITY PLATFORM</div>
          </div>
          <button className="icon-button mobile-close" onClick={() => setMobileOpen(false)}>
            <X size={18} />
          </button>
        </div>

        <div className="workspace-chip">
          <div className="workspace-dot" /> XCADTA Workspace <ChevronRight size={14} />
        </div>

        <nav className="nav-list">
          {navItems.map(item => (
            <div key={item.id}>
              {item.group && <div className="nav-group">{item.group}</div>}
              <button
                className={`nav-item ${page === item.id ? "active" : ""}`}
                onClick={() => { setPage(item.id); setMobileOpen(false); }}
              >
                <item.icon size={18} />
                <span>{item.label}</span>
                {page === item.id && <span className="nav-active-dot" />}
              </button>
            </div>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="profile">
            <div className="profile-avatar">XC</div>
            <div className="profile-info">
              <strong>XCADTA Project</strong>
              <span>Local environment</span>
            </div>
            <CircleHelp size={17} className="muted-icon" />
          </div>
        </div>
      </aside>
    </>
  );
}

function Header({ page, setMobileOpen }) {
  const titles = {
    dashboard: ["Overview"],
    predict: ["Affinity Prediction"],
    explain: ["Explainability"],
    history: ["Prediction History"],
    model: ["Model Architecture"],
  };

  return (
    <header className="topbar">
      <button className="icon-button menu-button" onClick={() => setMobileOpen(true)}>
        <Menu size={20} />
      </button>
      <div className="breadcrumbs">
        <span>Workspace</span><ChevronRight size={14} />
        <strong>{titles[page][0]}</strong>
      </div>
      <div className="topbar-right">
        <div className="system-pill"><span className="status-dot" /> Local environment</div>
        <div className="top-avatar">XC</div>
      </div>
    </header>
  );
}

function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

function MetricCard({ item }) {
  const Icon = item.icon;
  return (
    <div className="metric-card">
      <div className="metric-top">
        <div className={`metric-icon ${item.color}`}><Icon size={19} /></div>
        <span className="metric-tag">TESTSET</span>
      </div>
      <div className="metric-value">{item.value}</div>
      <div className="metric-label">{item.label}</div>
      <div className="metric-sub">{item.sub}</div>
    </div>
  );
}

function Dashboard({ setPage, history }) {
  return (
    <div className="page-content">
      <SectionHeading
        eyebrow="PLATFORM OVERVIEW"
        title="Molecular intelligence, simplified."
        description="Explore drug–target affinity predictions and molecular insights."
        action={<button className="primary-button" onClick={() => setPage("predict")}><Sparkles size={16} /> New prediction</button>}
      />

      <div className="welcome-banner">
        <div className="banner-glow" />
        <div className="banner-content">
          <div className="banner-label"><span className="banner-pulse" /> MODEL READY FOR INFERENCE</div>
          <h2>Explore the molecular<br /><span>language of affinity.</span></h2>
          <p>Graph neural networks meet protein language models through bidirectional cross-attention.</p>
          <button className="banner-button" onClick={() => setPage("predict")}>Start a prediction <ArrowRight size={16} /></button>
        </div>
        <div className="molecule-art">
          <div className="molecule-orbit orbit-one" />
          <div className="molecule-orbit orbit-two" />
          <div className="molecule-core"><Atom size={60} /></div>
          <div className="molecule-node node-one" />
          <div className="molecule-node node-two" />
          <div className="molecule-node node-three" />
        </div>
      </div>

      <div className="section-title-row">
        <div><h3>Model performance</h3><p>Latest cleaned-v1 KIBA test-set evaluation</p></div>
        <span className="dataset-chip"><Database size={13} /> KIBA · TEST</span>
      </div>

      <div className="metrics-grid">{metrics.map(m => <MetricCard key={m.label} item={m} />)}</div>

      <div className="dashboard-grid">
        <div className="panel performance-panel">
          <div className="panel-heading"><div><h3>Evaluation comparison</h3><p>Previous vs latest checkpoint</p></div><span className="tiny-chip">MSE</span></div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={historyData} barSize={38}>
                <CartesianGrid strokeDasharray="3 3" stroke="#263044" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#8b95a9", fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 0.4]} tick={{ fill: "#8b95a9", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#151d2c", border: "1px solid #303b50", borderRadius: 10, color: "#fff" }} />
                <Bar dataKey="mse" fill="#8b70ef" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-footnote">Lower MSE indicates smaller prediction error.</div>
        </div>

        <div className="panel model-panel">
          <div className="panel-heading"><div><h3>Model components</h3><p>Current architecture</p></div><BrainCircuit size={19} className="accent-icon" /></div>
          <div className="component-list">
            <div className="component-row"><div className="component-symbol purple"><Network size={17} /></div><div><strong>GAT + GCN</strong><span>Drug graph encoder</span></div><CheckCircle2 size={16} className="check-icon" /></div>
            <div className="component-row"><div className="component-symbol cyan"><Dna size={17} /></div><div><strong>ESM-2</strong><span>Protein language model · 1280D</span></div><CheckCircle2 size={16} className="check-icon" /></div>
            <div className="component-row"><div className="component-symbol blue"><Zap size={17} /></div><div><strong>Cross-attention</strong><span>Bidirectional · 4 heads</span></div><CheckCircle2 size={16} className="check-icon" /></div>
            <div className="component-row"><div className="component-symbol orange"><ChartNoAxesCombined size={17} /></div><div><strong>Regression head</strong><span>Affinity score prediction</span></div><CheckCircle2 size={16} className="check-icon" /></div>
          </div>
          <button className="text-button" onClick={() => setPage("model")}>Explore architecture <ArrowUpRight size={15} /></button>
        </div>
      </div>

      <div className="panel" style={{ padding: 20, marginTop: 20 }}>
        <div className="panel-heading">
          <div><h3>Recent predictions</h3><p>Saved inference runs</p></div>
          <button className="text-button" onClick={() => setPage("history")}>View history <ArrowUpRight size={15} /></button>
        </div>
        <strong>{history.length}</strong> prediction{history.length === 1 ? "" : "s"} saved
      </div>

      <div className="disclaimer"><Info size={15} /> Metrics shown are from the recorded cleaned-v1 KIBA test evaluation. They are not live prediction results.</div>
    </div>
  );
}

/* =========================================================
   ATTENTION RESULTS AND IMPROVED HEATMAP
   ========================================================= */

function AttentionResults({ result }) {
  if (!result?.explainability?.top_residues) return null;

  const residues = result.explainability.top_residues;
  const heatmap = result.explainability.atom_residue_heatmap;
  const atoms = heatmap?.atoms || [];
  const heatmapResidues = heatmap?.residues || [];

  const allValues = atoms.flat().map(value => Math.max(0, Number(value) || 0));
  const maxValue = Math.max(...allValues, 1e-12);

  // Power-law contrast enhancement (γ = 0.7) improves visibility without changing
  // the actual attention values returned by the model.
  const cellOpacity = value => {
    const ratio = Math.max(0, Number(value) || 0) / maxValue;
    return 0.025 + 0.975 * Math.pow(ratio, 0.7);
  };

  return (
    <div className="panel" style={{ padding: 22, marginTop: 22 }}>
      <div className="panel-heading">
        <div>
          <h3>Attention-based explanation</h3>
          <p>Top protein residues receiving model attention</p>
        </div>
        <Microscope size={20} className="accent-icon" />
      </div>

      <div style={{ display: "grid", gap: 13, marginTop: 18 }}>
        {residues.map((r, index) => (
          <div key={`${r.position}-${r.amino_acid}`}>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 6,
              fontSize: 13
            }}>
              <strong>#{index + 1} · Residue {r.position} ({r.amino_acid})</strong>
              <span>{(r.attention * 100).toFixed(2)}%</span>
            </div>

            <div style={{
              height: 8,
              background: "#263044",
              borderRadius: 20,
              overflow: "hidden"
            }}>
              <div style={{
                height: "100%",
                width: `${Math.min(100, r.attention / (residues[0]?.attention || 1) * 100)}%`,
                background: "linear-gradient(90deg,#7661db,#35c4d8)",
                borderRadius: 20
              }} />
            </div>
          </div>
        ))}
      </div>

      {atoms.length > 0 && heatmapResidues.length > 0 && (
        <div style={{ marginTop: 30 }}>
          <div className="panel-heading">
            <div>
              <h3>Atom–residue attention heatmap</h3>
              <p>{atoms.length} drug atoms × {heatmapResidues.length} protein residues</p>
            </div>
            <Atom size={20} className="accent-icon" />
          </div>

          <div style={{
            marginTop: 16,
            padding: 16,
            borderRadius: 12,
            background: "#0b1220",
            border: "1px solid #263449"
          }}>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 10,
              flexWrap: "wrap",
              marginBottom: 14
            }}>
              <span style={{
                color: "#c7cce0",
                fontSize: 11,
                fontWeight: 700
              }}>
                DRUG ATOMS ↓ &nbsp; PROTEIN RESIDUES →
              </span>

              <span style={{ color: "#9aa5b8", fontSize: 11 }}>
                Power-law contrast enhancement (γ = 0.7)
              </span>
            </div>

            <div style={{
              width: "100%",
              overflowX: "auto",
              paddingBottom: 8
            }}>
              <div style={{
                minWidth: Math.max(850, 58 + heatmapResidues.length * 39),
                display: "grid",
                gap: 4
              }}>
                <div style={{
                  display: "grid",
                  gridTemplateColumns: `54px repeat(${heatmapResidues.length}, minmax(0, 1fr))`,
                  gap: 4,
                  alignItems: "end",
                  marginBottom: 5
                }}>
                  <div style={{
                    color: "#8290a8",
                    fontSize: 10,
                    textAlign: "center"
                  }}>
                    ATOM
                  </div>

                  {heatmapResidues.map((residue, index) => (
                    <div
                      key={`${residue.position}-${index}`}
                      style={{
                        color: "#c8d1e6",
                        fontSize: 10,
                        fontWeight: 700,
                        textAlign: "center",
                        lineHeight: 1.35,
                        paddingBottom: 5
                      }}
                    >
                      <div>{residue.amino_acid}</div>
                      <div style={{ color: "#8d9bb4" }}>{residue.position}</div>
                    </div>
                  ))}
                </div>

                {atoms.map((atom, atomIndex) => (
                  <div
                    key={atomIndex}
                    style={{
                      display: "grid",
                      gridTemplateColumns: `54px repeat(${heatmapResidues.length}, minmax(0, 1fr))`,
                      gap: 4,
                      alignItems: "center"
                    }}
                  >
                    <div style={{
                      color: "#aab6cc",
                      fontSize: 11,
                      fontWeight: 700,
                      textAlign: "center"
                    }}>
                      {atomIndex + 1}
                    </div>

                    {atom.map((value, residueIndex) => {
                      const numericValue = Math.max(0, Number(value) || 0);
                      const residue = heatmapResidues[residueIndex];

                      return (
                        <div
                          key={residueIndex}
                          title={`Atom ${atomIndex + 1} → ${residue.amino_acid}${residue.position}\nAttention: ${numericValue.toFixed(8)}`}
                          style={{
                            height: 28,
                            minWidth: 0,
                            borderRadius: 5,
                            background: `rgba(139,112,239,${cellOpacity(numericValue)})`,
                            border: "1px solid rgba(190,170,255,0.13)",
                            transition: "background 160ms ease"
                          }}
                        />
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            <div style={{
              borderTop: "1px solid #263449",
              marginTop: 18,
              paddingTop: 14
            }}>
              <div style={{
                color: "#aeb9cc",
                fontSize: 11,
                marginBottom: 10,
                fontWeight: 700
              }}>
                ATTENTION INTENSITY · POWER-LAW SCALE
              </div>

              <div style={{
                display: "flex",
                gap: 18,
                flexWrap: "wrap",
                alignItems: "center"
              }}>
                {[
                  { value: 0, label: "Zero" },
                  { value: maxValue * 0.1, label: "10% of maximum" },
                  { value: maxValue, label: "Maximum" }
                ].map(item => (
                  <div key={item.label} style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7
                  }}>
                    <div style={{
                      width: 17,
                      height: 17,
                      borderRadius: 4,
                      background: `rgba(139,112,239,${cellOpacity(item.value)})`,
                      border: "1px solid rgba(190,170,255,0.3)"
                    }} />
                    <span style={{ color: "#aeb9cc", fontSize: 10 }}>
                      {item.label}
                    </span>
                  </div>
                ))}
              </div>

              <p style={{
                color: "#8290a8",
                fontSize: 11,
                lineHeight: 1.6,
                marginTop: 13
              }}>
                Hover over any cell to inspect its exact attention weight.
                Power-law contrast enhancement reveals weaker values; the underlying
                attention values are unchanged.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="disclaimer" style={{ marginTop: 18 }}>
        <Info size={15} /> Attention weights describe model behaviour, not experimentally confirmed binding contacts.
      </div>
    </div>
  );
}

/* =========================================================
   AFFINITY PREDICTION
   ========================================================= */

function Prediction({ onPredictionComplete, lastPrediction, proteins, proteinLoading, drugs, drugLoading }) {
  const [smiles, setSmiles] = useState("");
  const [selectedDrug, setSelectedDrug] = useState("");
  const [sequence, setSequence] = useState("");
  const [selectedProtein, setSelectedProtein] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runPrediction(e) {
    e.preventDefault();
    setError("");
    onPredictionComplete(null);
    setLoading(true);

    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          smiles: smiles.trim(),
          protein_sequence: sequence.trim()
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed. Please check your inputs.");
      }

      onPredictionComplete(data);
    } catch (err) {
      setError(err.message || "Cannot connect to XCADTA API. Check that FastAPI is running.");
    } finally {
      setLoading(false);
    }
  }

  function downloadReport() {
    if (!lastPrediction) return;

    const report = {
      project: "XCADTA",
      report_generated_at: new Date().toISOString(),
      prediction: lastPrediction.prediction,
      drug: lastPrediction.drug,
      protein: lastPrediction.protein,
      explainability: lastPrediction.explainability,
      scientific_note:
        "Predicted KIBA score is not an experimentally measured affinity. Attention weights are not experimentally confirmed binding contacts."
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "XCADTA_Prediction_Report.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function selectProtein(id) {
    setSelectedProtein(id);
    const protein = proteins.find(p => p.id === id);
    if (protein) setSequence(protein.sequence);
  }

  return (
    <div className="page-content">
      <SectionHeading
        eyebrow="MODEL INFERENCE"
        title="Affinity prediction"
        description="Submit a small molecule and a cached protein sequence to estimate their binding affinity."
      />

      <div className="prediction-layout">
        <form className="panel input-panel" onSubmit={runPrediction}>
          <div className="form-section-title">
            <div className="step-number">01</div>
            <div><h3>Drug molecule</h3><p>Enter a valid SMILES representation</p></div>
          </div>

          <label className="field-label" htmlFor="drug-library">
            Available KIBA compounds ({drugLoading ? "loading..." : drugs.length})
          </label>

          <select
            id="drug-library"
            value={selectedDrug}
            onChange={e => {
              const value = e.target.value;
              setSelectedDrug(value);
              const drug = drugs.find(item => item.id === value);
              setSmiles(drug ? drug.smiles : "");
            }}
            style={{
              width: "100%",
              padding: 12,
              marginBottom: 14,
              background: "#111827",
              color: "#fff",
              border: "1px solid #303b50",
              borderRadius: 8
            }}
          >
            <option value="">Select a KIBA compound...</option>
            <option value="__custom__">✦ Enter custom SMILES</option>
            {drugs.map(drug => (
              <option key={drug.id} value={drug.id}>
                {drug.id} · {drug.atoms} atoms
              </option>
            ))}
          </select>

          <label className="field-label" htmlFor="smiles">
            {selectedDrug === "__custom__" ? "Custom SMILES string" : "Selected compound SMILES"}
          </label>

          <textarea
            id="smiles"
            className="code-input"
            value={smiles}
            onChange={e => setSmiles(e.target.value)}
            placeholder={selectedDrug === "__custom__" ? "Paste your valid molecular SMILES here..." : "Choose a KIBA compound above to fill its SMILES..."}
            rows={4}
            readOnly={selectedDrug !== "__custom__"}
            required
            style={selectedDrug !== "__custom__" ? { opacity: 0.9, cursor: "default" } : undefined}
          />

          <div className="field-hint">
            <Info size={13} /> Choose a compound from the KIBA dataset, or select Custom SMILES to enter your own molecule.
          </div>

          <MoleculeViewer smiles={smiles} />

          <div className="form-divider" />

          <div className="form-section-title">
            <div className="step-number">02</div>
            <div><h3>Target protein</h3><p>Choose a sequence from your local ESM-2 cache</p></div>
          </div>

          <label className="field-label" htmlFor="protein-library">
            Available proteins ({proteinLoading ? "loading..." : proteins.length})
          </label>

          <select
            id="protein-library"
            value={selectedProtein}
            onChange={e => selectProtein(e.target.value)}
            style={{
              width: "100%",
              padding: 12,
              marginBottom: 14,
              background: "#111827",
              color: "#fff",
              border: "1px solid #303b50",
              borderRadius: 8
            }}
          >
            <option value="">Select a cached protein...</option>
            {proteins.map(protein => (
              <option key={protein.id} value={protein.id}>
                {protein.label || protein.id}
              </option>
            ))}
          </select>

          <label className="field-label" htmlFor="sequence">Protein sequence</label>
          <textarea
            id="sequence"
            className="code-input sequence-input"
            value={sequence}
            onChange={e => {
              setSequence(e.target.value);
              setSelectedProtein("");
            }}
            placeholder="Select a cached protein or paste its amino acid sequence..."
            rows={6}
            required
          />

          <div className="field-hint">
            <Info size={13} /> Only sequences matching the local 1280-dimensional embedding cache can be predicted.
          </div>

          {error && (
            <div className="notice-box" style={{ marginTop: 15 }}>
              <Info size={17} />
              <div><strong>Prediction error</strong><span>{error}</span></div>
            </div>
          )}

          <button className="primary-button full-button" type="submit" disabled={loading || proteinLoading}>
            {loading
              ? <><Activity size={16} /> Running model...</>
              : <><Sparkles size={16} /> Run affinity prediction <ArrowRight size={16} /></>}
          </button>

          {loading && <p className="field-hint">Your molecular graph and protein representation are being processed.</p>}
        </form>

        <div className="prediction-aside">
          <div className="panel aside-card">
            <div className="aside-icon"><ShieldCheck size={21} /></div>
            <h3>Integrated inference workflow</h3>
            <p>Inputs are validated and processed through your trained XCADTA inference pipeline.</p>
            <div className="aside-divider" />
            <div className="aside-stat"><span>Drug encoder</span><strong>GAT + GCN</strong></div>
            <div className="aside-stat"><span>Protein encoder</span><strong>ESM-2</strong></div>
            <div className="aside-stat"><span>Fusion</span><strong>Cross-attention</strong></div>
            <div className="aside-stat"><span>Cached proteins</span><strong>{proteins.length || "—"}</strong></div>
          </div>

          <div className="tip-card">
            <Sparkles size={17} />
            <div><strong>Input tip</strong><p>FASTA headers are not required. Select a protein from the library to fill its sequence automatically.</p></div>
          </div>
        </div>
      </div>

      {lastPrediction && (
        <>
          <div className="panel" style={{ padding: 24, marginTop: 24 }}>
            <div className="panel-heading">
              <div><h3>Prediction completed</h3><p>Output from your trained XCADTA model</p></div>
              <CheckCircle2 size={22} className="check-icon" />
            </div>

            <div style={{ marginTop: 20 }}>
              <div className="eyebrow">PREDICTED KIBA AFFINITY SCORE</div>
              <div style={{ fontSize: 42, fontWeight: 750, margin: "8px 0" }}>
                {Number(lastPrediction.prediction.affinity).toFixed(4)}
              </div>
              <span className="dataset-chip">KIBA score</span>
            </div>

            <div className="metrics-grid" style={{ marginTop: 22 }}>
              <div className="metric-card"><div className="metric-label">Drug atoms</div><div className="metric-value">{lastPrediction.drug.atoms}</div></div>
              <div className="metric-card"><div className="metric-label">Protein residues</div><div className="metric-value">{lastPrediction.protein.sequence_length}</div></div>
              <div className="metric-card"><div className="metric-label">Embedding dimension</div><div className="metric-value">{lastPrediction.protein.embedding_dimension}</div></div>
            </div>

            <div className="field-hint" style={{ marginTop: 16 }}>
              <Info size={13} /> This is a model-predicted score, not an experimentally measured affinity.
            </div>

            <button className="primary-button" style={{ marginTop: 16 }} onClick={downloadReport}>
              <Download size={16} /> Download prediction report
            </button>
          </div>

          <AttentionResults result={lastPrediction} />
        </>
      )}
    </div>
  );
}

/* =========================================================
   FULL EXPLAINABILITY
   ========================================================= */

function Explainability({ lastPrediction }) {
  const [fullExplanation, setFullExplanation] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationNote, setRecommendationNote] = useState("");
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const [recommendationsError, setRecommendationsError] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setFullExplanation(null);
    setRecommendations([]);
    setRecommendationNote("");
    setRecommendationsError("");
    setRecommendationsLoading(false);
    setError("");
  }, [lastPrediction]);

  async function generateFullExplanation() {
    if (!lastPrediction?.drug?.smiles || !lastPrediction?.protein?.sequence) {
      setError("Run a fresh prediction first. The full explanation needs the original SMILES and protein sequence.");
      return;
    }

    setLoading(true);
    setError("");
    setFullExplanation(null);
    setRecommendations([]);
    setRecommendationsError("");

    try {
      const response = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          smiles: lastPrediction.drug.smiles,
          protein_sequence: lastPrediction.protein.sequence,
          max_residues: 20,
          permutations: 128,
          ig_steps: 64
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Explanation computation failed.");
      setFullExplanation(data.explanation);

      setRecommendationsLoading(true);
      try {
        const recommendationResponse = await fetch(`${API}/recommendations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            smiles: lastPrediction.drug.smiles,
            protein_sequence: lastPrediction.protein.sequence,
            limit: 5
          })
        });
        const recommendationData = await recommendationResponse.json();
        if (!recommendationResponse.ok) {
          throw new Error(recommendationData.detail || "Similar-compound search failed.");
        }
        setRecommendations(recommendationData.recommendations || []);
        setRecommendationNote(recommendationData.note || "");
      } catch (recommendationError) {
        setRecommendationsError(recommendationError.message || "Could not retrieve similar compounds.");
      } finally {
        setRecommendationsLoading(false);
      }
    } catch (err) {
      setError(err.message || "Could not generate the full explanation. Check that FastAPI is running.");
    } finally {
      setLoading(false);
    }
  }

  const explanation = fullExplanation;
  const proteinExplanation = explanation?.protein_residue_explanation;
  const atomExplanation = explanation?.drug_atom_explanation;
  const validation = proteinExplanation?.validation;
  const metrics = explanation?.metrics;
  const atomRows = atomExplanation?.ranked_atoms || [];
  const residueRows = proteinExplanation?.top_residues || [];
  const maxAtomImportance = Math.max(...atomRows.map(atom => Number(atom.importance) || 0), 1e-9);

  const displayMetric = value =>
    value === null || value === undefined || !Number.isFinite(Number(value))
      ? "N/A"
      : Number(value).toFixed(3);

  return (
    <div className="page-content">
      <SectionHeading
        eyebrow="MODEL INTERPRETATION"
        title="Explainability"
        description="Trace the prediction to drug atom features, protein residues and cross-modal attention."
        action={
          <button className="primary-button" onClick={generateFullExplanation} disabled={!lastPrediction || loading}>
            {loading
              ? <><Activity size={16} /> Computing explanations...</>
              : <><Microscope size={16} /> Generate full explanation</>}
          </button>
        }
      />

      {!lastPrediction ? (
        <div className="panel empty-state">
          <div className="empty-icon"><Microscope size={27} /></div>
          <h3>Start with an affinity prediction</h3>
          <p>Run a prediction first. Then generate atom Integrated Gradients, residue permutation-SHAP and faithfulness diagnostics for that exact drug–protein pair.</p>
        </div>
      ) : (
        <>
          <div className="notice-box explain-notice">
            <Info size={17} />
            <div>
              <strong>Explanation target · Predicted KIBA score {Number(lastPrediction.prediction.affinity).toFixed(4)}</strong>
              <span>{lastPrediction.drug.atoms} drug atoms · {lastPrediction.protein.sequence_length} protein residues. The detailed analysis runs separately so normal prediction remains fast.</span>
            </div>
          </div>

          {error && (
            <div className="notice-box explain-notice">
              <Info size={17} />
              <div><strong>Explanation could not be generated</strong><span>{error}</span></div>
            </div>
          )}

          {loading && (
            <div className="panel explanation-loading">
              <Activity size={22} />
              <div>
                <strong>Computing model explanations</strong>
              </div>
            </div>
          )}

          {explanation && (
            <>
              <div className="section-title-row explanation-section-title">
                <div><h3>Quantitative explanation diagnostics</h3><p>Computed for this prediction · not fixed benchmark values</p></div>
                <span className="dataset-chip">LIVE ANALYSIS</span>
              </div>

              <div className="metrics-grid explain-metrics">
                <div className="metric-card">
                  <div className="metric-top"><div className="metric-icon violet"><ChartNoAxesCombined size={19} /></div></div>
                  <div className="metric-value">{displayMetric(metrics?.shap_reconstruction_residual)}</div>
                  <div className="metric-label">SHAP reconstruction residual</div>
                  <div className="metric-sub">Path-sum consistency check; not an independent quality score</div>
                </div>

                <div className="metric-card">
                  <div className="metric-top"><div className="metric-icon cyan"><Activity size={19} /></div></div>
                  <div className="metric-value">{metrics?.deletion_lift_over_random == null ? "N/A" : `${displayMetric(metrics.deletion_lift_over_random)}×`}</div>
                  <div className="metric-label">Deletion lift vs random</div>
                  <div className="metric-sub">Above 1× means top-SHAP deletion changes output more</div>
                </div>

                <div className="metric-card">
                  <div className="metric-top"><div className="metric-icon blue"><ShieldCheck size={19} /></div></div>
                  <div className="metric-value">{displayMetric(metrics?.attribution_stability_spearman)}</div>
                  <div className="metric-label">Attribution stability</div>
                  <div className="metric-sub">Split-half rank correlation across 64 permutations per half</div>
                </div>

                <div className="metric-card">
                  <div className="metric-top"><div className="metric-icon violet"><Network size={19} /></div></div>
                  <div className="metric-value">{displayMetric(metrics?.attention_shap_spearman)}</div>
                  <div className="metric-label">Attention–SHAP agreement</div>
                  <div className="metric-sub">Spearman correlation of candidate rankings</div>
                </div>

                <div className="metric-card">
                  <div className="metric-top"><div className="metric-icon cyan"><Atom size={19} /></div></div>
                  <div className="metric-value">{displayMetric(metrics?.integrated_gradients_completeness_error)}</div>
                  <div className="metric-label">IG normalized completeness error</div>
                  <div className="metric-sub">Relative residual between attribution sum and prediction difference</div>
                </div>
              </div>

              <div className="panel explanation-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Protein residue contributions</h3>
                    <p>{proteinExplanation.method} · {proteinExplanation.permutations} permutations · ranked by absolute SHAP value</p>
                  </div>
                  <Dna size={19} className="accent-icon" />
                </div>

                <p className="explanation-method-note">{proteinExplanation.baseline}</p>

                <div className="explanation-table-wrap">
                  <table className="explanation-table">
                    <thead>
                      <tr>
                        <th>Rank</th><th>Residue</th><th>SHAP contribution</th><th>Attention</th><th>Effect on model output</th>
                      </tr>
                    </thead>
                    <tbody>
                      {residueRows.map((residue, index) => (
                        <tr key={residue.position}>
                          <td>{index + 1}</td>
                          <td><strong>{residue.amino_acid}{residue.position}</strong></td>
                          <td className={residue.shap_value >= 0 ? "contribution-positive" : "contribution-negative"}>
                            {Number(residue.shap_value).toFixed(5)}
                          </td>
                          <td>{Number(residue.attention).toFixed(5)}</td>
                          <td>{residue.direction}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="field-hint">
                  <Info size={13} /> Positive/negative signs describe changes relative to the stated embedding baseline, not experimentally measured binding energy.
                </div>
              </div>

              <div className="panel explanation-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Drug atom feature attribution</h3>
                    <p>{atomExplanation.method} · {atomExplanation.steps} integration steps</p>
                  </div>
                  <Atom size={20} className="accent-icon" />
                </div>

                <p className="explanation-method-note">{atomExplanation.interpretation}</p>

                <div className="atom-attribution-list">
                  {atomRows.slice(0, 12).map(atom => (
                    <div className="atom-attribution-row" key={atom.atom_index}>
                      <div className="atom-attribution-label">
                        <strong>Atom {atom.atom_index + 1} · {atom.element}{atom.is_aromatic ? " (aromatic)" : ""}</strong>
                        <span className={atom.contribution >= 0 ? "contribution-positive" : "contribution-negative"}>
                          {Number(atom.contribution).toFixed(5)}
                        </span>
                      </div>
                      <div className="atom-attribution-track">
                        <div className="atom-attribution-fill" style={{
                          width: `${Math.max(1, Number(atom.importance) / maxAtomImportance * 100)}%`
                        }} />
                      </div>
                      <div className="field-hint">
                        Importance {Number(atom.importance).toFixed(5)} · strongest feature group: {atom.strongest_feature_group}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="field-hint">
                  <Info size={13} /> Atom numbering follows RDKit molecular graph order. The zero-feature reference is computational, not a chemically valid molecule.
                </div>
              </div>

              <div className="panel explanation-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Deletion and sufficiency tests</h3>
                    <p>SHAP-ranked residue perturbations compared with random rankings</p>
                  </div>
                  <Activity size={19} className="accent-icon" />
                </div>

                <div className="explanation-table-wrap">
                  <table className="explanation-table">
                    <thead>
                      <tr>
                        <th>Candidate residues removed</th>
                        <th>Top-SHAP |Δ prediction|</th>
                        <th>Random |Δ prediction|</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(validation?.fractions_removed || []).map((fraction, index) => (
                        <tr key={fraction}>
                          <td>{Math.round(fraction * 100)}%</td>
                          <td>{Number(validation.top_shap_absolute_prediction_change[index]).toFixed(5)}</td>
                          <td>{Number(validation.random_absolute_prediction_change[index]).toFixed(5)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="explanation-sufficiency-grid">
                  {(validation?.sufficiency || []).map(item => (
                    <div className="sufficiency-item" key={item.kept_top_residues}>
                      <strong>Keep top {item.kept_top_residues}</strong>
                      <span>|Δ prediction| {Number(item.absolute_prediction_difference).toFixed(5)}</span>
                    </div>
                  ))}
                </div>

                <p className="explanation-method-note">{validation?.scope}</p>
              </div>

              <div className="panel explanation-panel recommendation-panel">
                <div className="panel-heading">
                  <div>
                    <h3>AI-driven drug discovery recommendations</h3>
                    <p>Pair-specific insights from atom attribution, residue SHAP and model-scored chemical neighbours</p>
                  </div>
                  <Sparkles size={20} className="accent-icon" />
                </div>

                <div className="recommendation-insights">
                  {atomRows.length > 0 && (
                    <div className="recommendation-insight">
                      <span className="recommendation-kicker">MOLECULAR FEATURE</span>
                      <p>Atom {atomRows[0].atom_index + 1} ({atomRows[0].element}) has the highest absolute Integrated Gradients importance in this molecule. Review its local chemical environment when comparing related structures.</p>
                    </div>
                  )}
                  {residueRows.length > 0 && (
                    <div className="recommendation-insight">
                      <span className="recommendation-kicker">PROTEIN FEATURE</span>
                      <p>Residue {residueRows[0].amino_acid}{residueRows[0].position} has the largest absolute SHAP contribution among the screened residues. Prioritise it for follow-up analysis, while treating the attribution as a model explanation rather than a confirmed binding contact.</p>
                    </div>
                  )}
                  <div className="recommendation-insight">
                    <span className="recommendation-kicker">NEXT STEP</span>
                    <p>Compare the shortlisted analogues below using both structural similarity and their XCADTA-predicted KIBA scores, then prioritise selected candidates for molecular docking and experimental validation.</p>
                  </div>
                </div>

                <div className="recommendation-compounds-heading">
                  <div><h4>Structurally similar KIBA compounds</h4><p>RDKit Morgan fingerprints · Tanimoto similarity · same target protein</p></div>
                  {recommendationsLoading && <span className="dataset-chip">SEARCHING & SCORING</span>}
                </div>

                {recommendationsLoading && (
                  <div className="recommendation-state"><Activity size={18} /> Finding similar compounds and running XCADTA predictions...</div>
                )}
                {recommendationsError && (
                  <div className="notice-box"><Info size={16} /><div><strong>Similar-compound search unavailable</strong><span>{recommendationsError}</span></div></div>
                )}
                {!recommendationsLoading && !recommendationsError && recommendations.length === 0 && (
                  <p className="explanation-method-note">No valid similar compounds were returned from the local KIBA library for this query.</p>
                )}
                {recommendations.length > 0 && (
                  <div className="recommendation-table-wrap">
                    <table className="explanation-table recommendation-table">
                      <thead><tr><th>Compound ID</th><th>Atoms</th><th>Tanimoto similarity</th><th>Predicted KIBA score</th></tr></thead>
                      <tbody>
                        {recommendations.map((compound, index) => (
                          <tr key={compound.id}>
                            <td><strong>{index + 1}. {compound.id}</strong></td>
                            <td>{compound.atoms}</td>
                            <td>{Number(compound.tanimoto_similarity).toFixed(3)}</td>
                            <td className="contribution-positive">{Number(compound.predicted_affinity).toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {recommendationNote && <p className="explanation-method-note">{recommendationNote}</p>}
              </div>

              <div className="panel explanation-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Methodology and interpretation</h3>
                    <p>What these measurements do—and do not—establish</p>
                  </div>
                  <Info size={19} className="accent-icon" />
                </div>

                <ul className="methodology-list">
                  {(explanation.methodological_notes || []).map((note, index) => (
                    <li key={index}>{note}</li>
                  ))}
                </ul>
              </div>
            </>
          )}

          <AttentionResults result={lastPrediction} />
        </>
      )}
    </div>
  );
}

/* =========================================================
   PREDICTION HISTORY
   ========================================================= */

function HistoryPage({ history, loading, error, refresh }) {
  return (
    <div className="page-content">
      <SectionHeading
        eyebrow="ACTIVITY"
        title="Prediction history"
        description="Review your saved XCADTA inference runs."
        action={<button className="primary-button" onClick={refresh}><Clock3 size={16} /> Refresh</button>}
      />

      {error && (
        <div className="notice-box">
          <Info size={17} />
          <div><strong>Could not load history</strong><span>{error}</span></div>
        </div>
      )}

      {loading ? (
        <div className="panel empty-state"><Activity size={26} /><h3>Loading predictions...</h3></div>
      ) : history.length === 0 ? (
        <div className="panel empty-state">
          <div className="empty-icon"><History size={28} /></div>
          <h3>No prediction history yet</h3>
          <p>Completed predictions will automatically appear here.</p>
        </div>
      ) : (
        <div className="panel" style={{ padding: 20, overflowX: "auto" }}>
          <div className="panel-heading">
            <div><h3>Saved predictions</h3><p>{history.length} most recent records</p></div>
          </div>

          <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: 13 }}>
            <thead>
              <tr>
                {["ID", "SMILES", "Protein length", "KIBA score", "Created at"].map(x => (
                  <th key={x} style={{ padding: 12, color: "#9aa5b8", borderBottom: "1px solid #303b50" }}>{x}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map(item => (
                <tr key={item.id}>
                  <td style={{ padding: 12 }}>#{item.id}</td>
                  <td style={{ padding: 12, maxWidth: 200, overflowWrap: "anywhere" }}>{item.smiles}</td>
                  <td style={{ padding: 12 }}>{item.protein_sequence.length}</td>
                  <td style={{ padding: 12, fontWeight: 700 }}>{Number(item.affinity).toFixed(4)}</td>
                  <td style={{ padding: 12 }}>{item.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* =========================================================
   MODEL ARCHITECTURE
   ========================================================= */

function Model() {
  const parts = [
    ["01", "Drug graph encoder", "GAT + GCN", "Learns molecular graph representations from atom-level features."],
    ["02", "Protein representation", "ESM-2", "Uses frozen ESM-2 t33 650M embeddings with 1280-dimensional representations."],
    ["03", "Cross-modal fusion", "Bidirectional attention", "Four-head cross-attention connects drug and protein representations."],
    ["04", "Prediction head", "Regression", "Mean and max pooled features feed the affinity regression head."],
  ];

  return (
    <div className="page-content">
      <SectionHeading
        eyebrow="MODEL DETAILS"
        title="Model architecture"
        description="A closer look at the components of Explainable Cross-Attention Drug-Target Affinity Prediction."
      />

      <div className="architecture-banner">
        <div className="architecture-symbol"><Atom size={34} /></div>
        <div>
          <div className="eyebrow">XCADTA FRAMEWORK</div>
          <h2>Cross-modal molecular learning</h2>
          <p>Graph-based drug encoding and pretrained protein representations are integrated through bidirectional cross-attention.</p>
        </div>
      </div>

      <div
        className="architecture-flow"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
          gap: 12,
          margin: "22px 0"
        }}
      >
        {[
          { number: "01", title: "Drug graph", detail: "Molecular structure and atom-level features", model: "GAT + GCN", color: "#9b82f5" },
          { number: "02", title: "Protein sequence", detail: "Pretrained protein language representation", model: "ESM-2 · 1280D", color: "#35c4d8" },
          { number: "03", title: "Cross-attention", detail: "Bidirectional interaction between both modalities", model: "4 attention heads", color: "#6da8ff" },
          { number: "04", title: "Affinity output", detail: "Fused features produce the predicted KIBA score", model: "Regression head", color: "#f0a45d" }
        ].map((stage, index) => (
          <div key={stage.number} style={{ minHeight: 142, padding: "17px 16px", borderRadius: 14, background: "linear-gradient(145deg, rgba(25,35,54,0.96), rgba(16,24,39,0.96))", border: "1px solid #303b50", overflow: "hidden" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 15 }}>
              <span style={{ color: stage.color, fontSize: 11, fontWeight: 800, letterSpacing: "0.12em" }}>STEP {stage.number}</span>
              {index < 3 && <ArrowRight size={16} style={{ color: "#8290a8" }} />}
            </div>
            <h3 style={{ margin: "0 0 7px", fontSize: 16 }}>{stage.title}</h3>
            <p style={{ margin: "0 0 13px", color: "#9aa5b8", fontSize: 12, lineHeight: 1.55 }}>{stage.detail}</p>
            <span style={{ display: "inline-block", padding: "5px 9px", borderRadius: 20, color: stage.color, background: `${stage.color}18`, fontSize: 11, fontWeight: 700 }}>{stage.model}</span>
          </div>
        ))}
      </div>

      <div className="architecture-list">
        {parts.map(p => (
          <div className="panel architecture-item" key={p[0]}>
            <div className="architecture-number">{p[0]}</div>
            <div className="architecture-text">
              <h3>{p[1]}</h3>
              <span>{p[2]}</span>
              <p>{p[3]}</p>
            </div>
            <CheckCircle2 size={18} className="check-icon" />
          </div>
        ))}
      </div>

      <div className="disclaimer">
        <Info size={15} /> Architecture summary reflects the current XCADTA implementation configuration.
      </div>
    </div>
  );
}

/* =========================================================
   MAIN APPLICATION
   ========================================================= */

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [proteins, setProteins] = useState([]);
  const [proteinLoading, setProteinLoading] = useState(true);
  const [drugs, setDrugs] = useState([]);
  const [drugLoading, setDrugLoading] = useState(true);

  async function refreshHistory() {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(`${API}/history`);
      if (!response.ok) throw new Error("History service returned an error.");
      const data = await response.json();
      setHistory(data.predictions || []);
    } catch (error) {
      setHistoryError(error.message || "Cannot connect to the API.");
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    refreshHistory();

    async function loadProteins() {
      try {
        const response = await fetch(`${API}/proteins`);
        if (!response.ok) throw new Error("Could not load protein library.");
        const data = await response.json();
        setProteins(data.proteins || []);
      } catch (error) {
        console.error("Protein library error:", error);
      } finally {
        setProteinLoading(false);
      }
    }

    loadProteins();

    async function loadDrugs() {
      try {
        const response = await fetch(`${API}/drugs`);
        if (!response.ok) throw new Error("Could not load KIBA drug library.");
        const data = await response.json();
        setDrugs(data.drugs || []);
      } catch (error) {
        console.error("KIBA drug library error:", error);
      } finally {
        setDrugLoading(false);
      }
    }

    loadDrugs();
  }, []);

  function handlePrediction(data) {
    if (data) {
      setLastPrediction(data);
      refreshHistory();
    } else {
      setLastPrediction(null);
    }
  }

  const pages = {
    dashboard: <Dashboard setPage={setPage} history={history} />,
    predict: (
      <Prediction
        onPredictionComplete={handlePrediction}
        lastPrediction={lastPrediction}
        proteins={proteins}
        proteinLoading={proteinLoading}
        drugs={drugs}
        drugLoading={drugLoading}
      />
    ),
    explain: <Explainability lastPrediction={lastPrediction} />,
    history: <HistoryPage history={history} loading={historyLoading} error={historyError} refresh={refreshHistory} />,
    model: <Model />,
  };

  return (
    <div className="app-shell">
      <Sidebar page={page} setPage={setPage} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen} />
      <main className="main-area">
        <Header page={page} setMobileOpen={setMobileOpen} />
        {pages[page]}
        <footer className="app-footer">
          <span>© 2026 XCADTA Platform</span>
          <span><span className="footer-dot" /> AI-powered affinity predictions · Local instance</span>
        </footer>
      </main>
    </div>
  );
}
