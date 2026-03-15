import { useState, useEffect, useRef, useCallback } from "react";

// ─── Palette & Config ───────────────────────────────────────────────────────
const SHAPES = ["circle", "square", "triangle", "star", "hexagon", "diamond"];
const PALETTE = {
  circle:   { fill: "#60efff", stroke: "#00b4d8", label: "Circle"   },
  square:   { fill: "#f9c74f", stroke: "#f3722c", label: "Square"   },
  triangle: { fill: "#90be6d", stroke: "#43aa8b", label: "Triangle" },
  star:     { fill: "#f94144", stroke: "#c1121f", label: "Star"     },
  hexagon:  { fill: "#c77dff", stroke: "#7b2d8b", label: "Hexagon"  },
  diamond:  { fill: "#ff9f1c", stroke: "#e76f51", label: "Diamond"  },
};

// ─── Math helpers ────────────────────────────────────────────────────────────
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const rand = (lo, hi) => Math.random() * (hi - lo) + lo;

// ─── Noise-based "fake GAN" generator ────────────────────────────────────────
function generateSample(shape, epoch, totalEpochs) {
  const progress = epoch / totalEpochs;           // 0 → 1
  const noise    = 1 - clamp(progress, 0, 1);     // noise fades with training
  return { shape, noise, progress, epoch };
}

// ─── Draw a shape on a canvas ────────────────────────────────────────────────
function drawShape(ctx, shape, noise, size) {
  const w = size, h = size, cx = w / 2, cy = h / 2;
  const { fill, stroke } = PALETTE[shape] || PALETTE.circle;
  const r = (w * 0.35) * lerp(1, rand(0.4, 1.6), noise * 0.6);

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = "#0d0d1a";
  ctx.fillRect(0, 0, w, h);

  // Noise overlay
  if (noise > 0.05) {
    const imgData = ctx.createImageData(w, h);
    for (let i = 0; i < imgData.data.length; i += 4) {
      const v = Math.random() * 255 * noise * 0.4;
      imgData.data[i]     = v;
      imgData.data[i + 1] = v;
      imgData.data[i + 2] = v;
      imgData.data[i + 3] = Math.random() * 180 * noise;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // Wobble due to noise
  const wobble = () => rand(-noise * 12, noise * 12);

  ctx.save();
  ctx.translate(cx + wobble(), cy + wobble());
  ctx.rotate(rand(-noise * 0.3, noise * 0.3));

  ctx.shadowColor   = fill;
  ctx.shadowBlur    = lerp(4, 20, 1 - noise);
  ctx.strokeStyle   = stroke;
  ctx.lineWidth     = 2.5;
  ctx.fillStyle     = fill + Math.floor(lerp(100, 220, 1 - noise)).toString(16).padStart(2, "0");

  ctx.beginPath();
  switch (shape) {
    case "circle":
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      break;
    case "square": {
      const s = r * 1.35;
      ctx.rect(-s / 2 + wobble(), -s / 2 + wobble(), s + wobble(), s + wobble());
      break;
    }
    case "triangle": {
      const pts = [
        [0, -r + wobble()],
        [ r * 0.866 + wobble(),  r * 0.5 + wobble()],
        [-r * 0.866 + wobble(),  r * 0.5 + wobble()],
      ];
      ctx.moveTo(...pts[0]);
      pts.slice(1).forEach(p => ctx.lineTo(...p));
      ctx.closePath();
      break;
    }
    case "star": {
      const spikes = 5, inner = r * 0.45;
      for (let i = 0; i < spikes * 2; i++) {
        const a = (Math.PI / spikes) * i - Math.PI / 2;
        const rr = i % 2 === 0 ? r : inner;
        const px = Math.cos(a) * rr + wobble() * 0.5;
        const py = Math.sin(a) * rr + wobble() * 0.5;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      break;
    }
    case "hexagon": {
      for (let i = 0; i < 6; i++) {
        const a  = (Math.PI / 3) * i - Math.PI / 6;
        const px = Math.cos(a) * (r + wobble() * 0.3);
        const py = Math.sin(a) * (r + wobble() * 0.3);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      break;
    }
    case "diamond": {
      const pts = [
        [0, -(r * 1.2) + wobble()],
        [ r * 0.7 + wobble(), 0],
        [0,  (r * 1.2) + wobble()],
        [-r * 0.7 + wobble(), 0],
      ];
      ctx.moveTo(...pts[0]);
      pts.slice(1).forEach(p => ctx.lineTo(...p));
      ctx.closePath();
      break;
    }
    default:
      ctx.arc(0, 0, r, 0, Math.PI * 2);
  }
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

// ─── Mini Loss Chart ──────────────────────────────────────────────────────────
function LossChart({ gLoss, dLoss }) {
  const max = Math.max(...gLoss, ...dLoss, 1);
  const W = 260, H = 80, PAD = 6;
  const pts = (arr) =>
    arr.map((v, i) => {
      const x = PAD + (i / (arr.length - 1 || 1)) * (W - PAD * 2);
      const y = H - PAD - (v / max) * (H - PAD * 2);
      return `${x},${y}`;
    }).join(" ");

  return (
    <svg width={W} height={H} style={{ display: "block" }}>
      <rect width={W} height={H} rx="6" fill="#0a0a18" />
      {gLoss.length > 1 && (
        <polyline points={pts(gLoss)} fill="none" stroke="#60efff" strokeWidth="1.5" strokeLinejoin="round" />
      )}
      {dLoss.length > 1 && (
        <polyline points={pts(dLoss)} fill="none" stroke="#f94144" strokeWidth="1.5" strokeLinejoin="round" />
      )}
      <text x={PAD + 2} y={H - PAD - 2} fill="#60efff" fontSize="9" fontFamily="monospace">G</text>
      <text x={PAD + 14} y={H - PAD - 2} fill="#f94144" fontSize="9" fontFamily="monospace">D</text>
    </svg>
  );
}

// ─── Single Generated Image Card ─────────────────────────────────────────────
function GeneratedCard({ sample, size = 120 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    drawShape(ctx, sample.shape, sample.noise, size);
  }, [sample, size]);

  const pct = Math.round(sample.progress * 100);
  const { label, stroke } = PALETTE[sample.shape];

  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
      background: "#13132a", borderRadius: 10, padding: "10px 12px",
      border: `1px solid ${stroke}44`,
    }}>
      <canvas ref={canvasRef} width={size} height={size} style={{ borderRadius: 6 }} />
      <div style={{ fontFamily: "monospace", fontSize: 11, color: stroke, letterSpacing: 1 }}>
        {label.toUpperCase()}
      </div>
      <div style={{ width: "100%", background: "#0d0d1a", borderRadius: 4, height: 4 }}>
        <div style={{
          height: "100%", borderRadius: 4, width: `${pct}%`,
          background: `linear-gradient(90deg, ${stroke}, ${PALETTE[sample.shape].fill})`,
          transition: "width 0.3s ease",
        }} />
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#555" }}>
        epoch {sample.epoch} · noise {sample.noise.toFixed(2)}
      </div>
    </div>
  );
}

// ─── Architecture Diagram ─────────────────────────────────────────────────────
function ArchDiagram({ activeShape }) {
  const color = activeShape ? PALETTE[activeShape].fill : "#60efff";
  const boxes = [
    { label: "Noise z",    sub: "100-dim",   x: 0   },
    { label: "Label y",    sub: activeShape || "e.g. circle", x: 180 },
    { label: "Generator",  sub: "G(z, y)",   x: 90, y: 70, wide: true },
    { label: "Fake Image", sub: "generated", x: 90, y: 150 },
    { label: "Real Image", sub: "dataset",   x: 0,  y: 230 },
    { label: "Discriminator", sub: "D(x, y)", x: 90, y: 230, wide: true },
    { label: "Real / Fake",   sub: "output",  x: 90, y: 310 },
  ];

  return (
    <svg width={280} height={350} style={{ display: "block" }}>
      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#444" />
        </marker>
      </defs>
      {/* Arrows */}
      {[
        [40,22,130,62],[220,22,175,62],
        [150,102,150,142],
        [40,242,130,242],[150,262,150,302],
        [150,172,150,222],
      ].map(([x1,y1,x2,y2],i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="#333" strokeWidth="1.5" markerEnd="url(#arr)" />
      ))}
      {boxes.map(({ label, sub, x = 0, y = 0, wide }) => {
        const w = wide ? 100 : 80, h = 40;
        const bx = x + (wide ? 40 : 50), by = y;
        const isG = label === "Generator", isD = label === "Discriminator";
        const fill = isG ? color + "22" : isD ? "#f9414422" : "#13132a";
        const border = isG ? color : isD ? "#f94144" : "#333";
        return (
          <g key={label} transform={`translate(${bx - w/2},${by})`}>
            <rect width={w} height={h} rx="6" fill={fill} stroke={border} strokeWidth={isG || isD ? 1.5 : 1} />
            <text x={w/2} y={16} textAnchor="middle" fill="#ddd" fontSize="10" fontFamily="monospace" fontWeight="600">{label}</text>
            <text x={w/2} y={30} textAnchor="middle" fill="#666" fontSize="9" fontFamily="monospace">{sub}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function CGANDemo() {
  const [selectedShape, setSelectedShape] = useState(null);
  const [inputText, setInputText]         = useState("");
  const [training, setTraining]           = useState(false);
  const [epoch, setEpoch]                 = useState(0);
  const [samples, setSamples]             = useState([]);
  const [gLoss, setGLoss]                 = useState([]);
  const [dLoss, setDLoss]                 = useState([]);
  const [tab, setTab]                     = useState("generate");  // generate | arch | howto
  const TOTAL_EPOCHS = 30;
  const timerRef = useRef(null);

  const recognizeShape = (text) => {
    const t = text.toLowerCase().trim();
    return SHAPES.find(s => t.includes(s)) || null;
  };

  const startTraining = useCallback((shape) => {
    if (!shape) return;
    setTraining(true);
    setEpoch(0);
    setGLoss([]);
    setDLoss([]);
    setSamples([]);

    let e = 0;
    timerRef.current = setInterval(() => {
      e++;
      const progress  = e / TOTAL_EPOCHS;
      const gL = Math.max(0.12, 1.2 - progress * 0.9 + rand(-0.1, 0.1));
      const dL = Math.max(0.08, 0.9 - progress * 0.6 + rand(-0.08, 0.08));
      setEpoch(e);
      setGLoss(prev => [...prev.slice(-60), +gL.toFixed(3)]);
      setDLoss(prev => [...prev.slice(-60), +dL.toFixed(3)]);
      // Add a new sample every 5 epochs
      if (e % 5 === 0 || e === 1) {
        setSamples(prev => [
          generateSample(shape, e, TOTAL_EPOCHS),
          ...prev.slice(0, 7),
        ]);
      }
      if (e >= TOTAL_EPOCHS) {
        clearInterval(timerRef.current);
        setTraining(false);
      }
    }, 160);
  }, []);

  useEffect(() => () => clearInterval(timerRef.current), []);

  const handleGenerate = () => {
    const shape = recognizeShape(inputText) || selectedShape;
    if (!shape) return;
    setSelectedShape(shape);
    startTraining(shape);
  };

  const handleQuick = (shape) => {
    setSelectedShape(shape);
    setInputText(shape);
    startTraining(shape);
  };

  const progress = epoch / TOTAL_EPOCHS;
  const activeColor = selectedShape ? PALETTE[selectedShape].fill : "#60efff";

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #06060f 0%, #0d0d1f 60%, #0a0a16 100%)",
      color: "#e0e0ff",
      fontFamily: "'Courier New', monospace",
      padding: "0 0 40px",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: "1px solid #1e1e3a",
        padding: "18px 28px 14px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "#08081699",
      }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: 2, color: "#60efff" }}>
            CGAN STUDIO
          </div>
          <div style={{ fontSize: 10, color: "#444", letterSpacing: 3, marginTop: 2 }}>
            CONDITIONAL GENERATIVE ADVERSARIAL NETWORK
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["generate","arch","howto"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: tab === t ? activeColor + "22" : "transparent",
              border: `1px solid ${tab === t ? activeColor : "#2a2a4a"}`,
              color: tab === t ? activeColor : "#666",
              borderRadius: 6, padding: "5px 14px", cursor: "pointer",
              fontSize: 11, letterSpacing: 1, fontFamily: "monospace",
              transition: "all 0.2s",
            }}>
              {t.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "28px 20px" }}>

        {/* ── GENERATE TAB ── */}
        {tab === "generate" && (
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>

            {/* Left panel */}
            <div style={{ flex: "0 0 280px" }}>
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 11, color: "#555", letterSpacing: 2, marginBottom: 8 }}>
                  ENTER LABEL
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    value={inputText}
                    onChange={e => setInputText(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && handleGenerate()}
                    placeholder="e.g. circle, star…"
                    style={{
                      flex: 1, background: "#0d0d1a", border: "1px solid #2a2a4a",
                      color: "#e0e0ff", borderRadius: 8, padding: "8px 12px",
                      fontFamily: "monospace", fontSize: 13, outline: "none",
                    }}
                  />
                  <button onClick={handleGenerate} disabled={training} style={{
                    background: training ? "#1a1a2e" : `linear-gradient(135deg, ${activeColor}44, ${activeColor}22)`,
                    border: `1px solid ${training ? "#333" : activeColor}`,
                    color: training ? "#444" : activeColor,
                    borderRadius: 8, padding: "8px 16px", cursor: training ? "not-allowed" : "pointer",
                    fontSize: 12, fontFamily: "monospace", letterSpacing: 1, transition: "all 0.2s",
                  }}>
                    {training ? "…" : "RUN"}
                  </button>
                </div>
              </div>

              {/* Quick-pick buttons */}
              <div style={{ fontSize: 11, color: "#555", letterSpacing: 2, marginBottom: 10 }}>
                QUICK SELECT
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 }}>
                {SHAPES.map(s => {
                  const { fill, stroke, label } = PALETTE[s];
                  const active = selectedShape === s;
                  return (
                    <button key={s} onClick={() => handleQuick(s)} disabled={training} style={{
                      background: active ? fill + "22" : "#0d0d1a",
                      border: `1px solid ${active ? fill : "#2a2a4a"}`,
                      color: active ? fill : "#777",
                      borderRadius: 20, padding: "5px 14px",
                      cursor: training ? "not-allowed" : "pointer",
                      fontSize: 11, fontFamily: "monospace", letterSpacing: 1,
                      transition: "all 0.2s",
                    }}>{label}</button>
                  );
                })}
              </div>

              {/* Training Status */}
              <div style={{
                background: "#0d0d1a", border: "1px solid #1e1e3a",
                borderRadius: 10, padding: "14px 16px", marginBottom: 18,
              }}>
                <div style={{ fontSize: 10, color: "#555", letterSpacing: 2, marginBottom: 10 }}>
                  TRAINING STATUS
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <span style={{ fontSize: 11, color: "#666" }}>Epoch</span>
                  <span style={{ fontSize: 11, color: activeColor }}>{epoch}/{TOTAL_EPOCHS}</span>
                </div>
                <div style={{ background: "#080810", borderRadius: 4, height: 5, marginBottom: 12 }}>
                  <div style={{
                    height: "100%", borderRadius: 4, width: `${progress * 100}%`,
                    background: `linear-gradient(90deg, #60efff, ${activeColor})`,
                    transition: "width 0.2s",
                    boxShadow: `0 0 8px ${activeColor}`,
                  }} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, fontSize: 11 }}>
                  <span style={{ color: "#60efff" }}>G Loss: {gLoss[gLoss.length-1]?.toFixed(3) ?? "—"}</span>
                  <span style={{ color: "#f94144" }}>D Loss: {dLoss[dLoss.length-1]?.toFixed(3) ?? "—"}</span>
                </div>
                <LossChart gLoss={gLoss} dLoss={dLoss} />
              </div>

              {/* Conditional input indicator */}
              {selectedShape && (
                <div style={{
                  background: PALETTE[selectedShape].fill + "11",
                  border: `1px solid ${PALETTE[selectedShape].fill}44`,
                  borderRadius: 8, padding: "10px 14px",
                  fontSize: 11, color: PALETTE[selectedShape].fill,
                  letterSpacing: 1,
                }}>
                  ▸ CONDITION: <strong>{selectedShape.toUpperCase()}</strong>
                  <div style={{ color: "#555", fontSize: 10, marginTop: 4 }}>
                    One-hot encoded → concatenated with noise z
                  </div>
                </div>
              )}
            </div>

            {/* Right: generated images */}
            <div style={{ flex: 1, minWidth: 280 }}>
              <div style={{ fontSize: 11, color: "#555", letterSpacing: 2, marginBottom: 14 }}>
                GENERATED OUTPUTS (newest first)
              </div>
              {samples.length === 0 ? (
                <div style={{
                  height: 200, display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#2a2a4a", fontSize: 13, border: "1px dashed #1e1e3a", borderRadius: 12,
                  letterSpacing: 2,
                }}>
                  SELECT A SHAPE TO BEGIN
                </div>
              ) : (
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
                  gap: 12,
                }}>
                  {samples.map((s, i) => (
                    <GeneratedCard key={i} sample={s} size={110} />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── ARCHITECTURE TAB ── */}
        {tab === "arch" && (
          <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 11, color: "#555", letterSpacing: 2, marginBottom: 16 }}>
                CGAN ARCHITECTURE
              </div>
              <ArchDiagram activeShape={selectedShape} />
            </div>
            <div style={{ flex: 1, minWidth: 240 }}>
              <div style={{ fontSize: 11, color: "#555", letterSpacing: 2, marginBottom: 16 }}>
                HOW IT WORKS
              </div>
              {[
                ["Noise Vector z", "A random 100-dim Gaussian vector — the Generator's creative seed. Every unique z produces a different image."],
                ["Label y (Condition)", "A one-hot encoded class vector (e.g. [0,1,0,0,0,0] for 'square'). Concatenated with z before feeding the Generator."],
                ["Generator G(z,y)", "A neural net that maps (z,y) → a fake image. It learns to produce images matching the requested label to fool D."],
                ["Discriminator D(x,y)", "Given an image x and a label y, it outputs P(real). It sees both real dataset images and Generator fakes — its job is to tell them apart."],
                ["Adversarial Training", "G tries to maximise D's mistakes; D tries to minimise them. The minimax game drives both networks to improve simultaneously."],
                ["Convergence", "At equilibrium, G produces realistic, label-conditioned images that D can no longer distinguish from real ones."],
              ].map(([title, body]) => (
                <div key={title} style={{
                  background: "#0d0d1a", border: "1px solid #1e1e3a",
                  borderRadius: 8, padding: "12px 16px", marginBottom: 10,
                }}>
                  <div style={{ fontSize: 12, color: activeColor, marginBottom: 4, fontWeight: 700 }}>{title}</div>
                  <div style={{ fontSize: 11, color: "#888", lineHeight: 1.7 }}>{body}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── HOW-TO TAB ── */}
        {tab === "howto" && (
          <div style={{ maxWidth: 620 }}>
            <div style={{ fontSize: 11, color: "#555", letterSpacing: 2, marginBottom: 18 }}>
              KEY CONCEPTS
            </div>
            {[
              {
                tag: "01", title: "Vanilla GAN vs CGAN",
                body: "In a vanilla GAN, the Generator has no control over what it produces. A CGAN adds a condition y (class label, text, attribute) to both G and D, giving the model steerable control over the output."
              },
              {
                tag: "02", title: "One-Hot Encoding",
                body: `With 6 shapes, circle → [1,0,0,0,0,0], square → [0,1,0,0,0,0], etc. This vector is concatenated with the noise z for G, and with the image pixels for D.`
              },
              {
                tag: "03", title: "Objective Function",
                body: "min_G max_D  E[log D(x,y)] + E[log(1 − D(G(z,y),y))]. D maximises; G minimises. The label y forces both networks to be class-aware."
              },
              {
                tag: "04", title: "Noise Schedule",
                body: "Early training: high noise → blurry, distorted outputs. As G learns the conditional distribution, noise in the outputs decreases and shapes become crisp — exactly what you see in this demo."
              },
              {
                tag: "05", title: "Loss Dynamics",
                body: "Generator loss (blue) should decrease as it gets better at fooling D. Discriminator loss (red) often stabilises near 0.5 log 2 ≈ 0.69 at equilibrium — it can no longer tell real from fake."
              },
              {
                tag: "06", title: "Real-World CGANs",
                body: "Pix2Pix (image-to-image translation), StackGAN (text→image), ACGAN (auxiliary classifier GAN), and StyleGAN2-ADA all use conditioning mechanisms descended from this core idea."
              },
            ].map(({ tag, title, body }) => (
              <div key={tag} style={{
                display: "flex", gap: 16, marginBottom: 16,
                background: "#0d0d1a", border: "1px solid #1e1e3a",
                borderRadius: 10, padding: "14px 18px",
              }}>
                <div style={{ fontSize: 22, color: "#1e1e3a", fontWeight: 900, minWidth: 28 }}>{tag}</div>
                <div>
                  <div style={{ fontSize: 12, color: activeColor, fontWeight: 700, marginBottom: 5 }}>{title}</div>
                  <div style={{ fontSize: 11, color: "#888", lineHeight: 1.8 }}>{body}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
