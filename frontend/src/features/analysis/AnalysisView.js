import React, { useState, useRef, useEffect } from 'react';
import { Upload, Layers, Activity, AlertTriangle, Scan, FileImage, Droplet, Info, Aperture, Crosshair, FileText, Download } from 'lucide-react';
import styles from './AnalysisView.module.css';

// --- SUB-COMPONENT: Cybernetic Loader ---
const ScanningLoader = () => {
  const steps = [
    "INITIALIZING NEURAL TENSORS...", "INJECTING GAUSSIAN NOISE (TTA)...",
    "RUNNING BAYESIAN INFERENCE...", "CALCULATING EPISTEMIC UNCERTAINTY...",
    "FILTERING ORGANIC PARTICULATES...", "SYNTHESIZING VIRTUAL STAIN..."
  ];
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => setIndex(p => (p + 1) % steps.length), 1200);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className={styles.loaderContainer}>
      <Activity className="spin-slow" size={48} color="var(--color-primary)" />
      <div className={styles.loaderText}>
        <span style={{color:'var(--color-primary)'}}>[ SYSTEM BUSY ]</span>
        <span style={{fontFamily:'var(--font-mono)', color:'var(--color-text-muted)'}}>{steps[index]}</span>
      </div>
      <div className={styles.progressBar}><div className={styles.progressFill} style={{width:`${(index+1)*(100/steps.length)}%`}}/></div>
    </div>
  );
};

// --- SUB-COMPONENT: Standby HUD ---
const StandbyScreen = () => (
  <div className={styles.idleContainer}>
    <div className={styles.gridBackground} />
    <div className={styles.hudRing}>
      <Aperture size={64} color="var(--color-primary)" strokeWidth={1} />
    </div>
    <div className={styles.systemStatus}>
      <span className={styles.statusLine}>:: NEURAL ENGINE ONLINE ::</span>
      <span className={styles.statusLine}>:: SENSORS CALIBRATED ::</span>
      <h2 className={styles.instructionMain}>AWAITING SPECIMEN</h2>
      <div style={{marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', justifyContent: 'center', opacity: 0.5}}>
        <Crosshair size={14} />
        <span style={{fontSize: '0.7rem'}}>OPTICAL FEED DISCONNECTED</span>
      </div>
    </div>
  </div>
);

// --- CONFIG: Layers ---
const LAYER_CONFIG = {
  original: { label: 'Raw Optical', icon: FileImage, desc: "No overlay. Raw brightfield sensor input." },
  mask: { label: 'Segmentation Mask', icon: Scan, desc: "RED: Fragment (Hard Plastic) | YELLOW: Fiber (Textile)" },
  stain: { label: 'Virtual Stain', icon: Droplet, desc: "ORANGE: Synthetic | PURPLE: Organic" },
  heatmap: { label: 'xAI Uncertainty', icon: Activity, desc: "RED GLOW: Low Confidence Area" }
};

export const AnalysisView = () => {
  const [imageState, setImageState] = useState({ original: null, mask: null, heatmap: null, stain: null });
  const [metrics, setMetrics] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeLayer, setActiveLayer] = useState('original');
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [error, setError] = useState(null);
  
  // [CHANGE 2] Capture Request ID State
  const [currentRequestId, setCurrentRequestId] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsProcessing(true);
    setError(null);
    setMetrics(null);
    setHoveredInfo(null);
    
    const localUrl = URL.createObjectURL(file);
    setImageState({ original: localUrl, mask: null, heatmap: null, stain: null });

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Ensure this matches your backend URL (127.0.0.1 is safer than localhost on Windows)
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('Analysis Failed');
      const data = await response.json();
      
      // [CHANGE 2] Set the ID
      setCurrentRequestId(data.request_id);

      const baseUrl = `http://127.0.0.1:8000/results/${data.request_id}`;
      
      setImageState({
        original: localUrl,
        mask: `${baseUrl}/mask`,
        heatmap: `${baseUrl}/heatmap`,
        stain: `${baseUrl}/stain`, 
      });

      setMetrics({
        count: data.particle_count,
        distribution: data.distribution
      });
      
      setActiveLayer('mask');

    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className={styles.container}>
      {/* SIDEBAR */}
      <aside className={styles.sidebar}>
        <div>
          <h1 className={styles.header} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Scan size={18} /> AquaEye
          </h1>
          <div className={styles.uploadBox} onClick={() => !isProcessing && fileInputRef.current.click()}>
            <input type="file" ref={fileInputRef} hidden onChange={handleFileUpload} accept="image/*" />
            {isProcessing ? <ScanningLoader /> : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <Upload size={32} color="var(--color-text-muted)" />
                <p style={{fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--color-text-muted)'}}>LOAD SPECIMEN</p>
              </div>
            )}
          </div>
          {error && <div className={styles.errorBox}><AlertTriangle size={16}/> {error}</div>}
        </div>

        {metrics && (
          <div className="animate-fade-in">
            <h2 className={styles.header}><Activity size={16} /> Metrics</h2>
            <div className={styles.statsGrid}>
              <div className={styles.statCard}>
                <span className={styles.statValue}>{metrics.count}</span>
                <span className={styles.statLabel}>Particles</span>
              </div>
              <div className={styles.statCard} style={{borderColor: 'var(--color-accent)'}}>
                <span className={styles.statValue}>{metrics.count > 0 ? ((metrics.distribution.data[1] / metrics.count) * 100).toFixed(1) : 0}%</span>
                <span className={styles.statLabel}>Fiber Ratio</span>
              </div>
            </div>

            {/* [CHANGE 3] ISO REPORT BUTTON */}
            <div style={{marginTop: '1.5rem', padding: '1rem', background: 'rgba(0, 123, 255, 0.05)', borderRadius: '8px', border: '1px solid rgba(0, 123, 255, 0.2)'}}>
              <h3 className={styles.header} style={{marginBottom: '0.5rem', fontSize: '0.75rem', color: 'var(--color-primary)'}}>
                <FileText size={14}/> REGULATORY OUTPUT
              </h3>
              {currentRequestId && (
                <a 
                  href={`http://127.0.0.1:8000/report/${currentRequestId}`} 
                  target="_blank" 
                  rel="noreferrer"
                  style={{textDecoration: 'none'}}
                >
                  <button className={styles.activeBtn} style={{width: '100%', justifyContent: 'center', background: 'var(--color-primary)', color: 'white'}}>
                    <Download size={16} /> Download ISO CoA
                  </button>
                </a>
              )}
              <div style={{textAlign: 'center', marginTop: '0.5rem', fontSize: '0.65rem', opacity: 0.6, fontFamily: 'var(--font-mono)'}}>
                FORMAT: ISO 16232-10:2007
              </div>
            </div>

            <div style={{marginTop: '2rem'}}>
              <h3 className={styles.header}><Layers size={16} /> Layers</h3>
              <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
                {['original', 'mask', 'stain', 'heatmap'].map(layerId => {
                   const config = LAYER_CONFIG[layerId];
                   return (
                     <div key={layerId} style={{position: 'relative'}}>
                       <button onClick={() => setActiveLayer(layerId)} className={activeLayer === layerId ? styles.activeBtn : styles.inactiveBtn}>
                         <config.icon size={16} /> {config.label}
                       </button>
                       {layerId !== 'original' && (
                          <div className={styles.infoTrigger} onMouseEnter={() => setHoveredInfo(layerId)} onMouseLeave={() => setHoveredInfo(null)}>
                            <Info size={14}/>
                          </div>
                       )}
                       {hoveredInfo === layerId && (
                         <div className={styles.legendPopup}>
                           <strong style={{display:'block', marginBottom:'0.75rem', color: 'var(--color-primary)', borderBottom:'1px solid #333', paddingBottom:'4px'}}>{config.label}</strong>
                           <div style={{fontSize:'0.75rem', lineHeight:'1.5'}}>{config.desc}</div>
                         </div>
                       )}
                     </div>
                   );
                })}
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* MAIN STAGE */}
      <main className={styles.stage}>
        {imageState.original ? (
          <div style={{ position: 'relative', width: '90%', height: '90%' }}>
            <img src={imageState.original} style={{...imgStyle, opacity: activeLayer === 'original' ? 1 : 0.2}} alt="Original"/>
            {imageState.mask && <img src={imageState.mask} style={{...imgStyle, opacity: activeLayer === 'mask' ? 1 : 0, mixBlendMode: 'screen'}} alt="Mask"/>}
            {imageState.stain && <img src={imageState.stain} style={{...imgStyle, opacity: activeLayer === 'stain' ? 1 : 0}} alt="Stain"/>}
            {imageState.heatmap && <img src={imageState.heatmap} style={{...imgStyle, opacity: activeLayer === 'heatmap' ? 0.9 : 0}} alt="Heatmap"/>}
          </div>
        ) : ( <StandbyScreen /> )}
      </main>
    </div>
  );
};

const imgStyle = { width: '100%', height: '100%', objectFit: 'contain', position: 'absolute', top: 0, left: 0, transition: 'opacity 0.3s ease' };