import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Camera, Sparkles, Sprout, Info, CheckCircle, AlertTriangle } from 'lucide-react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';

// Custom Components (Internal for simplicity in first pass)
const Header = () => (
  <header className="w-full py-8 text-center">
    <motion.div 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center justify-center gap-3 mb-2"
    >
      <div className="p-3 rounded-2xl bg-primary-500/20 backdrop-blur-xl border border-primary-500/30">
        <Sprout className="w-8 h-8 text-primary-400" />
      </div>
      <h1 className="text-4xl font-bold tracking-tight text-white bg-clip-text">
        FlorAI <span className="text-primary-400">ID</span>
      </h1>
    </motion.div>
    <p className="text-gray-400 text-lg">AI-Powered Flower Species Identification</p>
  </header>
);

const ConfidenceBar = ({ value }) => (
  <div className="w-full bg-gray-800 rounded-full h-2 mt-2 overflow-hidden">
    <motion.div 
      initial={{ width: 0 }}
      animate={{ width: `${value * 100}%` }}
      transition={{ duration: 1, ease: "easeOut" }}
      className="h-full bg-primary-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]"
    />
  </div>
);

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResults(null);
    setError(null);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    multiple: false
  });

  const handleIdentify = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      // Pointing to FastAPI backend
      const response = await axios.post('https://flower-system-0-1.onrender.com/api/v1/identify', formData);
      setResults(response.data.predictions);
    } catch (err) {
      setError(err.response?.data?.detail || "Connection Error. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen text-gray-100 flex flex-col items-center px-4 pb-20">
      <Header />

      <main className="w-full max-w-4xl flex flex-col items-center gap-12 mt-8">
        
        {/* Upload Section */}
        {!results && (
          <motion.div 
            layout
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full"
          >
            <div 
              {...getRootProps()} 
              className={`
                relative group cursor-pointer aspect-video w-full max-w-2xl mx-auto
                rounded-[2.5rem] border-2 border-dashed transition-all duration-500
                flex flex-col items-center justify-center gap-6 overflow-hidden
                ${isDragActive ? 'border-primary-500 bg-primary-500/10' : 'border-gray-700 bg-gray-900/50 hover:border-primary-500/50 hover:bg-gray-800/50'}
              `}
            >
              <input {...getInputProps()} />
              
              {preview ? (
                <img src={preview} className="absolute inset-0 w-full h-full object-cover opacity-60" alt="Preview" />
              ) : (
                <div className="absolute inset-0 bg-gradient-to-br from-primary-500/5 to-transparent pointer-events-none" />
              )}

              <div className="relative z-10 flex flex-col items-center text-center p-8">
                <div className={`p-5 rounded-full bg-gray-800 border border-gray-700 transition-transform duration-500 group-hover:scale-110 ${preview ? 'hidden' : ''}`}>
                  <Upload className="w-8 h-8 text-primary-400" />
                </div>
                <h3 className="text-xl font-semibold mt-4">{preview ? "Change Image" : "Drop flower image here"}</h3>
                <p className="text-gray-400 mt-2">or click to browse your files</p>
              </div>
            </div>

            {preview && !loading && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-8 flex justify-center"
              >
                <button 
                  onClick={handleIdentify}
                  className="px-8 py-4 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-2xl shadow-lg shadow-primary-600/20 flex items-center gap-3 transition-all group"
                >
                  <Sparkles className="w-5 h-5 group-hover:rotate-12 transition-transform" />
                  Identify Species
                </button>
              </motion.div>
            )}

            {loading && (
              <div className="mt-8 flex flex-col items-center gap-4">
                <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
                <p className="text-primary-400 font-medium animate-pulse">Analyzing botanical features...</p>
              </div>
            )}
            
            {error && (
              <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </motion.div>
        )}

        {/* Results Section */}
        <AnimatePresence>
          {results && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full flex flex-col lg:flex-row gap-8 items-start"
            >
              {/* Left: Image Preview */}
              <div className="w-full lg:w-1/2 sticky top-8">
                <motion.div 
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  className="rounded-[2.5rem] overflow-hidden border border-gray-800 shadow-2xl relative group"
                >
                  <img src={preview} className="w-full aspect-[4/5] object-cover" alt="Selected" />
                  <div className="absolute inset-x-0 bottom-0 p-8 bg-gradient-to-t from-black/80 to-transparent">
                    <button 
                      onClick={reset}
                      className="px-4 py-2 bg-white/10 backdrop-blur-md hover:bg-white/20 border border-white/20 rounded-xl text-white text-sm transition-all"
                    >
                      Process Another
                    </button>
                  </div>
                </motion.div>
              </div>

              {/* Right: Species Info */}
              <div className="w-full lg:w-1/2 flex flex-col gap-6">
                <h2 className="text-2xl font-bold text-gray-400 flex items-center gap-2">
                  <CheckCircle className="w-6 h-6 text-primary-400" />
                  Identification Results
                </h2>

                {results.map((pred, i) => (
                  <motion.div 
                    key={i}
                    initial={{ x: 20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: i * 0.2 }}
                    className={`
                      p-6 rounded-3xl border transition-all duration-300
                      ${i === 0 ? 'bg-primary-500/10 border-primary-500/40 shadow-xl shadow-primary-500/5' : 'bg-gray-900/50 border-gray-800'}
                    `}
                  >
                    <div className="flex gap-4">
                      <img src={pred.details.reference_image_url} className="w-20 h-20 rounded-2xl object-cover shrink-0" alt="Ref" />
                      <div className="flex-1 min-w-0">
                         <div className="flex justify-between items-start">
                            <h3 className="text-xl font-bold truncate">{pred.species.common_name}</h3>
                            <span className="text-sm font-mono text-primary-400">{(pred.confidence * 100).toFixed(1)}%</span>
                         </div>
                         <p className="text-gray-400 italic text-sm">{pred.species.scientific_name}</p>
                         <ConfidenceBar value={pred.confidence} />
                      </div>
                    </div>

                    {i === 0 && (
                      <motion.div 
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        className="mt-6 pt-6 border-t border-primary-500/20 grid grid-cols-2 gap-4"
                      >
                        <div className="col-span-2 flex items-center gap-2 text-primary-400 text-sm font-semibold mb-1">
                          <Info className="w-4 h-4" /> Botanical Facts
                        </div>
                        <div className="p-3 rounded-xl bg-black/30 border border-white/5">
                           <p className="text-[10px] text-gray-500 uppercase tracking-wider font-bold">Family</p>
                           <p className="text-sm">{pred.species.family}</p>
                        </div>
                        <div className="p-3 rounded-xl bg-black/30 border border-white/5">
                           <p className="text-[10px] text-gray-500 uppercase tracking-wider font-bold">Native Region</p>
                           <p className="text-sm">{pred.details.native_region}</p>
                        </div>
                        <div className="col-span-2 p-3 rounded-xl bg-black/30 border border-white/5">
                           <p className="text-[10px] text-gray-500 uppercase tracking-wider font-bold">Bloom Season</p>
                           <p className="text-sm">{pred.details.bloom_season}</p>
                        </div>
                        <div className="col-span-2 p-3 rounded-xl bg-primary-500/5 border border-primary-500/10">
                           <p className="text-[10px] text-primary-400 uppercase tracking-wider font-bold mb-1">Distinguishing Features</p>
                           <ul className="text-xs space-y-1">
                              {pred.details.distinguishing_features.map((f, idx) => (
                                <li key={idx} className="flex items-start gap-2">
                                   <div className="w-1 h-1 bg-primary-500 rounded-full mt-1.5" />
                                   {f}
                                </li>
                              ))}
                           </ul>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>
    </div>
  );
}
