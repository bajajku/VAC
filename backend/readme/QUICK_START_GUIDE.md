# 🚀 Quick Start Guide - Enhanced RAG with Preprocessing

## ⚡ 2-Minute Setup (Production Ready)

### **Step 1: Configure Environment**
```bash
# Copy environment template
cp env.example .env

# Edit .env file - set your API key
nano .env
# Set: TOGETHER_API_KEY=your_api_key_here
# Set: SKIP_AUTO_PROCESSING=true  (recommended for production)
```

### **Step 2: Preprocess Your Data (One Time)**
```bash
# Auto-process latest crawl data with enhanced cleaning
python preprocess_data.py --auto

# ✅ This takes 2-5 minutes but you only do it once!
# Creates optimally chunked, LLM-cleaned documents
```

### **Step 3: Start API (Fast!)**
```bash
# Start the enhanced API
python api.py

# ✅ Ready in 30 seconds - no more waiting!
```

### **Step 4: Test Your Enhanced System**
```bash
# Test the API
curl -X POST "http://localhost:8000/query-enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "retrieval_strategy": "hybrid",
    "k": 5
  }'
```

---

## 🎯 Key Benefits You Get

✅ **Lightning Fast Startup**: 30 seconds vs 5 minutes  
✅ **Better Answers**: Smart recursive text splitting + LLM cleaning  
✅ **Lower Costs**: No repeated expensive LLM calls  
✅ **Multiple Strategies**: Similarity, MMR, Hybrid, Ensemble retrieval  
✅ **Production Ready**: Robust error handling and monitoring  

---

## 📊 What Changed?

### **Before (Traditional)**
```
API Startup: Raw Data → LLM Clean → Split → Vector DB → Ready
Time: 5+ minutes every restart
Cost: High (repeated LLM calls)
```

### **After (Preprocessed)**
```
Offline: Raw Data → LLM Clean → Smart Split → Save
API Startup: Load Preprocessed → Vector DB → Ready  
Time: 30 seconds
Cost: Low (LLM runs once offline)
```

---

## 🛠️ Advanced Usage

### **Different Processing Methods**
```bash
# Enhanced processing (best quality)
python preprocess_data.py --auto

# Basic processing (faster, good quality)
python preprocess_data.py --auto --basic

# Custom chunk sizes
python preprocess_data.py --auto --chunk-size 1000 --chunk-overlap 200
```

### **Enhanced Query Strategies**
```python
import requests

# Hybrid retrieval (default, best overall)
response = requests.post("http://localhost:8000/query-enhanced", json={
    "question": "How does deep learning work?",
    "retrieval_strategy": "hybrid",
    "k": 5
})

# MMR (diverse results)
response = requests.post("http://localhost:8000/query-enhanced", json={
    "question": "What are neural networks?",
    "retrieval_strategy": "mmr",
    "k": 5
})

# Ensemble (highest quality, slower)
response = requests.post("http://localhost:8000/query-enhanced", json={
    "question": "Explain gradient descent",
    "retrieval_strategy": "ensemble",
    "k": 3
})
```

---

## 🔧 Configuration Options

### **Environment Variables (.env)**
```bash
# API Settings
TOGETHER_API_KEY=your_api_key_here
SKIP_AUTO_PROCESSING=true               # Skip expensive startup processing

# LLM Settings  
LLM_PROVIDER=chatopenai
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

# Processing Settings
CHUNK_SIZE=800                          # Chunk size for text splitting
CHUNK_OVERLAP=100                       # Overlap between chunks
```

### **Retrieval Strategies**
| Strategy | Speed | Quality | Use Case |
|----------|--------|---------|----------|
| `similarity` | Fastest | Good | Quick lookups |
| `mmr` | Fast | Great | Diverse results |
| `hybrid` | Medium | Excellent | Balanced (default) |
| `ensemble` | Slower | Best | Critical queries |

---

## 📁 File Structure After Setup

```
backend/
├── .env                                # Your configuration
├── api.py                             # Enhanced API server
├── preprocess_data.py                 # Offline preprocessing script
├── scripts/
│   ├── data_collection/
│   │   └── crawl_results/             # Raw scraped data
│   │       └── crawl_20240101.json
│   └── data_cleaning/
│       └── cleaned_data/              # Preprocessed data (auto-created)
│           ├── cleaned_data_20240101_120000.json
│           └── cleaned_data_20240101_120000_info.json
└── chroma_db/                         # Vector database (auto-created)
```

---

## 🚨 Troubleshooting

### **API Won't Start**
```bash
# Check if preprocessing is causing delays
export SKIP_AUTO_PROCESSING=true
python api.py
```

### **No Documents Found**
```bash
# Check for preprocessed data
ls scripts/data_cleaning/cleaned_data/

# If empty, run preprocessing
python preprocess_data.py --auto
```

### **Poor Answer Quality**
```bash
# Use enhanced processing for better cleaning
python preprocess_data.py --auto  # (not --basic)

# Try ensemble retrieval for important queries
curl -X POST "http://localhost:8000/query-enhanced" \
  -d '{"question": "your question", "retrieval_strategy": "ensemble"}'
```

---

## 🔄 Workflow for Different Scenarios

### **🏭 Production Deployment**
```bash
# 1. Set production config
echo "SKIP_AUTO_PROCESSING=true" >> .env

# 2. Preprocess data offline (during maintenance)
python preprocess_data.py --auto

# 3. Start API (fast startup)
python api.py

# 4. Monitor and update data as needed
```

### **🧪 Development & Testing**
```bash
# 1. Set dev config for flexibility
echo "SKIP_AUTO_PROCESSING=false" >> .env

# 2. Start API (will auto-process on startup)
python api.py

# 3. Test different settings
python preprocess_data.py --auto --basic  # Quick testing
python preprocess_data.py --auto          # Full quality
```

### **📊 Performance Optimization**
```bash
# Test different chunk sizes
python preprocess_data.py --auto --chunk-size 600   # Smaller chunks
python preprocess_data.py --auto --chunk-size 1200  # Larger chunks

# Compare retrieval strategies
curl -X POST "http://localhost:8000/query-enhanced" \
  -d '{"question": "test", "retrieval_strategy": "similarity"}'

curl -X POST "http://localhost:8000/query-enhanced" \
  -d '{"question": "test", "retrieval_strategy": "hybrid"}'
```

---

## 📈 Performance Monitoring

### **Check System Stats**
```bash
curl http://localhost:8000/stats
```

### **Monitor Retrieval Performance**
```bash
curl http://localhost:8000/retrieval-stats
```

### **Configure Retriever Settings**
```bash
curl -X POST "http://localhost:8000/configure-retriever" \
  -H "Content-Type: application/json" \
  -d '{
    "max_results": 10,
    "retrieval_strategy": "hybrid",
    "similarity_threshold": 0.8
  }'
```

---

## 🎉 You're Ready!

Your Enhanced RAG system now provides:

🚀 **Fast startup** (30 seconds vs 5 minutes)  
🎯 **Better answers** (smart chunking + LLM cleaning)  
💰 **Lower costs** (no repeated LLM preprocessing)  
🔧 **Multiple strategies** (similarity, MMR, hybrid, ensemble)  
📊 **Rich monitoring** (stats, performance tracking)  
🛡️ **Production ready** (robust error handling)  

**Next Steps:**
- Add your data to `scripts/data_collection/crawl_results/`
- Run `python preprocess_data.py --auto` when data changes
- Use different retrieval strategies for different use cases
- Monitor performance with `/stats` and `/retrieval-stats` endpoints

Happy querying! 🎊 