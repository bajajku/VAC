# 🔄 Data Preprocessing Guide for Enhanced RAG

## Overview

The Enhanced RAG API now supports **offline data preprocessing** to avoid expensive LLM-based cleaning on every API startup. This approach:

- ✅ **Reduces API startup time** from minutes to seconds
- ✅ **Saves API costs** by avoiding repeated LLM calls
- ✅ **Improves reliability** - preprocessed data loads faster
- ✅ **Better for production** - predictable startup performance

---

## 🏗️ Preprocessing Workflow

### **Traditional Workflow (Slow)**
```
API Startup → Load Raw Data → LLM Cleaning → Text Splitting → Vector DB → Ready
     ⏱️ 2-5 minutes with expensive LLM calls every time
```

### **New Preprocessing Workflow (Fast)**
```
Offline: Raw Data → LLM Cleaning → Text Splitting → Save Cleaned Data
API Startup: Load Preprocessed Data → Vector DB → Ready
     ⏱️ 10-30 seconds, no LLM calls
```

---

## 🚀 Quick Start

### **1. Preprocess Your Data (Offline)**
```bash
# Auto-process the latest crawl results
python preprocess_data.py --auto

# Process a specific file
python preprocess_data.py --input path/to/your/data.json

# Use basic processing (no LLM cleaning, faster)
python preprocess_data.py --auto --basic
```

### **2. Configure API for Fast Startup**
```bash
# Create/update your .env file
cp env.example .env

# Edit .env and set:
SKIP_AUTO_PROCESSING=true
```

### **3. Start API (Fast)**
```bash
python api.py
# API will load preprocessed data in seconds!
```

---

## 📋 Preprocessing Script Options

### **Command Line Usage**
```bash
python preprocess_data.py [OPTIONS]

Options:
  --auto, -a              Auto-process latest crawl results
  --input, -i FILE        Process specific input file
  --output, -o DIR        Output directory (default: scripts/data_cleaning/cleaned_data)
  --basic, -b             Use basic processing (no LLM cleaning)
  --chunk-size SIZE       Chunk size for text splitting (default: 800)
  --chunk-overlap SIZE    Chunk overlap (default: 100)
```

### **Examples**

#### Auto-process latest data with enhanced cleaning:
```bash
python preprocess_data.py --auto
```

#### Process specific file with custom chunk size:
```bash
python preprocess_data.py \
  --input scripts/data_collection/crawl_results/data.json \
  --chunk-size 1000 \
  --chunk-overlap 150
```

#### Basic processing (no LLM, faster):
```bash
python preprocess_data.py --auto --basic
```

---

## 📁 Directory Structure

```
backend/
├── scripts/
│   ├── data_collection/
│   │   └── crawl_results/          # Raw crawled data
│   │       └── crawl_*.json
│   └── data_cleaning/
│       └── cleaned_data/           # Preprocessed cleaned data
│           ├── cleaned_data_*.json
│           └── cleaned_data_*_info.json
├── preprocess_data.py              # Preprocessing script
└── api.py                          # Enhanced API
```

---

## ⚙️ Configuration Options

### **Environment Variables**

Create a `.env` file:
```bash
# Skip auto-processing on API startup (recommended for production)
SKIP_AUTO_PROCESSING=true

# API configuration
TOGETHER_API_KEY=your_api_key_here
LLM_PROVIDER=chatopenai
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

# Processing configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

### **Processing Modes**

#### **Enhanced Processing (Recommended)**
- Uses LLM-based data cleaning
- Removes ads, navigation, footers
- Smart recursive text splitting
- Rich metadata enhancement
- **Time**: 2-5 minutes offline, but only once
- **Quality**: Highest

#### **Basic Processing (Fast)**
- No LLM cleaning
- Basic text splitting
- Minimal metadata
- **Time**: 30 seconds
- **Quality**: Good

---

## 🔄 Workflow Examples

### **Production Workflow**

1. **Initial Setup**:
```bash
# Preprocess your data once
python preprocess_data.py --auto

# Configure for fast startup
echo "SKIP_AUTO_PROCESSING=true" >> .env
```

2. **Daily Operation**:
```bash
# API starts fast (no preprocessing)
python api.py
# ✅ Ready in 10-30 seconds
```

3. **When New Data Arrives**:
```bash
# Preprocess new data offline
python preprocess_data.py --auto

# Restart API to load new data
# (or use API endpoints to reload)
```

### **Development Workflow**

1. **Frequent Changes**:
```bash
# Skip preprocessing during development
echo "SKIP_AUTO_PROCESSING=false" >> .env
python api.py
# API will auto-process on startup
```

2. **Testing Different Settings**:
```bash
# Test different chunk sizes
python preprocess_data.py --auto --chunk-size 1200
python preprocess_data.py --auto --chunk-size 600

# Compare performance
```

---

## 📊 Performance Comparison

| Mode | Startup Time | LLM Calls | Data Quality | Use Case |
|------|-------------|-----------|--------------|----------|
| **Auto-processing** | 2-5 minutes | Every startup | Highest | Development |
| **Preprocessed** | 10-30 seconds | None | Highest | Production |
| **Basic processing** | 30 seconds | None | Good | Quick testing |

---

## 🛠️ Advanced Usage

### **Custom Preprocessing Pipeline**

```python
from preprocess_data import DataPreprocessor

# Initialize with custom settings
preprocessor = DataPreprocessor(
    chunk_size=1000,
    chunk_overlap=200
)

# Process multiple files
files = ["data1.json", "data2.json"]
all_documents = []

for file in files:
    docs = preprocessor.process_json_file(Path(file))
    all_documents.extend(docs)

# Save combined results
preprocessor.save_processed_data(
    all_documents, 
    Path("combined_cleaned_data.json")
)
```

### **Batch Processing Multiple Files**

```bash
# Process all files in directory
for file in scripts/data_collection/crawl_results/*.json; do
    python preprocess_data.py --input "$file"
done
```

### **Scheduled Preprocessing**

```bash
# Add to crontab for daily preprocessing
# 0 2 * * * cd /path/to/backend && python preprocess_data.py --auto
```

---

## 🔧 API Integration

### **Loading Preprocessed Data via API**

```python
import requests

# Load specific preprocessed file
response = requests.post("http://localhost:8000/load-json-enhanced", json={
    "file_path": "scripts/data_cleaning/cleaned_data/cleaned_data_20240101_120000.json",
    "use_enhanced_processing": False  # Already preprocessed
})

print(response.json())
```

### **Checking What's Loaded**

```python
# Get system stats
response = requests.get("http://localhost:8000/stats")
stats = response.json()
print(f"Documents loaded: {stats.get('total_documents', 0)}")
```

---

## 🧪 Testing Preprocessed Data

### **Test Preprocessing Quality**
```bash
# Preprocess with enhanced cleaning
python preprocess_data.py --auto

# Start API
python api.py

# Test enhanced queries
python test_enhanced_api.py
```

### **Compare Processing Methods**
```bash
# Test basic preprocessing
python preprocess_data.py --auto --basic

# Test enhanced preprocessing  
python preprocess_data.py --auto

# Compare results via API queries
```

---

## 🚨 Troubleshooting

### **Common Issues**

#### **"No preprocessed data found"**
```bash
# Check if cleaned_data directory exists
ls scripts/data_cleaning/cleaned_data/

# Run preprocessing if empty
python preprocess_data.py --auto
```

#### **"API takes too long to start"**
```bash
# Check environment variable
echo $SKIP_AUTO_PROCESSING

# Set to skip auto-processing
export SKIP_AUTO_PROCESSING=true
```

#### **"Preprocessed data seems outdated"**
```bash
# Check file timestamps
ls -la scripts/data_cleaning/cleaned_data/

# Reprocess if needed
python preprocess_data.py --auto
```

---

## 💡 Best Practices

### **For Production**
1. ✅ Always set `SKIP_AUTO_PROCESSING=true`
2. ✅ Preprocess data offline during maintenance windows
3. ✅ Use enhanced processing for best quality
4. ✅ Monitor preprocessing logs for errors
5. ✅ Keep both raw and preprocessed data for rollback

### **For Development**
1. ✅ Use `SKIP_AUTO_PROCESSING=false` for quick iteration
2. ✅ Test both basic and enhanced preprocessing
3. ✅ Use smaller chunk sizes for faster processing
4. ✅ Keep preprocessed data for consistent testing

### **For Data Quality**
1. ✅ Always use enhanced processing for production
2. ✅ Monitor chunk size distribution
3. ✅ Validate preprocessing output before deployment
4. ✅ Test retrieval quality after preprocessing

---

## 🎉 Summary

The new preprocessing workflow gives you:

- 🚀 **Fast API startup** (seconds instead of minutes)
- 💰 **Lower costs** (no repeated LLM calls)
- 🔧 **Better control** over data processing
- 📈 **Production ready** performance
- 🔄 **Flexible workflows** for different use cases

**Quick Commands:**
```bash
# Preprocess data once
python preprocess_data.py --auto

# Configure fast startup
echo "SKIP_AUTO_PROCESSING=true" >> .env

# Start API fast
python api.py
```

Your Enhanced RAG API is now production-ready! 🚀 