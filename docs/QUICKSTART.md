# 🚀 Quick Start Guide - PlagiCheck

Panduan cepat untuk memulai menggunakan PlagiCheck - Auto Paraphrasing System dalam 5 menit.

## ⚡ Setup Super Cepat

### 1. Clone & Setup (2 menit)
```bash
# Clone repository
git clone https://github.com/if-unismuh/plagicheck.git
cd plagicheck

# Automated setup (install semua dependencies)
python setup.py

# Verifikasi instalasi
python -c "print('✅ PlagiCheck ready!')"
```

### 2. Start Application (30 detik)
```bash
# Jalankan aplikasi
python main.py

# Buka browser: http://localhost:8000/docs
```

### 3. Test Pertama (1 menit)
```bash
# Test health check
curl http://localhost:8000/health

# Test paraphrase langsung
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Penelitian ini menggunakan metode kualitatif.",
    "method": "rule_based",
    "num_variants": 2
  }'
```

## 🎯 Use Cases Populer

### 📚 Academic Writing
```bash
# Upload thesis chapter
curl -X POST "http://localhost:8000/api/v2/documents/upload-enhanced" \
  -F "file=@thesis_bab1.pdf" \
  -F "chapter=BAB 1" \
  -F "preserve_structure=true" \
  -F "extract_academic_terms=true"

# Paraphrase dengan preservasi istilah akademik
curl -X POST "http://localhost:8000/api/v2/documents/{document_id}/paraphrase-enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "hybrid",
    "preserve_academic_terms": true,
    "preserve_citations": true,
    "num_variants": 3
  }'
```

### 📝 Content Creation
```bash
# Paraphrase artikel untuk multiple platform
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Teknologi artificial intelligence semakin berkembang pesat.",
    "method": "indot5",
    "num_variants": 5
  }'
```

### 🔍 Quality Check
```bash
# Analisis kualitas teks
curl -X POST "http://localhost:8000/api/v2/text/quality-assessment" \
  -H "Content-Type: application/json" \
  -d '"Penelitian ini menggunakan metode kualitatif untuk menganalisis data."'

# NLP analysis lengkap
curl -X POST "http://localhost:8000/api/v2/text/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Studi ini menerapkan pendekatan analisis kualitatif.",
    "extract_academic_terms": true
  }'
```

## 🌐 Web Interface Usage

### Swagger UI (Recommended)
1. Buka: `http://localhost:8000/docs`
2. Expand endpoint yang ingin ditest
3. Click "Try it out"
4. Input parameter
5. Execute

### ReDoc (Alternative)
1. Buka: `http://localhost:8000/redoc`
2. Browse dokumentasi lengkap
3. Copy curl examples

## 🐍 Python Integration

### Simple Usage
```python
import requests

# Client sederhana
def paraphrase_text(text, method="indot5"):
    url = "http://localhost:8000/api/v2/text/paraphrase-direct"
    data = {
        "text": text,
        "method": method,
        "num_variants": 3
    }
    response = requests.post(url, json=data)
    return response.json()

# Usage
result = paraphrase_text("Penelitian ini menggunakan metode kualitatif.")
print("Best variant:", result['best_variant'])
```

### Advanced Class
```python
import requests
from typing import List, Dict, Any

class PlagiCheckClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def paraphrase(self, text: str, method: str = "hybrid", 
                   num_variants: int = 3) -> Dict[str, Any]:
        """Paraphrase text with specified method"""
        url = f"{self.base_url}/api/v2/text/paraphrase-direct"
        data = {
            "text": text,
            "method": method,
            "preserve_academic_terms": True,
            "num_variants": num_variants
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text with NLP pipeline"""
        url = f"{self.base_url}/api/v2/text/analyze"
        data = {"text": text, "extract_academic_terms": True}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def upload_document(self, file_path: str, chapter: str = None) -> Dict[str, Any]:
        """Upload document for processing"""
        url = f"{self.base_url}/api/v2/documents/upload-enhanced"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'chapter': chapter,
                'preserve_structure': 'true',
                'extract_academic_terms': 'true'
            }
            response = requests.post(url, files=files, data=data)
        
        response.raise_for_status()
        return response.json()

# Usage examples
client = PlagiCheckClient()

# Paraphrase text
result = client.paraphrase(
    "Penelitian ini menggunakan metode analisis kualitatif.",
    method="hybrid"
)
print(f"Original: {result['original_text']}")
print(f"Best variant: {result['best_variant']}")

# Analyze text
analysis = client.analyze_text(
    "Studi ini menerapkan pendekatan metodologi kualitatif."
)
print(f"Readability: {analysis['overall_readability']}")
print(f"Academic terms: {analysis['academic_terms_count']}")

# Upload document
doc_result = client.upload_document("document.pdf", "BAB 1")
print(f"Document ID: {doc_result['document_id']}")
```

## 🌊 JavaScript/Node.js Integration

```javascript
const axios = require('axios');

class PlagiCheckClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async paraphrase(text, method = 'indot5', numVariants = 3) {
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/v2/text/paraphrase-direct`,
                {
                    text: text,
                    method: method,
                    preserve_academic_terms: true,
                    num_variants: numVariants
                }
            );
            return response.data;
        } catch (error) {
            throw new Error(`Paraphrase failed: ${error.response?.data?.detail || error.message}`);
        }
    }
    
    async analyzeText(text) {
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/v2/text/analyze`,
                {
                    text: text,
                    extract_academic_terms: true
                }
            );
            return response.data;
        } catch (error) {
            throw new Error(`Analysis failed: ${error.response?.data?.detail || error.message}`);
        }
    }
}

// Usage
const client = new PlagiCheckClient();

async function demo() {
    try {
        // Paraphrase
        const result = await client.paraphrase(
            'Penelitian ini menggunakan metode kualitatif.',
            'hybrid'
        );
        console.log('Best variant:', result.best_variant);
        
        // Analysis
        const analysis = await client.analyzeText(
            'Studi ini menerapkan pendekatan metodologi kualitatif.'
        );
        console.log('Readability:', analysis.overall_readability);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

demo();
```

## 📊 Performance Tips

### Optimasi untuk High Volume
```bash
# Gunakan batch processing
for file in *.pdf; do
    curl -X POST "http://localhost:8000/api/v2/documents/upload-enhanced" \
      -F "file=@$file" \
      -F "chapter=$(basename $file .pdf)" &
done
wait
```

### Monitoring Performance
```bash
# Check system stats
curl http://localhost:8000/api/v2/performance/stats

# Clear cache jika memory tinggi
curl -X POST http://localhost:8000/api/v2/performance/clear-cache
```

## 🔧 Configuration Cepat

### Development (Lightweight)
```env
DEBUG=true
USE_GPU=false
MAX_CONCURRENT_JOBS=2
DATABASE_URL=sqlite:///./plagicheck.db
LOG_LEVEL=DEBUG
```

### Production (Performance)
```env
DEBUG=false
USE_GPU=true
MAX_CONCURRENT_JOBS=5
DATABASE_URL=postgresql://user:pass@localhost/plagicheck
LOG_LEVEL=WARNING
ENABLE_RATE_LIMITING=true
```

## 🐛 Quick Troubleshooting

### Port sudah digunakan
```bash
# Find process
lsof -i :8000
# Kill process
kill -9 <PID>
# Atau gunakan port lain
uvicorn app.api.routes:app --port 8001
```

### Model loading error
```bash
# Clear cache
rm -rf ~/.cache/huggingface/
# Restart aplikasi
python main.py
```

### Database error
```bash
# Reset database
python migrate_db.py reset
python migrate_db.py migrate
```

### Memory issues
```bash
# Check memory
free -h
# Clear cache
curl -X POST http://localhost:8000/api/v2/performance/clear-cache
```

## 📱 Mobile/App Integration

### Flutter/Dart Example
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class PlagiCheckClient {
  final String baseUrl;
  
  PlagiCheckClient({this.baseUrl = 'http://localhost:8000'});
  
  Future<Map<String, dynamic>> paraphraseText(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v2/text/paraphrase-direct'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'text': text,
        'method': 'indot5',
        'num_variants': 3,
      }),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to paraphrase text');
    }
  }
}
```

### React Native Example
```javascript
const PlagiCheckAPI = {
  baseUrl: 'http://localhost:8000',
  
  async paraphraseText(text, method = 'indot5') {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/text/paraphrase-direct`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          method: method,
          num_variants: 3,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Paraphrase error:', error);
      throw error;
    }
  }
};

// Usage in React component
const ParaphraseComponent = () => {
  const [result, setResult] = useState('');
  
  const handleParaphrase = async () => {
    try {
      const response = await PlagiCheckAPI.paraphraseText(
        'Penelitian ini menggunakan metode kualitatif.'
      );
      setResult(response.best_variant);
    } catch (error) {
      console.error('Error:', error);
    }
  };
  
  return (
    <View>
      <Button title="Paraphrase" onPress={handleParaphrase} />
      <Text>{result}</Text>
    </View>
  );
};
```

## 📚 Learning Resources

### Sample Texts untuk Testing
```javascript
const sampleTexts = {
  academic: "Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji dampak teknologi digital terhadap pembelajaran mahasiswa.",
  
  formal: "Berdasarkan hasil analisis data, dapat disimpulkan bahwa implementasi sistem informasi memberikan dampak positif terhadap efisiensi operasional.",
  
  methodology: "Data dikumpulkan melalui teknik wawancara mendalam dengan 30 responden yang dipilih menggunakan purposive sampling.",
  
  conclusion: "Temuan penelitian ini konsisten dengan studi sebelumnya yang menunjukkan korelasi positif antara variabel X dan Y."
};
```

### Method Comparison
```bash
# Test semua method dengan text yang sama
text="Penelitian ini menggunakan metode kualitatif."

# IndoT5 (Neural model)
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$text\",\"method\":\"indot5\"}"

# Rule-based (Fast)
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$text\",\"method\":\"rule_based\"}"

# Hybrid (Best quality)
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$text\",\"method\":\"hybrid\"}"
```

## 🎉 You're Ready!

Selamat! Anda sudah siap menggunakan PlagiCheck. Untuk pembelajaran lebih lanjut:

- 📖 [Full Documentation](../README.md)
- 🧪 [Testing Guide](../TESTING.md) 
- 🔧 [Configuration Guide](CONFIGURATION.md)
- 📡 [API Reference](API.md)
- 📮 [Postman Collection](../postman/README.md)

---

💡 **Pro Tips:**
- Gunakan method `hybrid` untuk kualitas terbaik
- Aktifkan `preserve_academic_terms` untuk dokumen akademik
- Monitor performance dengan `/api/v2/performance/stats`
- Test dengan sample data sebelum production

🚀 **Happy Paraphrasing!**
