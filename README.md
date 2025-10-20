# Multi-Agent LLM Platform with Continuous Learning

A production-ready MLOps platform for fine-tuning, deploying, and continuously improving multi-agent LLMs based on user feedback.

## 🌟 Features

- **Automated Fine-Tuning**: QLoRA-based training optimized for 48GB GPU
- **Multi-Agent Architecture**: 5 specialized agents (Query Understanding, MDX Generation, Data Analysis, Visualization, Error Resolution)
- **Continuous Learning**: Automated retraining every 2-3 weeks based on feedback
- **Production API**: FastAPI with monitoring, metrics, and health checks
- **Feedback Loop**: Built-in feedback collection and quality scoring
- **Model Versioning**: Full model registry with rollback support
- **Monitoring**: Prometheus + Grafana dashboards
- **Deployment Options**: Docker, Kubernetes, or bare-metal

## 📋 System Requirements

### Hardware
- **GPU**: 48GB VRAM (A100/RTX 6000 Ada recommended)
- **RAM**: 100GB system memory
- **Storage**: 200GB SSD
- **CPU**: 16+ cores recommended

### Software
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.10+
- CUDA 12.1+
- Docker 20.10+ (optional)
- Kubernetes 1.25+ (optional)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd multi_agent_llm_platform

# Create project structure
bash scripts/setup_project.sh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### 2. Configuration

Edit configuration files in `config/`:

```yaml
# config/training_config.yaml
num_train_epochs: 3
per_device_train_batch_size: 4
learning_rate: 2e-4

# config/deployment_config.yaml
api_port: 8000
use_vllm: true

# config/feedback_config.yaml
retraining_interval_days: 14
min_feedback_samples: 500
```

### 3. Initial Setup

```bash
# Run complete initial setup
python scripts/orchestrator.py --mode initial

# This will:
# 1. Generate synthetic training data
# 2. Train initial model (~10 hours)
# 3. Deploy API service
# 4. Start retraining scheduler
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Query understanding
curl -X POST http://localhost:8000/api/query/understand \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_query": "موجودی نقد شرکت در سال ۱۴۰۳ چقدر است؟"
  }'

# Submit feedback
curl -X POST http://localhost:8000/api/feedback \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_query": "...",
    "agent_type": "query_understanding",
    "model_output": "...",
    "rating": 5
  }'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER INTERFACE                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Gateway                        │
│  • Authentication  • Rate Limiting  • Monitoring        │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Query   │   │   MDX    │   │   Data   │
  │  Under-  │   │  Gener-  │   │ Analysis │
  │  standing│   │  ation   │   │          │
  └──────────┘   └──────────┘   └──────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
              ┌──────────────────┐
              │  Fine-Tuned LLM  │
              │  (Qwen2.5-7B)    │
              └──────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Feedback │   │  Model   │   │  Metrics │
  │   DB     │   │ Registry │   │ & Logs   │
  └──────────┘   └──────────┘   └──────────┘
        │
        ▼
  ┌──────────────────┐
  │   Retraining     │
  │   Scheduler      │
  │  (Every 2 weeks) │
  └──────────────────┘
```

## 🔄 Continuous Learning Workflow

### 1. **Production Phase** (Weeks 1-2)
```
User Query → API → Agent → Response
                     ↓
                 Feedback DB
```

### 2. **Data Collection**
- All interactions logged
- User feedback collected (ratings, corrections)
- Error patterns identified
- Quality scoring applied

### 3. **Retraining Trigger** (Week 2-3)
Automatic retraining triggered when:
- ✅ 14+ days since last training
- ✅ 500+ new feedback samples
- ✅ Performance drop > 10%
- ✅ Error rate > 10%

### 4. **Retraining Process**
```bash
1. Export feedback data (high-quality samples)
2. Combine with synthetic data (70/30 ratio)
3. Fine-tune model (8-12 hours)
4. Evaluate on validation set
5. Compare with current model
6. Deploy if better, rollback if worse
```

### 5. **Deployment Decision**
```python
if new_model_score > current_model_score - 5%:
    deploy_new_model()
else:
    rollback_to_previous()
```

## 📊 Monitoring

### Grafana Dashboards
Access at `http://localhost:3000` (admin/admin)

**Model Performance Dashboard**:
- Request latency (p50, p95, p99)
- Token throughput
- Error rates by agent
- GPU utilization

**Feedback Dashboard**:
- Average ratings over time
- Feedback volume by agent
- Top error patterns
- Training data growth

### Prometheus Metrics
Access at `http://localhost:9091`

Key metrics:
- `api_requests_total`
- `api_request_latency_seconds`
- `api_tokens_generated`
- `api_errors_total`

## 🛠️ Operations

### Check System Status
```bash
python scripts/orchestrator.py --mode status
```

### Manual Retraining
```bash
# Trigger retraining immediately
python scripts/orchestrator.py --mode retrain

# Or use the pipeline directly
python -m src.pipeline.retraining_pipeline --run-once
```

### Model Management
```bash
# List all versions
curl http://localhost:8000/api/models/versions

# Rollback to specific version
python scripts/rollback_model.py --version v1.2

# Compare versions
python scripts/compare_models.py --v1 v1.3 --v2 v1.4
```

### Backup and Recovery
```bash
# Backup models and feedback
python scripts/backup.py --output /backup/$(date +%Y%m%d)

# Restore from backup
python scripts/restore.py --input /backup/20240115
```

## 🐳 Deployment

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose -f deployment/docker/docker-compose.yml up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Kubernetes
```bash
# Apply configurations
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -l app=llm-api

# Scale replicas
kubectl scale deployment llm-api --replicas=3
```

### Bare Metal (Systemd)
```bash
# Install services
sudo cp deployment/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start services
sudo systemctl start llm-api llm-scheduler

# Enable auto-start
sudo systemctl enable llm-api llm-scheduler
```

## 📈 Performance Optimization

### For 48GB GPU
**Optimal settings** (already configured):
- Batch size: 4
- Gradient accumulation: 8
- LoRA rank: 128
- 4-bit quantization
- Flash Attention 2

### For 24GB GPU
```yaml
# config/training_config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
max_seq_length: 2048

# config/model_config.yaml
lora_r: 64
lora_alpha: 128
```

### For Production Inference
```yaml
# config/deployment_config.yaml
use_vllm: true  # 10-20x faster
inference_quantization: "8bit"
vllm_gpu_memory_utilization: 0.9
```

## 🔒 Security

### API Authentication
```python
# Add to .env
API_KEY=your-secret-key

# In src/api/middleware.py
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")
```

### Rate Limiting
```python
# Install: pip install slowapi
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/query/understand")
@limiter.limit("100/minute")
async def understand_query(...):
    ...
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific component
pytest tests/test_training.py
pytest tests/test_api.py

# Integration tests
pytest tests/test_integration.py -v

# Load testing
locust -f tests/load_test.py --host http://localhost:8000
```

## 📝 Troubleshooting

### Training OOM
```bash
# Reduce batch size
# Edit config/training_config.yaml
per_device_train_batch_size: 2

# Or use CPU offloading
# Edit config/training_config.yaml
use_deepspeed: true
```

### Slow Inference
```bash
# Enable vLLM
# Edit config/deployment_config.yaml
use_vllm: true

# Install: pip install vllm
```

### Feedback Not Collecting
```bash
# Check database
python scripts/check_feedback_db.py

# Re-initialize if needed
python scripts/init_feedback_db.py
```

## 📚 Additional Resources

- [Training Guide](docs/TRAINING.md)
- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture Details](docs/ARCHITECTURE.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Qwen2.5 team for the base model
- Hugging Face for transformers and PEFT
- vLLM team for high-performance inference
- FastAPI for the excellent framework

---

**Questions?** Open an issue or contact the maintainers.
