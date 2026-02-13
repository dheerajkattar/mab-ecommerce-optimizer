# Multi-Armed Bandit E-Commerce Optimizer

> **Production-ready framework for intelligent A/B testing and real-time optimization using Multi-Armed Bandit algorithms**

Stop losing revenue to suboptimal A/B tests. Traditional A/B testing wastes traffic on underperforming variants while you wait for statistical significance. This framework uses Multi-Armed Bandit algorithms to dynamically shift traffic toward winning variants in real-time, maximizing conversions while still exploring alternatives.

## 🎯 What Problem Does This Solve?

Imagine you're running an e-commerce site and want to test three different call-to-action button colors: blue, green, and red. Traditional A/B testing would split traffic evenly (33% each) for weeks until you have statistical significance. If the green button converts at 18% while blue converts at 12% and red at 9%, you're losing money every day you send traffic to the underperforming variants.

**But it gets even better:** What if traffic patterns change throughout the day? Maybe the blue button performs poorly in the morning but converts exceptionally well during evening hours when a different demographic visits your site. Or perhaps the red button suddenly starts outperforming others after a design update elsewhere on your page. Multi-Armed Bandit algorithms detect these shifts in real-time and automatically redirect traffic to whichever arm is currently performing best - even if it was the worst performer an hour ago.

**Multi-Armed Bandit algorithms solve this by:**
- **Dynamically reallocating traffic** to better-performing variants as data accumulates
- **Adapting on the fly** to changing performance - if a previously underperforming variant starts converting better, the algorithm automatically shifts traffic back to it
- **Reducing regret** (the cost of showing suboptimal variants)
- **Accelerating learning** through Bayesian and confidence-based approaches
- **Continuously optimizing** without needing to "end" the experiment

## 🚀 Try It Now

**No installation required** - explore the live demos:

- **📊 [Interactive Dashboard](https://mab-ecommerce-optimizer.streamlit.app/)** - Simulator playground with live algorithm visualization
- **🔧 [API Documentation](https://mab-api.onrender.com/docs)** - Interactive Swagger UI with "Try it out" functionality

## ✨ Features

### Three Battle-Tested Strategies

All strategies continuously adapt to changing performance patterns, automatically shifting traffic when an arm's conversion rate changes:

1. **Thompson Sampling** (`thompson`)
   - Bayesian approach using Beta distributions
   - Excellent balance between exploration and exploitation
   - Naturally handles uncertainty through probabilistic sampling
   - Quickly adapts when arm performance shifts due to updated priors

2. **UCB1** (`ucb1`)
   - Upper Confidence Bound algorithm
   - Optimistic in the face of uncertainty
   - Strong theoretical guarantees with logarithmic regret bounds
   - Confidence intervals automatically tighten or widen based on recent performance

3. **Epsilon-Greedy** (`epsilon_greedy`)
   - Simple yet effective exploration-exploitation strategy
   - Configurable exploration rate (default ε=0.1)
   - Easy to understand and tune
   - Periodically re-explores all arms to detect performance changes

### Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   FastAPI   │─────▶│    Redis     │◀─────│  Streamlit  │
│  (API Layer)│      │ (State Store)│      │ (Dashboard) │
└─────────────┘      └──────────────┘      └─────────────┘
      │                      │                      │
      │                      │                      │
   Decision              Arm State             Live Stats
   & Reward              Storage            & Simulation
```

- **FastAPI**: High-performance async API for decision requests and reward tracking
- **Redis**: Fast, persistent state storage for arm statistics and experiment configuration
- **Streamlit**: Interactive dashboard for monitoring experiments and running simulations
- **Docker**: Containerized deployment with Docker Compose for local development
- **AWS Lambda**: Serverless deployment option with SAM template

---

## 📖 API Documentation

### Base URLs

- **Production**: `https://mab-api.onrender.com`
- **Local Development**: `http://localhost:8000`

### Health Check

```bash
# Check API health
curl https://mab-api.onrender.com/health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

## 🔄 Core Workflow

The typical Multi-Armed Bandit workflow follows these steps:

1. **Create Experiment** - Define your test with multiple variants (arms)
2. **Request Decision** - For each user, get the recommended variant to show
3. **Report Reward** - Send conversion data (1.0 for success, 0.0 for failure)
4. **Monitor Results** - Track performance in real-time via dashboard or API

The algorithm continuously learns and adapts - if arm performance changes, traffic reallocation happens automatically.

### Real-World Adaptive Scenario

**9 AM:** Your blue button converts at 15%, green at 10%, red at 8%. The algorithm sends 70% of traffic to blue.

**2 PM:** A different demographic visits your site. Now blue converts at 8%, but red jumps to 18%. Within minutes, the algorithm detects this shift and automatically redirects 65% of traffic to red.

**6 PM:** Evening shoppers arrive. Green now converts at 20%, outperforming both others. The algorithm adapts again, reallocating traffic to green.

**Result:** You've maximized conversions throughout the day by automatically adapting to each audience segment, without manual intervention or stopping the experiment.

Let's walk through each step with detailed examples.

---

## 📚 API Endpoints Reference

### 1. Create Experiment

Create a new experiment with multiple variants (arms) to test.

**Endpoint:** `POST /experiments`

**Request Body:**
```json
{
  "experiment_id": "cta-button-test",
  "arm_ids": ["blue-button", "green-button", "red-button"],
  "strategy": "thompson",
  "strategy_params": {
    "seed": 42
  }
}
```

**Field Descriptions:**
- `experiment_id` (required): Unique identifier for your experiment (1-120 characters)
- `arm_ids` (required): Array of variant IDs to test (minimum 2, must be unique)
- `strategy` (optional): Algorithm to use - `thompson`, `ucb1`, or `epsilon_greedy` (defaults to server config)
- `strategy_params` (optional): Strategy-specific parameters (e.g., `epsilon` for epsilon-greedy, `seed` for reproducibility)

**Example with curl:**
```bash
curl -X POST https://mab-api.onrender.com/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "cta-button-test",
    "arm_ids": ["blue-button", "green-button", "red-button"],
    "strategy": "thompson"
  }'
```

**Example with httpie:**
```bash
http POST https://mab-api.onrender.com/experiments \
  experiment_id=cta-button-test \
  arm_ids:='["blue-button", "green-button", "red-button"]' \
  strategy=thompson
```

**Response (201 Created):**
```json
{
  "experiment_id": "cta-button-test",
  "arm_ids": ["blue-button", "green-button", "red-button"],
  "strategy": "thompson",
  "strategy_params": {}
}
```

---

### 2. Get Experiment Details

Retrieve configuration for an existing experiment.

**Endpoint:** `GET /experiments/{experiment_id}`

**Example:**
```bash
curl https://mab-api.onrender.com/experiments/cta-button-test
```

**Response (200 OK):**
```json
{
  "experiment_id": "cta-button-test",
  "arm_ids": ["blue-button", "green-button", "red-button"],
  "strategy": "thompson",
  "strategy_params": {}
}
```

---

### 3. Request Decision

**This is the main endpoint** - call it for each user to get which variant to show. The algorithm automatically balances exploration and exploitation.

**Endpoint:** `GET /decision?experiment_id={experiment_id}`

**Parameters:**
- `experiment_id` (required): The experiment ID to get a decision for

**Example:**
```bash
curl "https://mab-api.onrender.com/decision?experiment_id=cta-button-test"
```

**Response (200 OK):**
```json
{
  "experiment_id": "cta-button-test",
  "arm_id": "green-button",
  "strategy": "thompson",
  "metadata": {
    "arms": ["blue-button", "green-button", "red-button"]
  }
}
```

**Integration Pattern:**
```python
# Your application code
import requests

response = requests.get(
    "https://mab-api.onrender.com/decision",
    params={"experiment_id": "cta-button-test"}
)
arm = response.json()["arm_id"]

# Show the recommended variant to the user
render_button(color=arm.split("-")[0])
```

---

### 4. Report Reward

Send conversion data back to the API after user interaction. This is how the algorithm learns which variants perform best.

**Endpoint:** `POST /reward`

**Request Body:**
```json
{
  "experiment_id": "cta-button-test",
  "arm_id": "green-button",
  "reward": 1.0
}
```

**Field Descriptions:**
- `experiment_id` (required): The experiment ID
- `arm_id` (required): The variant that was shown (from the decision response)
- `reward` (required): Numeric value between 0.0 and 1.0
  - `1.0` = positive outcome (user clicked, converted, purchased)
  - `0.0` = no conversion
  - Values in between = partial credit (e.g., 0.5 for add-to-cart without purchase)

**Example with curl:**
```bash
curl -X POST https://mab-api.onrender.com/reward \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "cta-button-test",
    "arm_id": "green-button",
    "reward": 1.0
  }'
```

**Example with httpie:**
```bash
http POST https://mab-api.onrender.com/reward \
  experiment_id=cta-button-test \
  arm_id=green-button \
  reward:=1.0
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "experiment_id": "cta-button-test",
  "arm_id": "green-button",
  "strategy": "thompson"
}
```

---

### 5. Add Arms (Mid-Experiment)

Add new variants to an existing experiment without disrupting ongoing tests.

**Endpoint:** `POST /experiments/{experiment_id}/arms`

**Request Body:**
```json
{
  "arm_ids": ["orange-button", "purple-button"]
}
```

**Example:**
```bash
curl -X POST https://mab-api.onrender.com/experiments/cta-button-test/arms \
  -H "Content-Type: application/json" \
  -d '{
    "arm_ids": ["orange-button", "purple-button"]
  }'
```

**Response (200 OK):**
```json
{
  "experiment_id": "cta-button-test",
  "arm_ids": ["blue-button", "green-button", "red-button", "orange-button", "purple-button"],
  "strategy": "thompson",
  "strategy_params": {}
}
```

---

### 6. Hot-Swap Strategy

Change the algorithm for an experiment without losing accumulated data.

**Endpoint:** `POST /config`

**Request Body:**
```json
{
  "experiment_id": "cta-button-test",
  "strategy": "ucb1",
  "strategy_params": {}
}
```

**Example:**
```bash
curl -X POST https://mab-api.onrender.com/config \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "cta-button-test",
    "strategy": "ucb1"
  }'
```

**Response (200 OK):**
```json
{
  "experiment_id": "cta-button-test",
  "strategy": "ucb1",
  "strategy_params": {}
}
```

---

## 🎬 Complete Example Workflow

Here's a complete end-to-end example testing product image variants:

```bash
# 1. Create experiment with 3 product image variants
curl -X POST https://mab-api.onrender.com/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "product-image-test",
    "arm_ids": ["lifestyle-photo", "white-background", "in-use-photo"],
    "strategy": "thompson"
  }'

# 2. For each user visit, get a decision
curl "https://mab-api.onrender.com/decision?experiment_id=product-image-test"
# Response: {"arm_id": "lifestyle-photo", ...}

# 3a. User adds to cart (positive outcome)
curl -X POST https://mab-api.onrender.com/reward \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "product-image-test",
    "arm_id": "lifestyle-photo",
    "reward": 1.0
  }'

# 3b. Or user bounces (negative outcome)
curl -X POST https://mab-api.onrender.com/reward \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "product-image-test",
    "arm_id": "white-background",
    "reward": 0.0
  }'

# 4. Check experiment status
curl https://mab-api.onrender.com/experiments/product-image-test
```

---

## 🎮 Interactive Tools

### Live Dashboard

**URL:** [https://mab-ecommerce-optimizer.streamlit.app/](https://mab-ecommerce-optimizer.streamlit.app/)

The Streamlit dashboard provides two powerful interfaces:

#### Live Stats Tab
- Real-time monitoring of active experiments
- Visualize empirical conversion rates for each arm
- Track pull counts and performance metrics
- Auto-refresh capability for continuous monitoring
- Direct Redis connection for instant updates

#### Simulator Playground
- **Interactive experimentation** without affecting production
- Adjust true conversion rates for each arm
- Select algorithm (Thompson Sampling, UCB1, Epsilon-Greedy)
- Watch cumulative regret and reward evolve in real-time
- Visualize arm selection patterns
- Perfect for understanding algorithm behavior before deployment

### API Documentation

**Swagger UI:** [https://mab-api.onrender.com/docs](https://mab-api.onrender.com/docs)

Interactive API explorer with:
- Complete endpoint documentation
- Request/response schemas with examples
- "Try it out" functionality to test endpoints directly
- Authentication and validation details

**ReDoc:** [https://mab-api.onrender.com/redoc](https://mab-api.onrender.com/redoc)

Alternative documentation format with:
- Clean, readable layout
- Searchable endpoint reference
- Detailed model schemas

---

## 🛠️ Local Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for full stack)
- Redis (or use Docker)

### Option 1: Full Stack with Docker Compose (Recommended)

This brings up the API, Redis, and Dashboard together:

```bash
# Clone the repository
git clone <your-repo-url>
cd mab-ecommerce-optimizer

# Copy environment variables
cp .env.example .env

# Start all services
docker compose up --build
```

**Services will be available at:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Redis: localhost:6379
- API Docs: http://localhost:8000/docs
- Load Tester (Locust): http://localhost:8089

### Option 2: Local Python Development

Install dependencies with optional groups:

```bash
# Install everything for local development
python -m pip install -U pip
pip install -e ".[api,dashboard,bench,dev]"

# Or install specific components
pip install -e ".[api]"         # API only
pip install -e ".[dashboard]"   # Dashboard only
```

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Strategy configuration
ACTIVE_STRATEGY=UCB1              # Default strategy (UCB1, THOMPSON, EPSILON_GREEDY)
LOG_LEVEL=INFO

# Redis connection
REDIS_URL=redis://localhost:6379/0

# For Docker Compose
BANDIT_API_URL=http://bandit-api:8000
```

### Run API Locally

```bash
pip install -e ".[api]"
uvicorn bandit_api.main:app --reload
```

### Run Dashboard Locally

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

---

## 🧪 Strategy Comparison

### When to Use Each Algorithm

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Thompson Sampling** | Most use cases, especially with sparse data or shifting patterns | Fast convergence, naturally handles uncertainty, adapts quickly to performance changes | Requires more computation (Beta distribution sampling) |
| **UCB1** | Risk-averse scenarios, need theoretical guarantees | Deterministic, strong regret bounds, good for detecting improving arms | Can be overly exploratory early on |
| **Epsilon-Greedy** | Simple scenarios, educational purposes, stable conversion rates | Easy to understand and tune, constant exploration ensures detection of changes | Fixed exploration rate may miss rapid changes between exploration rounds |

### Strategy Parameters

**Thompson Sampling:**
```json
{
  "strategy": "thompson",
  "strategy_params": {
    "seed": 42  // Optional: for reproducible experiments
  }
}
```

**UCB1:**
```json
{
  "strategy": "ucb1",
  "strategy_params": {
    "seed": 42  // Optional: for reproducible experiments
  }
}
```

**Epsilon-Greedy:**
```json
{
  "strategy": "epsilon_greedy",
  "strategy_params": {
    "epsilon": 0.1,  // Exploration rate (0-1), default 0.1 means 10% exploration
    "seed": 42       // Optional: for reproducible experiments
  }
}
```

---

## 🚀 Production Deployment

### AWS Lambda with SAM

The API has been validated with load testing and algorithm performance tests. See [Testing and Benchmarking](#-testing-and-benchmarking) for evidence and reproducible steps.

Serverless deployment files are in `infra/sam/`:

```bash
cd infra/sam

# Validate template
sam validate

# Build container image
sam build

# Deploy (first time - interactive)
sam deploy --guided

# Subsequent deployments
sam deploy
```

**Required AWS Resources:**
- VPC with private subnets
- ElastiCache Redis cluster
- Lambda container image
- HTTP API Gateway

**Configuration in `samconfig.toml`:**
- `VpcId`: Your VPC ID
- `PrivateSubnetIds`: At least two private subnets for Redis

### Docker Production Build

```bash
# Build production API image
docker build -f bandit_api/Dockerfile -t mab-api:latest .

# Build dashboard image
docker build -f dashboard/Dockerfile -t mab-dashboard:latest .

# Run with production settings
docker run -e REDIS_URL=redis://your-redis:6379 -p 8000:8000 mab-api:latest
```

---

## 📊 Monitoring and Observability

### Metrics to Track

1. **Per-Experiment Metrics:**
   - Total pulls per arm
   - Empirical conversion rate per arm (track over time to see shifts)
   - Cumulative regret
   - Cumulative reward
   - Recent performance trends (e.g., last hour vs. last day)

2. **API Performance:**
   - Decision request latency
   - Reward post latency
   - Redis connection health
   - Request success rate

3. **Algorithm Behavior:**
   - Exploration vs exploitation ratio
   - Arm selection distribution over time (watch for traffic shifts)
   - Convergence to optimal arm
   - Detection of performance reversals (when previously poor arms improve)

### Using the Dashboard

The Live Stats tab provides real-time insights:
- Select experiment from dropdown
- View arm performance table
- Interactive charts of conversion rates
- Auto-refresh for continuous monitoring

---

## 🧪 Testing and Benchmarking

### Run Tests

```bash
pip install -e ".[dev]"
pytest
```

### Load Testing with Locust

The repository includes a Locust load test configuration:

```bash
# Via Docker Compose
docker compose up load-tester

# Visit http://localhost:8089 and configure:
# - Number of users
# - Spawn rate
# - Host (already set to API)

# Or run locally
pip install -e ".[bench]"
locust -f tests/locustfile.py --host=http://localhost:8000
```

### Load Test Results (Evidence)

We exercised the API under concurrent traffic using Locust and observed **0% failures (no errors/timeouts)** during the run. The load scenario is captured in [`tests/locustfile.py`](tests/locustfile.py) and is designed to resemble an e-commerce integration pattern:

- **Traffic mix**: ~5x more `GET /decision` calls than `POST /reward` calls (decisioning is high-frequency; conversions are lower-frequency).
- **Realistic state behavior**: experiment creation is attempted on startup and treats `409 Conflict` as expected when multiple users race to create the same experiment.
- **Strategy coverage**: exercises **Thompson Sampling** under load (the production default).

To reproduce and export your own run artifacts, use the headless mode shown in the comments at the top of [`tests/locustfile.py`](tests/locustfile.py) (CSV export).

### Algorithm Performance Testing (Evidence)

In addition to API load testing, we validated the learning behavior of the core strategies with a deterministic convergence benchmark. Full details are in [`convergence_test_results.md`](convergence_test_results.md).

**Key findings (20 synthetic 4-arm environments, 2,000 rounds each, seed=42):**
- **Thompson Sampling** (recommended default): median convergence round **110**, mean final regret **20.89**
- **Epsilon-Greedy (eps=0.1)**: median convergence round **110**, mean final regret **62.93**
- **UCB1**: median convergence round **331**, mean final regret **93.68**

---

## 🤝 Contributing

Contributions welcome! This is a production-ready framework with:
- Comprehensive test coverage
- Type hints throughout
- Pydantic validation
- Async-ready architecture
- Docker and serverless deployment options

---

## 📄 License

[Add your license here]

---

## 🔗 Additional Resources

- **Bandit Algorithms**: [Bandit Algorithms for Website Optimization (O'Reilly)](https://www.oreilly.com/library/view/bandit-algorithms-for/9781449341565/)
- **Thompson Sampling**: [Thompson Sampling Tutorial](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- **Multi-Armed Bandits**: [Introduction to Multi-Armed Bandits](https://arxiv.org/abs/1904.07272)

---

**Built with:** FastAPI • Redis • Streamlit • Docker • AWS Lambda

**Try it now:** [Live Dashboard](https://mab-ecommerce-optimizer.streamlit.app/) | [API Docs](https://mab-api.onrender.com/docs)
