# 🛡️ Sentinel AI: The Ultimate Beginner's Guide 

Welcome to **Sentinel AI**! 

If you have zero technical background, don't worry. This guide is written exactly for you. It will take you step-by-step through setting up, training the AI, running the simulator, and viewing the live dashboard on your computer.

### ❓ What is Sentinel AI?
Sentinel AI is a **Predictive Credit Risk Engine** built for the Indian financial market. 
Imagine a bank that wants to know if a customer is about to default on a loan. Instead of waiting for them to miss a payment, Sentinel AI watches their **real-time bank transactions** (like Swiggy orders, ATM withdrawals, and Salary deposits) and uses an AI model to calculate a **Pulse Score** (a health score between 0 and 1000). If the score drops, the dashboard glows orange or red, warning the bank instantly.

---

## 🛠️ Step 0: What You Need Installed
Before you start, you need three pieces of software installed on your computer. If you don't have them, download them from Google:
1. **Python** (Version 3.11 or higher) - *The programming language that runs the AI.*
2. **Node.js** - *The software that runs the visual Dashboard.*
3. **Docker Desktop** - *A tool that runs databases (like PostgreSQL and Redis) smoothly in the background.*

---

## 🚀 The Step-by-Step "Zero to Hero" Run Guide

You will need to open a few different "Terminal" or "Command Prompt" windows to run different parts of the system at the same time. 

### Step 1: Install Python Requirements
Open your first Terminal, navigate to the `sentinel-ai` project folder, and install the required Python libraries.
```bash
# Navigate to the project
cd sentinel-ai

# Install all the AI and backend code requirements
pip install -r requirements.txt
```

### Step 2: Start the Databases (Docker)
Sentinel AI needs places to store its data. We use Docker to spin up these databases (PostgreSQL, Redis, Kafka, and DynamoDB).
Make sure **Docker Desktop** is open and running on your computer. Then, in your terminal, type:
```bash
docker compose up -d
```
*Wait about 20 seconds. This starts all the background engines.*

### Step 3: Initialize the System
Now we need to create the tables inside those databases where our transaction data will live.
```bash
# 1. Create the Indian banking tables in the database
psql -h localhost -U sentinel -d sentinel_db -f data_warehouse/schemas/migrate_002_txn_redesign.sql
# (If it asks for a password, type: sentinel2024)

# 2. Setup the Kafka streaming channels
python -m scripts.init_kafka_topics

# 3. Setup the AWS DynamoDB local tables (for the dashboard)
python -m scripts.init_dynamodb
```

---

## 🧠 Step 4: Train the Machine Learning Model
An AI needs to learn before it can predict! We provide simulated historical data to teach the AI what a "safe" customer looks like versus an "at-risk" customer.

1. **Build the Training Data**: This looks at past transactions and extracts "behavioral features" (like how much someone's savings dropped).
   ```bash
   python -m models.training_pipelines.build_training_data
   ```

2. **Train the AI Model**: This feeds the data into our LightGBM (Gradient Boosting) AI engine so it learns the patterns.
   ```bash
   python -m models.lightgbm.train_lgbm
   ```
*You now have a fully trained, highly intelligent AI model saved on your computer!*

---

## ⚙️ Step 5: Start the Real-Time Brain (Feature Pipeline)
This pipeline sits and listens. The moment a transaction happens, this pipeline grabs it, calculates the customer's new behavioral data, and asks the AI to generate a new Pulse Score.

In your terminal, run:
```bash
python -m ingestion.consumers.feature_pipeline
```
**Leave this terminal window open!** It will sit there waiting for transactions.

---

## 🔌 Step 6: Start the Backend API
This is the bridge between the AI and the visual Dashboard. 

Open a **NEW Terminal Window** (keep the previous one running), navigate to the `sentinel-ai` folder, and type:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
```
**Leave this window open!**

---

## 📊 Step 7: Start the Visual Dashboard
Let's turn on the beautiful screen you will actually look at.

Open a **NEW Terminal Window**, go to the `sentinel-ai` folder, and navigate into the `dashboards` folder:
```bash
cd dashboards

# Install the UI packages (only needed the very first time)
npm install

# Start the dashboard website
npm run dev
```
**Leave this window open!**
Now, open your web browser (like Chrome or Safari) and go to: **[http://localhost:3000](http://localhost:3000)**

*(If you are asked to log in, Sentinel uses secure database logins. Use the administrator credentials provided by your system admin).*

---

## 💳 Step 8: Run the Transaction Simulator! (The Magic!)
Right now, your dashboard is probably empty or static because nobody is swiping their debit cards. Let's create some fake Indian customers and generate real-time transactions to watch the AI work!

Open one final **NEW Terminal Window** and navigate to your `sentinel-ai` folder. 

Run the simulator:
```bash
python -m sagemaker.inference
```

### What is the Simulator doing?
1. **Creates Fake Customers**: It generates realistic Indian customers (like an IT Professional from Bangalore, or a Shopkeeper from Jaipur).
2. **Assigns Finances**: It gives them a realistic salary, a Home/Car loan, and a savings account.
3. **Simulates Transactions**: It rapidly fakes them buying food on *Swiggy*, paying electricity on *BESCOM*, or withdrawing cash from an *SBI ATM*.
4. **Feeds the AI**: Every transaction is sent into the system. If the simulator decides a customer is going through financial stress (e.g., they empty their savings account and max out their credit card), you will see their **Pulse Score plummet in real-time on your Dashboard!**

---

### 🎉 Congratulations!
You are now running a sophisticated, real-time Machine Learning financial pipeline perfectly on your own computer. You can watch the Dashboard metrics change dynamically as the simulator generates transactions!
