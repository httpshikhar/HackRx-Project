# 🚂 Railway Deployment Guide for HackRx Document Intelligence API

This guide provides step-by-step instructions for deploying your HackRx Document Intelligence API to Railway.

## 📋 Prerequisites

### 1. Railway Account
- Sign up at [railway.app](https://railway.app)
- Connect your GitHub account (recommended)

### 2. Railway CLI
Install the Railway CLI:
```bash
npm install -g @railway/cli
```

Or using other methods:
```bash
# Using curl
curl -fsSL https://railway.app/install.sh | sh

# Using brew (macOS)
brew install railway
```

### 3. Environment Variables
Ensure your `.env` file contains all required variables:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

## 🚀 Automated Deployment (Recommended)

### Option 1: Using the Deployment Script

1. **Run the automated deployment script:**
```bash
python deploy.py
```

The script will:
- ✅ Check Railway CLI installation
- ✅ Validate environment variables
- ✅ Create Railway project
- ✅ Add PostgreSQL service
- ✅ Set environment variables
- ✅ Deploy the application
- ✅ Provide deployment URL

### Option 2: Using Railway CLI Directly

1. **Login to Railway:**
```bash
railway login
```

2. **Initialize project:**
```bash
railway init --name hackrx-document-intelligence
```

3. **Add PostgreSQL service:**
```bash
railway add postgresql
```

4. **Set environment variables:**
```bash
# API Authentication
railway variables set API_BEARER_TOKEN=hackrx2024_shikhar_secure_token

# Azure OpenAI
railway variables set AZURE_OPENAI_API_KEY=your_actual_key_here
railway variables set AZURE_OPENAI_ENDPOINT=your_actual_endpoint_here
railway variables set AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Pinecone
railway variables set PINECONE_API_KEY=your_actual_key_here
railway variables set PINECONE_INDEX_NAME=hackrx-documents

# Application settings
railway variables set PORT=8000
railway variables set ENVIRONMENT=production
railway variables set LOG_LEVEL=INFO
```

5. **Deploy:**
```bash
railway up
```

## 🌐 Manual Railway Dashboard Deployment

### 1. Create New Project
1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Choose "Deploy from GitHub repo"
4. Connect your GitHub account and select this repository

### 2. Add PostgreSQL Database
1. In your project dashboard, click "New Service"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically provide `DATABASE_URL`

### 3. Configure Environment Variables
In the project settings, add these environment variables:

```env
API_BEARER_TOKEN=hackrx2024_shikhar_secure_token
AZURE_OPENAI_API_KEY=your_actual_key_here
AZURE_OPENAI_ENDPOINT=your_actual_endpoint_here
AZURE_OPENAI_API_VERSION=2024-12-01-preview
PINECONE_API_KEY=your_actual_key_here
PINECONE_INDEX_NAME=hackrx-documents
PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 4. Deploy
Railway will automatically deploy when you push to your connected GitHub repository.

## 🧪 Testing Your Deployment

### 1. Health Check
Once deployed, test the health endpoint:
```bash
curl https://your-app-url.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-08T12:00:00.000Z",
  "database_connected": true,
  "system_stats": {...}
}
```

### 2. API Documentation
Visit your deployment URL + `/docs` to access the interactive API documentation:
```
https://your-app-url.railway.app/docs
```

### 3. Test the Main Endpoint
Test the hackathon endpoint:
```bash
curl -X POST "https://your-app-url.railway.app/hackrx/run" \
  -H "Authorization: Bearer hackrx2024_shikhar_secure_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

## 📊 Monitoring Your Deployment

### View Logs
```bash
railway logs
```

### Check Status
```bash
railway status
```

### View Environment Variables
```bash
railway variables
```

## 🔧 Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   - Verify all required variables are set in Railway dashboard
   - Check for typos in variable names

2. **Database Connection Issues**
   - Ensure PostgreSQL service is added and running
   - Check that `DATABASE_URL` is automatically provided

3. **Azure OpenAI API Issues**
   - Verify your Azure OpenAI credentials
   - Check endpoint URL format
   - Ensure API version is correct

4. **Pinecone Connection Issues**
   - Verify Pinecone API key
   - Ensure index exists with correct name
   - Check Pinecone region settings

### Debugging Commands

```bash
# View detailed logs
railway logs --tail

# Connect to your service shell
railway shell

# View all environment variables
railway variables list

# Restart your service
railway redeploy
```

## 🎯 Submitting to Hackathon

### 1. Get Your Webhook URL
Your hackathon endpoint will be:
```
https://your-app-name.railway.app/hackrx/run
```

### 2. Test Before Submission
Use the comprehensive test request:
```bash
curl -X POST "https://your-app-name.railway.app/hackrx/run" \
  -H "Authorization: Bearer hackrx2024_shikhar_secure_token" \
  -H "Content-Type: application/json" \
  -d @comprehensive_test_request.json
```

### 3. Submit to Platform
Enter your Railway deployment URL in the hackathon submission form.

## 📈 Performance Optimization

### 1. Monitor Resource Usage
- Check Railway dashboard for CPU/Memory usage
- Scale up if needed for high traffic

### 2. Database Optimization
- Monitor PostgreSQL performance
- Consider upgrading database plan for production

### 3. Application Settings
Adjust these environment variables for better performance:
```env
MAX_CONCURRENT_REQUESTS=20
QUERY_TIMEOUT_SECONDS=300
CACHE_TTL_HOURS=48
```

## 🆘 Support

- **Railway Documentation**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Railway Support**: support@railway.app

## ✅ Deployment Checklist

- [ ] Railway CLI installed and logged in
- [ ] All environment variables configured
- [ ] PostgreSQL service added
- [ ] Application deployed successfully
- [ ] Health check endpoint working
- [ ] Main API endpoint tested
- [ ] Documentation accessible
- [ ] Webhook URL submitted to hackathon platform

---

🎉 **Your HackRx Document Intelligence API is now live on Railway!**
