# Railway GitHub Deployment Guide

## 🚀 Deploy HackRx Document Intelligence from GitHub

Your code has been successfully pushed to GitHub: https://github.com/httpshikhar/HackRx-Project

### Step 1: Create Railway Project from GitHub

1. **Go to Railway Dashboard**: https://railway.app/dashboard
2. **Create New Project**: Click "New Project"
3. **Choose "Deploy from GitHub repo"**
4. **Select Repository**: Choose `httpshikhar/HackRx-Project`
5. **Railway will automatically detect your `Dockerfile`** ✅

### Step 2: Add PostgreSQL Database

1. **In your Railway project dashboard**, click "Add Service"
2. **Select "Database"** → **"PostgreSQL"**
3. **Railway will automatically provide `DATABASE_URL`** environment variable

### Step 3: Configure Environment Variables

In Railway dashboard, go to your service → **Variables** tab and add:

```bash
# API Configuration
API_BEARER_TOKEN=hackrx2024_shikhar_secure_token
PORT=8000
ENVIRONMENT=production

# Azure OpenAI Configuration (REQUIRED)
AZURE_OPENAI_API_KEY=your_actual_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_O3_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBED_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=o3-mini

# Pinecone Configuration (REQUIRED)
PINECONE_API_KEY=your_actual_pinecone_key
PINECONE_INDEX_NAME=document-intelligence

# PostgreSQL (AUTO-PROVIDED by Railway)
# DATABASE_URL=postgresql://... (automatically set)
```

### Step 4: Deploy

1. **Railway will automatically trigger deployment** when you push to GitHub
2. **Your app will be available at**: `https://your-app-name.up.railway.app`
3. **Health check endpoint**: `https://your-app-name.up.railway.app/health`
4. **Hackathon API endpoint**: `https://your-app-name.up.railway.app/hackrx/run`

### Step 5: Test Your Deployment

```bash
# Health Check
curl https://your-app-name.up.railway.app/health

# Test API (replace with your actual URL and bearer token)
curl -X POST https://your-app-name.up.railway.app/hackrx/run \
  -H "Authorization: Bearer hackrx2024_shikhar_secure_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": ["What is the main topic of this document?"]
  }'
```

## 🔧 Configuration Details

### Automatic Features
- ✅ **Dockerfile detected** - Railway uses your optimized Dockerfile
- ✅ **Health checks** configured via `railway.json`
- ✅ **PostgreSQL** database with automatic connection
- ✅ **SSL/TLS** encryption for HTTPS
- ✅ **Auto-scaling** and monitoring

### Environment Variables Required
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint
- `PINECONE_API_KEY` - Your Pinecone API key

### Environment Variables Auto-Set
- `DATABASE_URL` - PostgreSQL connection string (auto-generated)
- `RAILWAY_ENVIRONMENT` - Set to "production"
- `PORT` - Set to 8000

## 🚨 Security Features

- ✅ **No secrets in code** - All API keys are environment variables
- ✅ **GitHub secrets scanning** passed
- ✅ **Bearer token authentication** 
- ✅ **Input validation** and rate limiting
- ✅ **Secure headers** and CORS configuration

## 📊 Monitoring & Logs

### View Logs
```bash
# If you have Railway CLI installed
railway logs

# Or check Railway dashboard → Your Service → Logs tab
```

### Performance Monitoring
- Railway provides built-in metrics
- View CPU, memory, and network usage
- Monitor response times and error rates

## 🛠️ Advanced Configuration

### Custom Domain (Optional)
1. Go to Railway dashboard → Your service → Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed

### Database Management
- Railway provides a PostgreSQL web interface
- Access via Railway dashboard → PostgreSQL service → Data tab
- Run SQL queries, view tables, manage data

## 📞 Support & Troubleshooting

### Common Issues

1. **App not starting**: Check environment variables are set
2. **Database connection errors**: Ensure PostgreSQL service is running
3. **API key errors**: Verify Azure OpenAI and Pinecone keys are correct

### Get Help
- Railway Docs: https://docs.railway.app/
- Railway Discord: https://railway.app/discord
- GitHub Issues: https://github.com/httpshikhar/HackRx-Project/issues

---

## 🎯 Ready for Hackathon!

Once deployed, your API endpoint will be:
**`https://your-app-name.up.railway.app/hackrx/run`**

This endpoint is ready for:
- ✅ Document processing from URLs
- ✅ Multi-question answering
- ✅ Advanced o3-mini conflict detection
- ✅ PostgreSQL logging and caching
- ✅ Production-grade performance

**Submit this URL to your hackathon platform!** 🏆
