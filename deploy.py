#!/usr/bin/env python3
"""
Railway Deployment Script for HackRx Document Intelligence API

This script helps automate the deployment process to Railway.
"""

import os
import sys
import json
import subprocess
import time
from typing import Dict, Any

def check_railway_cli():
    """Check if Railway CLI is installed."""
    try:
        result = subprocess.run(['railway', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Railway CLI found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Railway CLI not found. Please install it:")
    print("   npm install -g @railway/cli")
    print("   or visit: https://docs.railway.app/develop/cli")
    return False

def load_env_vars():
    """Load environment variables from .env file."""
    env_vars = {}
    
    # Try to load from .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars

def validate_required_vars(env_vars: Dict[str, str]) -> bool:
    """Validate that all required environment variables are present."""
    required_vars = [
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME',
    ]
    
    missing_vars = []
    for var in required_vars:
        if not env_vars.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease update your .env file with the correct values.")
        return False
    
    print("✅ All required environment variables are present")
    return True

def create_railway_project():
    """Create a new Railway project."""
    print("\n🚂 Creating Railway project...")
    
    try:
        # Login check
        result = subprocess.run(['railway', 'whoami'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Please login to Railway first:")
            subprocess.run(['railway', 'login'])
        
        # Create project
        result = subprocess.run([
            'railway', 'init', 
            '--name', 'hackrx-document-intelligence'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Railway project created successfully")
            return True
        else:
            print(f"❌ Failed to create Railway project: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating Railway project: {e}")
        return False

def add_postgresql_service():
    """Add PostgreSQL service to Railway project."""
    print("\n🐘 Adding PostgreSQL service...")
    
    try:
        result = subprocess.run([
            'railway', 'add', 'postgresql'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL service added successfully")
            print("   DATABASE_URL will be automatically provided")
            return True
        else:
            print(f"⚠️  Could not add PostgreSQL automatically: {result.stderr}")
            print("   Please add PostgreSQL manually from Railway dashboard")
            return False
            
    except Exception as e:
        print(f"❌ Error adding PostgreSQL: {e}")
        return False

def set_environment_variables(env_vars: Dict[str, str]):
    """Set environment variables in Railway."""
    print("\n🔧 Setting environment variables...")
    
    # Environment variables to set
    vars_to_set = {
        'API_BEARER_TOKEN': env_vars.get('API_BEARER_TOKEN', 'hackrx2024_shikhar_secure_token'),
        'AZURE_OPENAI_API_KEY': env_vars.get('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_ENDPOINT': env_vars.get('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_O3_API_VERSION': env_vars.get('AZURE_OPENAI_O3_API_VERSION', '2024-12-01-preview'),
        'AZURE_OPENAI_EMBED_API_VERSION': env_vars.get('AZURE_OPENAI_EMBED_API_VERSION', '2024-02-01'),
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT': env_vars.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002'),
        'AZURE_OPENAI_CHAT_DEPLOYMENT': env_vars.get('AZURE_OPENAI_CHAT_DEPLOYMENT', 'o3-mini'),
        'PINECONE_API_KEY': env_vars.get('PINECONE_API_KEY'),
        'PINECONE_INDEX_NAME': env_vars.get('PINECONE_INDEX_NAME'),
        'PORT': '8000',
        'ENVIRONMENT': 'production',
        'LOG_LEVEL': 'INFO',
    }
    
    success_count = 0
    for key, value in vars_to_set.items():
        if value and not value.startswith('your_'):  # Skip template values
            try:
                result = subprocess.run([
                    'railway', 'variables', '--set', f'{key}={value}'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Set {key}")
                    success_count += 1
                else:
                    print(f"❌ Failed to set {key}: {result.stderr}")
            except Exception as e:
                print(f"❌ Error setting {key}: {e}")
        else:
            print(f"⚠️  Skipped {key} (no value provided or template value)")
    
    print(f"\n✅ Successfully set {success_count}/{len(vars_to_set)} environment variables")
    return success_count > 0

def deploy_application():
    """Deploy the application to Railway."""
    print("\n🚀 Deploying application...")
    
    try:
        # Deploy
        result = subprocess.run([
            'railway', 'up', '--detach'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Deployment initiated successfully")
            print("   Deployment is running in the background...")
            return True
        else:
            print(f"❌ Deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during deployment: {e}")
        return False

def get_deployment_url():
    """Get the deployed application URL."""
    print("\n🔗 Getting deployment URL...")
    
    try:
        # Wait a bit for deployment to process
        print("   Waiting for deployment to complete...")
        time.sleep(10)
        
        result = subprocess.run([
            'railway', 'domain'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            url = result.stdout.strip()
            if url:
                print(f"✅ Deployment URL: {url}")
                print(f"   Health check: {url}/health")
                print(f"   API endpoint: {url}/hackrx/run")
                return url
        
        print("⚠️  Could not retrieve URL automatically")
        print("   Check Railway dashboard for the deployment URL")
        return None
        
    except Exception as e:
        print(f"❌ Error getting deployment URL: {e}")
        return None

def main():
    """Main deployment workflow."""
    print("🏗️  HackRx Document Intelligence - Railway Deployment")
    print("=" * 60)
    
    # Check Railway CLI
    if not check_railway_cli():
        return 1
    
    # Load environment variables
    print("\n📋 Loading environment configuration...")
    env_vars = load_env_vars()
    
    # Validate environment variables
    if not validate_required_vars(env_vars):
        return 1
    
    # Create Railway project
    if not create_railway_project():
        print("❌ Failed to create Railway project")
        return 1
    
    # Add PostgreSQL service
    add_postgresql_service()  # Optional - continue even if it fails
    
    # Set environment variables
    if not set_environment_variables(env_vars):
        print("❌ Failed to set environment variables")
        return 1
    
    # Deploy application
    if not deploy_application():
        print("❌ Deployment failed")
        return 1
    
    # Get deployment URL
    deployment_url = get_deployment_url()
    
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    if deployment_url:
        print(f"\n🌐 Your API is live at: {deployment_url}")
        print(f"📊 Health check: {deployment_url}/health")
        print(f"🎯 Hackathon endpoint: {deployment_url}/hackrx/run")
        print(f"📚 API documentation: {deployment_url}/docs")
    
    print("\n📝 Next steps:")
    print("   1. Test your deployment with the health endpoint")
    print("   2. Submit your webhook URL to the hackathon platform")
    print("   3. Monitor your deployment in Railway dashboard")
    
    print("\n🔧 Useful Railway commands:")
    print("   railway logs        - View application logs")
    print("   railway status      - Check deployment status")
    print("   railway variables   - Manage environment variables")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
