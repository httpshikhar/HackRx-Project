"""
Test script for the Document Intelligence System

This script demonstrates how to use the system and provides basic verification.
"""

import asyncio
import os
from dotenv import load_dotenv
from document_intelligence import DocumentIntelligenceSystem, process_document, query_document

# Load environment variables
load_dotenv()


async def test_system():
    """Test the document intelligence system with a sample PDF."""
    
    print("🚀 Starting Document Intelligence System Test")
    print("=" * 60)
    
    # Initialize the system
    try:
        system = DocumentIntelligenceSystem()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Set up your .env file with API keys")
        print("   2. Created a Pinecone index")
        print("   3. Deployed models in Azure OpenAI")
        return
    
    # Test with a sample PDF (you can replace this with any public PDF URL)
    # This is a sample research paper from arXiv
    sample_pdf_url = "https://arxiv.org/pdf/2301.00234.pdf"
    
    print(f"\n📄 Processing document: {sample_pdf_url}")
    print("⏳ This may take a few minutes...")
    
    # Process the document
    try:
        result = await system.process_document(sample_pdf_url)
        
        if result["success"]:
            print("✅ Document processed successfully!")
            print(f"   📊 Document ID: {result['document_id']}")
            print(f"   📝 Chunks created: {result['chunks_processed']}")
            print(f"   📏 Total characters: {result['total_characters']}")
        else:
            print(f"❌ Failed to process document: {result['error']}")
            return
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🔍 Testing queries...")
    
    # Test queries
    test_questions = [
        "What is this document about?",
        "What are the main findings or conclusions?",
        "What methodology was used in this research?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Query {i}: {question}")
        print("⏳ Processing query...")
        
        try:
            query_result = await system.query_document(question, top_k=3)
            
            if query_result["success"]:
                print(f"✅ Query processed successfully!")
                print(f"🎯 Confidence: {query_result.get('confidence', 0):.3f}")
                print(f"📚 Sources found: {len(query_result.get('source_clauses', []))}")
                print(f"\n💬 Answer: {query_result['answer']}")
                
                print(f"\n📋 Top source clauses:")
                for j, clause in enumerate(query_result['source_clauses'][:2], 1):
                    print(f"   {j}. Similarity: {clause['similarity_score']:.3f}")
                    print(f"      Preview: {clause['text'][:150]}...")
            else:
                print(f"❌ Query failed: {query_result['error']}")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    
    # Optional: Clean up (delete the test document)
    cleanup = input("\n🧹 Delete test document from vector database? (y/n): ")
    if cleanup.lower() == 'y':
        try:
            delete_result = await system.delete_document(sample_pdf_url)
            if delete_result["success"]:
                print("✅ Document deleted successfully!")
            else:
                print(f"❌ Failed to delete document: {delete_result['error']}")
        except Exception as e:
            print(f"❌ Error deleting document: {e}")


def test_environment_setup():
    """Test if all required environment variables are set."""
    
    print("🔧 Checking environment setup...")
    
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set up your .env file using .env.template as a guide")
        return False
    else:
        print("✅ All required environment variables are set")
        return True


if __name__ == "__main__":
    print("🧪 Document Intelligence System - Test Suite")
    print("=" * 60)
    
    # First check environment setup
    if not test_environment_setup():
        exit(1)
    
    # Run the main test
    try:
        asyncio.run(test_system())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during test: {e}")
        print("Please check your configuration and try again")
