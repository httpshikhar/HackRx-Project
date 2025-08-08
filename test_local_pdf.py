"""
Test script for the local Arogya Sanjeevani Policy PDF

This script demonstrates how to use the system with a local PDF file.
"""

import asyncio
import os
from dotenv import load_dotenv
from document_intelligence import DocumentIntelligenceSystem

# Load environment variables
load_dotenv()


async def test_local_pdf():
    """Test the document intelligence system with the local PDF."""
    
    print("🏥 Arogya Sanjeevani Policy - Document Intelligence Test")
    print("=" * 65)
    
    # Initialize the system
    try:
        system = DocumentIntelligenceSystem()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Use the local PDF file
    pdf_path = "/home/shikhar/Desktop/HackRx/Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"\n📄 Processing local PDF: {os.path.basename(pdf_path)}")
    print("⏳ This will take a few moments...")
    
    # Process the document
    try:
        result = await system.process_document(pdf_path)
        
        if result["success"]:
            print("✅ Document processed successfully!")
            print(f"   📊 Document ID: {result['document_id'][:12]}...")
            print(f"   📝 Text chunks: {result['chunks_processed']}")
            print(f"   📏 Total characters: {result['total_characters']:,}")
        else:
            print(f"❌ Failed to process document: {result['error']}")
            return
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return
    
    print("\n" + "=" * 65)
    print("🔍 Testing health insurance policy queries...")
    
    # Health insurance specific questions
    insurance_questions = [
        "What is the coverage amount of this policy?",
        "What are the key benefits provided?",
        "What is the waiting period for pre-existing diseases?",
        "What are the exclusions in this policy?",
        "What is the premium amount and payment terms?",
        "Who is eligible for this policy?",
        "What is the claim process?",
    ]
    
    for i, question in enumerate(insurance_questions, 1):
        print(f"\n📋 Query {i}: {question}")
        print("⏳ Searching for answer...")
        
        try:
            query_result = await system.query_document(question, top_k=3)
            
            if query_result["success"]:
                print(f"✅ Found answer with {query_result.get('confidence', 0):.3f} confidence")
                print(f"\n💬 Answer:")
                print(f"   {query_result['answer']}")
                
                source_clauses = query_result.get('source_clauses', [])
                if source_clauses:
                    print(f"\n📚 Top source clauses:")
                    for j, clause in enumerate(source_clauses[:2], 1):
                        print(f"   {j}. Similarity: {clause['similarity_score']:.3f}")
                        print(f"      Text: {clause['text'][:120]}...")
            else:
                print(f"❌ Query failed: {query_result['error']}")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("-" * 50)
    
    # Interactive query section
    print("\n" + "=" * 65)
    print("🎯 Interactive Query - Ask about the policy!")
    
    while True:
        custom_question = input("\n❓ Enter your question about the Arogya Sanjeevani Policy (or 'quit' to exit): ")
        
        if custom_question.lower() in ['quit', 'exit', 'q']:
            break
            
        if custom_question.strip():
            print(f"\n🤔 Processing: '{custom_question}'")
            
            try:
                custom_result = await system.query_document(custom_question, top_k=5)
                
                if custom_result["success"]:
                    print(f"\n💡 Answer (confidence: {custom_result.get('confidence', 0):.3f}):")
                    print(f"{custom_result['answer']}")
                    
                    source_clauses = custom_result.get('source_clauses', [])
                    print(f"\n📖 Based on {len(source_clauses)} source clauses")
                    
                    # Show source preview
                    if source_clauses:
                        print("\n📋 Key sources:")
                        for i, clause in enumerate(source_clauses[:2], 1):
                            print(f"   {i}. {clause['text'][:100]}...")
                else:
                    print(f"❌ Failed: {custom_result['error']}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\n" + "=" * 65)
    print("🏁 Test completed!")
    
    # Cleanup option
    cleanup = input("\n🧹 Delete test document from vector database? (y/n): ")
    if cleanup.lower() in ['y', 'yes']:
        try:
            delete_result = await system.delete_document(pdf_path)
            if delete_result["success"]:
                print("✅ Test document deleted from vector database")
            else:
                print(f"❌ Failed to delete: {delete_result['error']}")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    print("\n💡 Your Arogya Sanjeevani Policy has been successfully analyzed!")
    print("   The system can now answer questions about coverage, benefits, exclusions, and more.")


def check_environment():
    """Check if all required environment variables are set."""
    
    print("🔧 Checking environment setup...")
    
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    else:
        print("✅ All required environment variables are set")
        print(f"🔗 Azure OpenAI Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        print(f"🔗 Pinecone Index: {os.getenv('PINECONE_INDEX_NAME')}")
        print(f"🤖 Chat Model: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')} (API: {os.getenv('AZURE_OPENAI_O3_API_VERSION')})")
        print(f"🔤 Embedding Model: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')} (API: {os.getenv('AZURE_OPENAI_EMBED_API_VERSION')})")
        return True


if __name__ == "__main__":
    print("🧪 Document Intelligence System - Local PDF Test")
    print("=" * 65)
    
    # Check environment setup
    if not check_environment():
        exit(1)
    
    # Run the test
    try:
        asyncio.run(test_local_pdf())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please check your configuration and try again")
