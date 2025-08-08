"""
Simple Example Usage of Document Intelligence System

This script demonstrates the basic functionality with a real example.
Make sure to set up your .env file before running this script.
"""

import asyncio
import os
from dotenv import load_dotenv
from document_intelligence import DocumentIntelligenceSystem

# Load environment variables from .env file
load_dotenv()


async def main():
    """Main example function demonstrating the system usage."""
    
    print("🤖 Document Intelligence System - Example Usage")
    print("=" * 55)
    
    # Initialize the system
    try:
        system = DocumentIntelligenceSystem()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("\n💡 Make sure you have configured your .env file properly!")
        return
    
    # Example: Process a research paper from arXiv
    # You can replace this with any public PDF URL
    pdf_url = "https://arxiv.org/pdf/2301.00234.pdf"
    
    print(f"\n📄 Processing document: {pdf_url}")
    print("⏳ This will take a few moments...")
    
    # Step 1: Process the document
    try:
        result = await system.process_document(pdf_url)
        
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
    
    print("\n" + "=" * 55)
    print("🔍 Now let's ask some questions about the document...")
    
    # Step 2: Ask questions about the document
    questions = [
        "What is the main topic of this research paper?",
        "What methodology or approach is used?",
        "What are the key findings or results?",
        "What are the limitations mentioned in the paper?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("⏳ Searching for answer...")
        
        try:
            query_result = await system.query_document(question, top_k=3)
            
            if query_result["success"]:
                print(f"✅ Found answer with {query_result['confidence']:.2f} confidence")
                print(f"\n💬 Answer:")
                print(f"   {query_result['answer']}")
                
                print(f"\n📚 Sources ({len(query_result['source_clauses'])} relevant clauses):")
                for j, clause in enumerate(query_result['source_clauses'][:2], 1):
                    print(f"   {j}. Similarity: {clause['similarity_score']:.3f}")
                    print(f"      Text preview: {clause['text'][:120]}...")
            else:
                print(f"❌ Query failed: {query_result['error']}")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("-" * 40)
    
    # Step 3: Demonstrate custom query
    print("\n" + "=" * 55)
    print("🎯 Custom Query Demo")
    
    custom_question = input("Enter your own question about the document: ")
    if custom_question.strip():
        print(f"\n🤔 Processing your question: '{custom_question}'")
        
        try:
            custom_result = await system.query_document(custom_question, top_k=5)
            
            if custom_result["success"]:
                print(f"\n💡 Answer (confidence: {custom_result['confidence']:.3f}):")
                print(custom_result['answer'])
                
                print(f"\n📖 Based on {len(custom_result['source_clauses'])} source clauses")
            else:
                print(f"❌ Failed to process your question: {custom_result['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 55)
    print("🏁 Example completed!")
    
    # Optional: Clean up
    cleanup = input("\n🧹 Would you like to delete the test document from the vector database? (y/n): ")
    if cleanup.lower() in ['y', 'yes']:
        try:
            delete_result = await system.delete_document(pdf_url)
            if delete_result["success"]:
                print("✅ Test document deleted from vector database")
            else:
                print(f"❌ Failed to delete: {delete_result['error']}")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    print("\n💡 You can now use this system with your own PDF documents!")
    print("   Just replace the pdf_url with your document URL and run similar queries.")


async def simple_api_example():
    """
    Simple example showing the most basic usage of the API.
    Uncomment and run this instead of main() for a simpler demo.
    """
    from document_intelligence import process_document, query_document
    
    # Process a document
    pdf_url = "https://arxiv.org/pdf/2301.00234.pdf"
    
    print("Processing document...")
    result = await process_document(pdf_url)
    
    if result["success"]:
        print(f"✅ Processed {result['chunks_processed']} chunks")
        
        # Query the document
        print("Querying document...")
        answer = await query_document("What is this document about?")
        
        if answer["success"]:
            print(f"Answer: {answer['answer']}")
        else:
            print(f"Query failed: {answer['error']}")
    else:
        print(f"Processing failed: {result['error']}")


if __name__ == "__main__":
    # Check if environment variables are set
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "PINECONE_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set up your .env file using the .env.template as a guide")
        print("   Then run this script again.")
    else:
        # Run the main example
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n⏹️  Example interrupted by user")
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            print("Please check your configuration and try again")
