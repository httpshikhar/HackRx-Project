# Document Intelligence System

A Python-based document intelligence system that processes PDF documents, generates embeddings using Azure OpenAI, stores them in Pinecone vector database, and provides semantic search with AI-powered question answering.

## 🚀 Features

- **PDF Processing**: Extract and intelligently chunk PDF documents into meaningful clauses
- **Vector Embeddings**: Generate embeddings using Azure OpenAI embedding models
- **Vector Storage**: Store embeddings in Pinecone vector database with metadata
- **Semantic Search**: Retrieve relevant document clauses based on semantic similarity
- **AI Question Answering**: Generate comprehensive answers using Azure OpenAI chat completion
- **Async Operations**: Efficient async processing for better performance
- **Error Handling**: Robust error handling and logging

## 📋 Prerequisites

### Azure OpenAI Setup
1. **Azure OpenAI Service**: Create an Azure OpenAI service in the Azure Portal
2. **Model Deployments**: Deploy the following models in Azure OpenAI Studio:
   - `text-embedding-ada-002` (for embeddings)
   - `gpt-4` or `gpt-35-turbo` (for chat completion)
3. **API Keys**: Get your API key and endpoint from Azure Portal

### Pinecone Setup
1. **Pinecone Account**: Create a free account at [pinecone.io](https://pinecone.io)
2. **Create Index**: Create a new index with the following specifications:
   - **Name**: `document-intelligence` (or your preferred name)
   - **Dimensions**: `1536` (for text-embedding-ada-002)
   - **Metric**: `cosine`
   - **Cloud Provider**: Any (AWS recommended)
3. **API Key**: Get your API key from the Pinecone console

## 🛠️ Installation

### 1. Clone or Download the Code
```bash
# If you have the code in a repository
git clone <repository-url>
cd document-intelligence

# Or create the directory and add the files manually
mkdir document-intelligence
cd document-intelligence
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Also install python-dotenv for environment variable loading
pip install python-dotenv
```

### 4. Environment Setup
```bash
# Copy the environment template
cp .env.template .env

# Edit the .env file with your actual credentials
nano .env  # or your preferred editor
```

Fill in your actual values in the `.env` file:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure OpenAI Model Deployments (use your deployment names)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4

# Pinecone Configuration
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_INDEX_NAME=document-intelligence
```

## 🧪 Testing

Run the test script to verify everything is working:

```bash
python test_system.py
```

This will:
1. Check your environment setup
2. Initialize the system
3. Process a sample PDF document
4. Run test queries
5. Display results with confidence scores and source clauses

## 💻 Usage

### Basic Usage

```python
import asyncio
from document_intelligence import DocumentIntelligenceSystem

async def main():
    # Initialize system
    system = DocumentIntelligenceSystem()
    
    # Process a document
    pdf_url = "https://example.com/document.pdf"
    result = await system.process_document(pdf_url)
    
    if result["success"]:
        print(f"Document processed: {result['chunks_processed']} chunks created")
        
        # Query the document
        query_result = await system.query_document(
            "What are the main points in this document?", 
            top_k=5
        )
        
        if query_result["success"]:
            print(f"Answer: {query_result['answer']}")
            print(f"Confidence: {query_result['confidence']:.3f}")
            
            # Show source clauses
            for i, clause in enumerate(query_result['source_clauses']):
                print(f"Source {i+1}: {clause['text'][:100]}...")

# Run the async function
asyncio.run(main())
```

### Convenience Functions

For simpler usage, you can use the convenience functions:

```python
import asyncio
from document_intelligence import process_document, query_document

async def simple_usage():
    # Process document
    result = await process_document("https://example.com/document.pdf")
    
    # Query document
    answer = await query_document("What is this document about?")
    print(answer['answer'])

asyncio.run(simple_usage())
```

### API Reference

#### `DocumentIntelligenceSystem` Class

##### `process_document(pdf_url: str) -> Dict[str, Any]`
Process a PDF document and store embeddings in Pinecone.

**Parameters:**
- `pdf_url`: URL of the PDF document to process

**Returns:**
- Dictionary with processing results including success status, document ID, and chunk count

##### `query_document(question: str, top_k: int = 5) -> Dict[str, Any]`
Query the document database and generate an answer.

**Parameters:**
- `question`: The question to ask
- `top_k`: Number of relevant clauses to retrieve (default: 5)

**Returns:**
- Dictionary with answer, source clauses, and confidence score

##### `delete_document(pdf_url: str) -> Dict[str, Any]`
Delete all vectors for a specific document from Pinecone.

**Parameters:**
- `pdf_url`: URL of the document to delete

**Returns:**
- Dictionary with deletion status

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Yes |
| `AZURE_OPENAI_API_VERSION` | API version (default: 2024-02-01) | No |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name | No |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Chat model deployment name | No |
| `PINECONE_API_KEY` | Pinecone API key | Yes |
| `PINECONE_INDEX_NAME` | Pinecone index name | Yes |

### Chunking Configuration

You can modify the text chunking behavior by adjusting parameters in the `_chunk_text_into_clauses` method:

- `max_chunk_size`: Maximum characters per chunk (default: 1000)
- Minimum chunk size: 50 characters (filters out very short chunks)

### Model Configuration

The system uses these models by default:
- **Embedding Model**: `text-embedding-ada-002` (1536 dimensions)
- **Chat Model**: `gpt-4` (can use `gpt-35-turbo` for faster/cheaper responses)

## 🚨 Troubleshooting

### Common Issues

1. **"Could not connect to Pinecone index"**
   - Ensure your Pinecone index exists and is active
   - Check your API key and index name
   - Verify index dimensions match embedding model (1536 for ada-002)

2. **"Failed to generate embedding"**
   - Check Azure OpenAI API key and endpoint
   - Verify your embedding model deployment is active
   - Check API quota and rate limits

3. **"No text extracted from PDF"**
   - PDF might be image-based (requires OCR)
   - PDF might be password-protected
   - Try with a different PDF

4. **Rate limiting errors**
   - Azure OpenAI has rate limits - the system will retry automatically
   - Consider reducing batch sizes for large documents

### Debug Mode

Enable debug logging for more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Considerations

- **Embedding Generation**: Each chunk requires an API call to Azure OpenAI
- **Batch Processing**: Vectors are uploaded to Pinecone in batches of 100
- **Async Operations**: All I/O operations are async for better performance
- **Memory Usage**: Large PDFs are processed chunk by chunk to manage memory

## 🔐 Security

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use `.env` files that are gitignored
- **HTTPS**: All API calls use HTTPS encryption
- **Input Validation**: URLs and inputs are validated before processing

## 📚 Example Use Cases

1. **Legal Document Analysis**: Process contracts and legal documents for clause extraction
2. **Research Paper Review**: Analyze academic papers and answer research questions  
3. **Technical Documentation**: Create searchable knowledge base from manuals and guides
4. **Compliance Checking**: Search through policy documents for specific requirements
5. **Due Diligence**: Analyze business documents for key information extraction

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## 📄 License

This project is provided as-is for educational and development purposes.

---

**Need Help?** Check the troubleshooting section or create an issue with details about your problem.
