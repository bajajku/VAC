# Data Cleaning Module

This module provides intelligent data cleaning capabilities using LLM-powered chains to process unstructured data and convert it into clean, relevant documents suitable for RAG systems.

## Overview

The `DataCleaner` class uses a `NormalChain` with a specialized prompt to:
- Remove irrelevant content (ads, navigation, boilerplate text)
- Extract main content useful for knowledge bases
- Ensure text coherence and structure
- Remove duplicates and redundant information
- Fix formatting issues
- Store results in LangChain `Document` format

## Features

- **LLM-Powered Cleaning**: Uses advanced language models to intelligently clean content
- **Document Format**: Outputs clean data in LangChain `Document` format with metadata
- **Flexible Input**: Accepts JSON data with various structures
- **Quality Filtering**: Automatically filters out low-quality or irrelevant content
- **Statistics**: Provides cleaning statistics and efficiency metrics
- **Persistence**: Save cleaned documents to JSON files

## Installation

Ensure you have the required dependencies:

```bash
pip install langchain langchain-core python-dotenv
```

## Environment Setup

Set your API key in a `.env` file or environment variable:

```bash
export TOGETHER_API_KEY="your_api_key_here"
```

## Usage

### Basic Usage

```python
from scripts.data_cleaning.data_cleaner import DataCleaner

# Sample unclean data
unclean_data = {
    "https://example.com/article": {
        "title": "Machine Learning Basics",
        "description": "Introduction to ML",
        "text_content": "ADVERTISEMENT: Buy now! Machine learning is..."
    }
}

# Initialize and clean
cleaner = DataCleaner(unclean_data)
cleaned_documents = cleaner.clean_data()

# Access cleaned documents
for doc in cleaned_documents:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### Advanced Usage

```python
# Get cleaning statistics
stats = cleaner.get_statistics()
print(f"Cleaning efficiency: {stats['cleaning_efficiency']}")

# Save cleaned documents
cleaner.save_cleaned_documents("output/cleaned_docs.json")

# Get documents (cleans if not already done)
documents = cleaner.get_cleaned_documents()
```

### Integration with RAG System

```python
from core.app import RAGApplication
from scripts.data_cleaning.data_cleaner import DataCleaner

# Clean your data
cleaner = DataCleaner(raw_data)
cleaned_docs = cleaner.clean_data()

# Add to RAG system
app = RAGApplication()
app.initialize(app_type="rag_chain", ...)

# Convert to text and metadata for RAG system
texts = [doc.page_content for doc in cleaned_docs]
metadatas = [doc.metadata for doc in cleaned_docs]
app.add_documents_from_text(texts, metadatas)
```

## Input Data Format

The DataCleaner accepts JSON data in the following formats:

### Format 1: URL-keyed with structured content
```json
{
    "https://example.com/page1": {
        "title": "Page Title",
        "description": "Page description",
        "text_content": "Main content here..."
    },
    "https://example.com/page2": {
        "title": "Another Title",
        "text_content": "More content..."
    }
}
```

### Format 2: Simple key-value pairs
```json
{
    "doc1": "Raw text content here...",
    "doc2": "More raw content..."
}
```

### Format 3: Mixed format
```json
{
    "structured_doc": {
        "content": "Structured content...",
        "metadata": {"source": "web"}
    },
    "simple_doc": "Simple text content..."
}
```

## Output Format

Cleaned documents are returned as LangChain `Document` objects:

```python
Document(
    page_content="Cleaned and relevant content...",
    metadata={
        "source": "original_key",
        "original_title": "Original Title",
        "original_description": "Original Description",
        "cleaned": True,
        "cleaning_method": "llm_chain"
    }
)
```

## Configuration

### Prompt Customization

You can customize the cleaning prompt by modifying the `_create_cleaning_prompt` method:

```python
class CustomDataCleaner(DataCleaner):
    def _create_cleaning_prompt(self) -> Prompt:
        template = """
        Custom cleaning instructions here...
        
        Document: {input}
        
        Return cleaned content:
        """
        return Prompt(template=template)
```

### LLM Configuration

Modify the LLM settings in the `_initialize_chain` method:

```python
llm = LLM(
    provider="chatopenai",
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key=api_key,
    temperature=0.3,  # Lower for more consistent cleaning
    max_tokens=2048   # Adjust based on content length
)
```

## Example Script

Run the example script to see the DataCleaner in action:

```bash
cd backend/scripts/data_cleaning
python example_usage.py
```

This will:
1. Create sample unclean data
2. Initialize the DataCleaner
3. Clean the documents
4. Display results and statistics
5. Save cleaned documents to a JSON file

## API Reference

### DataCleaner Class

#### `__init__(self, unclean_data: json)`
Initialize the DataCleaner with unclean data.

**Parameters:**
- `unclean_data`: JSON data to be cleaned (dict or JSON string)

#### `clean_data(self) -> List[Document]`
Clean the data and return Document objects.

**Returns:**
- List of cleaned LangChain Document objects

#### `get_cleaned_documents(self) -> List[Document]`
Get cleaned documents, cleaning if necessary.

**Returns:**
- List of cleaned Document objects

#### `save_cleaned_documents(self, output_path: str)`
Save cleaned documents to a JSON file.

**Parameters:**
- `output_path`: Path to save the cleaned documents

#### `get_statistics(self) -> Dict[str, Any]`
Get cleaning statistics.

**Returns:**
- Dictionary with cleaning metrics

## Error Handling

The DataCleaner includes robust error handling:

- **API Key Missing**: Clear error message if TOGETHER_API_KEY not set
- **Invalid Input**: Graceful handling of malformed JSON data
- **Processing Errors**: Individual document errors don't stop the entire process
- **Empty Content**: Automatic filtering of documents with no useful content

## Performance Considerations

- **Batch Processing**: Documents are processed individually for better error isolation
- **Content Filtering**: Low-quality content is automatically filtered out
- **Memory Efficient**: Processes documents one at a time
- **API Rate Limits**: Consider implementing delays for large datasets

## Best Practices

1. **Data Preprocessing**: Clean obvious formatting issues before using DataCleaner
2. **Batch Size**: Process large datasets in smaller batches
3. **Quality Control**: Review cleaned documents for quality
4. **Prompt Tuning**: Customize prompts for your specific domain
5. **Error Monitoring**: Monitor cleaning statistics for quality assessment

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   ValueError: TOGETHER_API_KEY environment variable not set
   ```
   **Solution**: Set the environment variable or add to .env file

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'models.rag_chain'
   ```
   **Solution**: Ensure you're running from the backend directory

3. **Empty Results**
   - Check if input data has meaningful content
   - Review the cleaning prompt for your use case
   - Verify API connectivity

4. **Memory Issues**
   - Process smaller batches
   - Use a lighter LLM model
   - Implement streaming for large documents

## Contributing

To contribute to the DataCleaner module:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

This module is part of the RAG system and follows the same license terms. 